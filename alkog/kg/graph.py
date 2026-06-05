"""
KnowledgeGraph — the agent's structured world model.

This class is the single entry point for all KG operations during training.
It maintains an in-memory graph for the hot path (every training step) and
delegates persistence to a Neo4jStore at episode boundaries.

Thread safety
-------------
KnowledgeGraph is NOT thread-safe.  Each parallel environment worker
should have its own KG instance.  Cross-episode aggregation (merging KGs
from multiple workers) is handled by the training coordinator in Block 7.

Hot path operations (called every step)
----------------------------------------
  propose_node()      O(n) cosine scan over all current node embeddings.
                      If Neo4j is enabled, this also queries the vector
                      index — but for small KGs (< 128 nodes) the numpy
                      scan is faster and preferred.  The Neo4j vector
                      search is used when the KG exceeds a size threshold.

  reinforce_node()    O(1)
  propose_edge()      O(1) lookup + insert
  to_pyg()            O(V + E) — called once per step for the GNN encoder.
                      Cached between steps; cache is invalidated on any write.

Episode-boundary operations
----------------------------
  on_episode_end()    Applies edge confidence decay, prunes dead nodes/edges,
                      and (if sync is due) flushes to Neo4j.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from alkog.config.kg import EdgeType, KGConfig
from alkog.kg.types import KGEdge, KGNode

if TYPE_CHECKING:
    from alkog.kg.neo4j_store import Neo4jStore

    try:
        from torch_geometric.data import Data  # type: ignore[import]
    except ImportError:
        Data = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

# Size threshold above which we delegate similarity search to Neo4j vector index
# instead of the local numpy scan.
_NEO4J_SEARCH_THRESHOLD = 64


class NodeProposalResult:
    """Returned by propose_node() so the caller has all reward-relevant info."""

    __slots__ = ("node_id", "is_new", "matched_similarity")

    def __init__(self, node_id: str, is_new: bool, matched_similarity: float) -> None:
        self.node_id = node_id
        self.is_new = is_new
        self.matched_similarity = matched_similarity
        """
        If is_new=False, this is the cosine similarity to the matched node.
        If is_new=True, this is the similarity to the closest existing node
        (which was below the redundancy threshold).
        """


class KnowledgeGraph:
    def __init__(
        self,
        cfg: KGConfig,
        run_name: str,
        neo4j: "Neo4jStore | None" = None,
        current_episode: int = 0,
    ) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self._neo4j = neo4j
        self.current_episode = current_episode

        # Core in-memory stores
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[str, KGEdge] = {}

        # For to_pyg() caching
        self._pyg_cache: "Data | None" = None
        self._pyg_dirty: bool = True

        # Track novel nodes pending their first reuse (for novel_abstraction reward)
        self._pending_novel_nodes: set[str] = set()

        # Episode sync counter
        self._episodes_since_last_sync: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def nodes(self) -> dict[str, KGNode]:
        return self._nodes

    @property
    def edges(self) -> dict[str, KGEdge]:
        return self._edges

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def propose_node(
        self,
        embedding: np.ndarray,
        affordance_vector: np.ndarray,
    ) -> tuple[NodeProposalResult, float]:
        """
        Propose a new bounding box region as a KG node.

        Returns
        -------
        result : NodeProposalResult
            node_id     — ID of the matched or newly created node
            is_new      — True if a new node was created
            matched_similarity — cosine similarity to closest existing node

        reward_delta : float
            The reward signal to feed back to the RL agent.
            Positive for new productive nodes; negative for redundant ones.
            The caller is responsible for accumulating this into the step reward.
        """
        embedding = embedding.astype(np.float32)
        affordance_vector = affordance_vector.astype(np.float32)

        best_id, best_sim = self._find_most_similar(embedding)

        if best_id is not None and best_sim >= self.cfg.redundancy_similarity_threshold:
            # --- Redundant proposal: match to existing node ---
            node = self._nodes[best_id]
            node.reinforce(
                embedding,
                affordance_vector,
                self.current_episode,
                self.cfg.min_occurrences_for_stable_node,
            )
            self._pyg_dirty = True

            # Check if this was a pending novel node — first reuse!
            novel_reward = 0.0
            if best_id in self._pending_novel_nodes:
                self._pending_novel_nodes.discard(best_id)
                novel_reward = 0.0  # novel_abstraction reward applied by caller from cfg

            result = NodeProposalResult(node_id=best_id, is_new=False, matched_similarity=best_sim)
            return result, self.cfg.node_embedding_dim * 0.0  # no direct reward here

        # --- New node ---
        if len(self._nodes) >= self.cfg.max_nodes:
            self._prune_nodes()

        new_node = KGNode.create(
            embedding=embedding,
            affordance_vector=affordance_vector,
            episode=self.current_episode,
        )
        self._nodes[new_node.node_id] = new_node
        self._pending_novel_nodes.add(new_node.node_id)
        self._pyg_dirty = True

        result = NodeProposalResult(
            node_id=new_node.node_id,
            is_new=True,
            matched_similarity=best_sim if best_sim is not None else 0.0,
        )
        return result, 0.0  # novel_abstraction reward is deferred to first reuse

    def reinforce_node(
        self,
        node_id: str,
        embedding: np.ndarray,
        affordance_vector: np.ndarray,
    ) -> bool:
        """
        Explicitly reinforce an existing node (e.g., agent revisits a known object).
        Returns True if the node exists, False otherwise.
        """
        if node_id not in self._nodes:
            return False
        self._nodes[node_id].reinforce(
            embedding,
            affordance_vector,
            self.current_episode,
            self.cfg.min_occurrences_for_stable_node,
        )
        self._pyg_dirty = True
        return True

    def check_novel_reuse(self, node_id: str) -> bool:
        """
        Check whether this node should trigger the novel_abstraction reward.
        Called when the agent successfully uses a node to complete a task.
        Clears the pending flag so the reward is only given once.
        """
        if node_id in self._pending_novel_nodes:
            self._pending_novel_nodes.discard(node_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def propose_edge(
        self,
        src_id: str,
        edge_type: EdgeType,
        tgt_id: str,
        spatial_subtype: str | None = None,
    ) -> tuple[str, bool]:
        """
        Add or reinforce a directed edge between two existing nodes.

        Returns (edge_id, is_new).

        Raises KeyError if either src_id or tgt_id does not exist.
        """
        if src_id not in self._nodes:
            raise KeyError(f"Source node '{src_id}' not found in KG.")
        if tgt_id not in self._nodes:
            raise KeyError(f"Target node '{tgt_id}' not found in KG.")

        # Check for existing edge of the same type between these nodes
        existing = self._find_edge(src_id, edge_type, tgt_id)
        if existing is not None:
            existing.reinforce(self.cfg.edge_confidence_boost, self.current_episode)
            self._pyg_dirty = True
            return existing.edge_id, False

        if len(self._edges) >= self.cfg.max_edges:
            self._prune_edges()

        new_edge = KGEdge.create(
            src_id=src_id,
            tgt_id=tgt_id,
            edge_type=edge_type,
            episode=self.current_episode,
            spatial_subtype=spatial_subtype,
        )
        self._edges[new_edge.edge_id] = new_edge
        self._pyg_dirty = True
        return new_edge.edge_id, True

    def contradict_edge(
        self,
        src_id: str,
        edge_type: EdgeType,
        tgt_id: str,
        contradiction_penalty: float = 0.3,
    ) -> bool:
        """
        Called when an observed interaction contradicts a KG prediction.
        Reduces the edge's confidence.  Returns True if the edge was found.
        """
        existing = self._find_edge(src_id, edge_type, tgt_id)
        if existing is None:
            return False
        existing.contradict(contradiction_penalty, self.current_episode)
        self._pyg_dirty = True
        return True

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_episode_end(self) -> dict[str, int]:
        """
        Called at the end of each episode.

        1. Applies edge confidence decay.
        2. Prunes dead edges (confidence < threshold).
        3. Prunes low-occurrence nodes if over the cap.
        4. Syncs to Neo4j if due.

        Returns a summary dict for logging.
        """
        self.current_episode += 1
        pruned_edges = self._decay_and_prune_edges()
        pruned_nodes = self._prune_nodes() if len(self._nodes) > self.cfg.max_nodes else 0

        self._episodes_since_last_sync += 1
        synced = False
        if (
            self._neo4j is not None
            and self._episodes_since_last_sync >= self.cfg.neo4j.sync_every_n_episodes
        ):
            self.sync_to_neo4j()
            self._episodes_since_last_sync = 0
            synced = True

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "pruned_nodes": pruned_nodes,
            "pruned_edges": pruned_edges,
            "synced_to_neo4j": int(synced),
        }

    # ------------------------------------------------------------------
    # Neo4j sync
    # ------------------------------------------------------------------

    def sync_to_neo4j(self) -> None:
        """Flush the full in-memory KG to Neo4j."""
        if self._neo4j is None:
            return
        self._neo4j.bulk_upsert_nodes(list(self._nodes.values()), self.run_name)
        self._neo4j.bulk_upsert_edges(list(self._edges.values()), self.run_name)
        log.debug(
            f"Neo4j sync: {self.num_nodes} nodes, {self.num_edges} edges "
            f"(episode {self.current_episode})"
        )

    @classmethod
    def from_neo4j_snapshot(
        cls,
        cfg: KGConfig,
        run_name: str,
        neo4j: "Neo4jStore",
        current_episode: int = 0,
    ) -> "KnowledgeGraph":
        """Restore a KG from a Neo4j snapshot (for resuming training)."""
        kg = cls(cfg=cfg, run_name=run_name, neo4j=neo4j, current_episode=current_episode)
        nodes, edges = neo4j.load_snapshot(run_name)
        kg._nodes = {n.node_id: n for n in nodes}
        kg._edges = {e.edge_id: e for e in edges}
        kg._pyg_dirty = True
        log.info(f"Restored KG: {kg.num_nodes} nodes, {kg.num_edges} edges")
        return kg

    # ------------------------------------------------------------------
    # PyTorch Geometric export (for the GNN encoder)
    # ------------------------------------------------------------------

    def to_pyg(self) -> "Data":
        """
        Convert the in-memory KG to a PyTorch Geometric Data object.

        This is called every training step to feed the GNN encoder.
        The result is cached and only recomputed when the graph has changed.

        Returns an empty graph (no nodes) if the KG is empty.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError(
                "torch and torch_geometric are required to call to_pyg(). "
                "Install with: pip install torch torch-geometric"
            )

        if not self._pyg_dirty and self._pyg_cache is not None:
            return self._pyg_cache

        if not self._nodes:
            self._pyg_cache = Data(
                x=torch.zeros((0, self.cfg.node_embedding_dim + self.cfg.affordance_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, self.cfg.edge_feature_dim)),
            )
            self._pyg_dirty = False
            return self._pyg_cache

        # Build node feature matrix: [embedding | affordance_vector]
        node_ids = list(self._nodes.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        x_parts = [
            np.concatenate([n.embedding, n.affordance_vector])
            for n in self._nodes.values()
        ]
        x = torch.tensor(np.stack(x_parts), dtype=torch.float32)

        # Build edge index and edge attributes
        # Edge type is encoded as a one-hot vector of length 4
        edge_type_to_idx = {et: i for i, et in enumerate(EdgeType)}
        edge_index_list: list[list[int]] = [[], []]
        edge_attr_list: list[np.ndarray] = []

        for edge in self._edges.values():
            if edge.src_id not in node_idx or edge.tgt_id not in node_idx:
                continue  # dangling edge (pruned node not yet cleaned up)
            edge_index_list[0].append(node_idx[edge.src_id])
            edge_index_list[1].append(node_idx[edge.tgt_id])

            one_hot = np.zeros(len(EdgeType), dtype=np.float32)
            one_hot[edge_type_to_idx[edge.edge_type]] = 1.0
            attr = np.concatenate([one_hot, [edge.confidence]])
            # Pad to edge_feature_dim
            if len(attr) < self.cfg.edge_feature_dim:
                attr = np.pad(attr, (0, self.cfg.edge_feature_dim - len(attr)))
            edge_attr_list.append(attr[: self.cfg.edge_feature_dim])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = (
            torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)
            if edge_attr_list
            else torch.zeros((0, self.cfg.edge_feature_dim))
        )

        self._pyg_cache = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self._pyg_dirty = False
        return self._pyg_cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_most_similar(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """
        Find the existing node most similar (cosine) to the given embedding.
        Returns (node_id, similarity) or (None, 0.0) if the KG is empty.

        For KGs above _NEO4J_SEARCH_THRESHOLD nodes, delegates to Neo4j
        vector index.  Below the threshold, uses a numpy scan (faster for
        small graphs since it avoids the network round-trip).
        """
        if not self._nodes:
            return None, 0.0

        if (
            self._neo4j is not None
            and self.cfg.neo4j.enabled
            and self.num_nodes >= _NEO4J_SEARCH_THRESHOLD
        ):
            results = self._neo4j.find_similar_nodes(embedding, self.run_name, top_k=1)
            if results:
                return results[0]
            return None, 0.0

        # Numpy scan
        embeddings = np.stack([n.embedding for n in self._nodes.values()])
        node_ids = list(self._nodes.keys())

        # Normalise
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        similarities = embeddings / norms @ query_norm

        best_idx = int(np.argmax(similarities))
        return node_ids[best_idx], float(similarities[best_idx])

    def _find_edge(
        self, src_id: str, edge_type: EdgeType, tgt_id: str
    ) -> KGEdge | None:
        for edge in self._edges.values():
            if edge.src_id == src_id and edge.tgt_id == tgt_id and edge.edge_type == edge_type:
                return edge
        return None

    def _decay_and_prune_edges(self) -> int:
        """Decay all edge confidences; remove those below the prune threshold."""
        to_prune = []
        for edge in self._edges.values():
            edge.decay(self.cfg.edge_confidence_decay)
            if edge.confidence < self.cfg.edge_prune_confidence_threshold:
                to_prune.append(edge.edge_id)

        if to_prune and self._neo4j is not None:
            for eid in to_prune:
                self._neo4j.delete_edge(eid)

        for eid in to_prune:
            del self._edges[eid]

        if to_prune:
            self._pyg_dirty = True

        return len(to_prune)

    def _prune_nodes(self) -> int:
        """
        Remove low-occurrence nodes to stay within max_nodes.

        Prunes to max_nodes - 1 so that when called from propose_node
        (before the new node is inserted) there is room for the new node.
        """
        target = self.cfg.max_nodes - 1
        if len(self._nodes) <= target:
            return 0

        # Sort by occurrence count ascending (fewest occurrences pruned first)
        candidates = sorted(
            [n for n in self._nodes.values() if not n.is_stable],
            key=lambda n: (n.occurrence_count, n.episode_last_seen),
        )
        to_prune_count = len(self._nodes) - target
        pruned = 0

        for node in candidates[:to_prune_count]:
            node_id = node.node_id
            # Remove all edges involving this node
            dead_edges = [
                eid
                for eid, e in self._edges.items()
                if e.src_id == node_id or e.tgt_id == node_id
            ]
            for eid in dead_edges:
                del self._edges[eid]

            if self._neo4j is not None:
                self._neo4j.delete_node(node_id, self.run_name)

            del self._nodes[node_id]
            self._pending_novel_nodes.discard(node_id)
            pruned += 1

        if pruned:
            self._pyg_dirty = True

        return pruned

    # ------------------------------------------------------------------
    # Debug / evaluation helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        stable = sum(1 for n in self._nodes.values() if n.is_stable)
        bound = sum(1 for n in self._nodes.values() if n.bound_token_id is not None)
        edge_type_counts = {et.value: 0 for et in EdgeType}
        for e in self._edges.values():
            edge_type_counts[e.edge_type.value] += 1
        return {
            "episode": self.current_episode,
            "num_nodes": self.num_nodes,
            "stable_nodes": stable,
            "token_bound_nodes": bound,
            "num_edges": self.num_edges,
            "edge_types": edge_type_counts,
            "pending_novel": len(self._pending_novel_nodes),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(run='{self.run_name}', "
            f"nodes={self.num_nodes}, edges={self.num_edges}, "
            f"episode={self.current_episode})"
        )
