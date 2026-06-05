"""
Core data types for the Knowledge Graph.

KGNode and KGEdge are plain dataclasses — no torch or neo4j dependency here.
They are the shared currency between the in-memory graph, the GNN encoder,
and the Neo4j store.

Node embedding update rule
--------------------------
When the agent matches an existing node (cosine similarity > threshold),
the stored embedding is updated with an exponential moving average:

    embedding = (1 - ema_alpha) * embedding + ema_alpha * new_embedding

This lets the node's visual representation drift gradually toward the
current visual encoder's output, handling slow representation shift
during training without creating redundant nodes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from alkog.config.kg import EdgeType


# ---------------------------------------------------------------------------
# KGNode
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    """
    A single node in the knowledge graph — a grounded perceptual category.

    Created when the agent proposes a productive bounding box that does not
    match any existing node (cosine similarity below redundancy_threshold).
    """

    node_id: str
    """UUID string, unique within a run."""

    embedding: np.ndarray
    """
    Visual/semantic embedding of this category.
    Shape: (node_embedding_dim,).  dtype: float32.
    Updated via EMA when the node is reinforced.
    """

    affordance_vector: np.ndarray
    """
    Binary or soft vector encoding what the agent can do with this object
    category (pushable, stackable, dangerous, edible, ...).
    Shape: (affordance_dim,).  dtype: float32.
    Updated when the agent interacts with instances of this category.
    """

    occurrence_count: int = 0
    """
    Total number of times the agent has matched this node across all episodes.
    Used for stability gating and pruning priority.
    """

    episode_first_seen: int = 0
    """Episode index when this node was created."""

    episode_last_seen: int = 0
    """Episode index when this node was last matched."""

    is_stable: bool = False
    """
    True once occurrence_count >= cfg.min_occurrences_for_stable_node.
    Stable nodes are protected from pruning and eligible for symbol binding.
    """

    label: str | None = None
    """
    Optional human-readable label for debugging and evaluation.
    Not used by the agent — assigned externally by the evaluator or the
    Neo4j browser for visualisation.
    """

    bound_token_id: int | None = None
    """
    Index of the vocabulary token currently most strongly bound to this node.
    None until symbol binding has produced a stable mapping.
    """

    ema_alpha: float = 0.1
    """EMA step size for embedding updates on reinforcement."""

    @classmethod
    def create(
        cls,
        embedding: np.ndarray,
        affordance_vector: np.ndarray,
        episode: int,
        ema_alpha: float = 0.1,
    ) -> "KGNode":
        """Factory: create a new node with a fresh UUID."""
        return cls(
            node_id=str(uuid.uuid4()),
            embedding=embedding.astype(np.float32).copy(),
            affordance_vector=affordance_vector.astype(np.float32).copy(),
            occurrence_count=1,
            episode_first_seen=episode,
            episode_last_seen=episode,
            ema_alpha=ema_alpha,
        )

    def reinforce(
        self,
        new_embedding: np.ndarray,
        new_affordance: np.ndarray,
        episode: int,
        min_occurrences_for_stable: int = 3,
    ) -> None:
        """
        Called when the agent matches this node in a new observation.
        Updates the embedding via EMA and increments occurrence_count.
        """
        self.embedding = (
            (1.0 - self.ema_alpha) * self.embedding
            + self.ema_alpha * new_embedding.astype(np.float32)
        )
        # Soft update for affordances too (new interactions may refine them)
        self.affordance_vector = (
            (1.0 - self.ema_alpha) * self.affordance_vector
            + self.ema_alpha * new_affordance.astype(np.float32)
        )
        self.occurrence_count += 1
        self.episode_last_seen = episode
        if self.occurrence_count >= min_occurrences_for_stable:
            self.is_stable = True

    def to_dict(self) -> dict:
        """Serialise to a plain dict for Neo4j storage."""
        return {
            "node_id": self.node_id,
            "embedding": self.embedding.tolist(),
            "affordance_vector": self.affordance_vector.tolist(),
            "occurrence_count": self.occurrence_count,
            "episode_first_seen": self.episode_first_seen,
            "episode_last_seen": self.episode_last_seen,
            "is_stable": self.is_stable,
            "label": self.label,
            "bound_token_id": self.bound_token_id,
            "ema_alpha": self.ema_alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KGNode":
        """Deserialise from a Neo4j node property dict."""
        return cls(
            node_id=d["node_id"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            affordance_vector=np.array(d["affordance_vector"], dtype=np.float32),
            occurrence_count=d["occurrence_count"],
            episode_first_seen=d["episode_first_seen"],
            episode_last_seen=d["episode_last_seen"],
            is_stable=d["is_stable"],
            label=d.get("label"),
            bound_token_id=d.get("bound_token_id"),
            ema_alpha=d.get("ema_alpha", 0.1),
        )


# ---------------------------------------------------------------------------
# KGEdge
# ---------------------------------------------------------------------------

@dataclass
class KGEdge:
    """
    A directed relationship between two KG nodes.

    Confidence decays each episode without reinforcement, and is boosted
    when an interaction confirms the relationship.  Edges that drop below
    the prune threshold are removed.
    """

    edge_id: str
    """UUID string, unique within a run."""

    src_id: str
    """node_id of the source node."""

    tgt_id: str
    """node_id of the target node."""

    edge_type: EdgeType
    """One of: spatial, causal, compositional, temporal."""

    confidence: float = 1.0
    """
    Current belief strength in [0, 1].
    Initialised to 1.0 on creation; decays multiplicatively each episode.
    """

    episode_first_seen: int = 0
    episode_last_seen: int = 0

    reinforcement_count: int = 0
    """Number of times this edge has been confirmed by an interaction."""

    contradiction_count: int = 0
    """Number of times this edge has been contradicted by an interaction."""

    spatial_subtype: str | None = None
    """
    Free-form spatial qualifier for SPATIAL edges only.
    Examples: 'on_top_of', 'next_to', 'inside', 'below'.
    Not used by non-spatial edge types.
    """

    @classmethod
    def create(
        cls,
        src_id: str,
        tgt_id: str,
        edge_type: EdgeType,
        episode: int,
        spatial_subtype: str | None = None,
    ) -> "KGEdge":
        """Factory: create a new edge with a fresh UUID."""
        return cls(
            edge_id=str(uuid.uuid4()),
            src_id=src_id,
            tgt_id=tgt_id,
            edge_type=edge_type,
            confidence=1.0,
            episode_first_seen=episode,
            episode_last_seen=episode,
            spatial_subtype=spatial_subtype,
        )

    def reinforce(self, boost: float, episode: int) -> None:
        self.confidence = min(1.0, self.confidence + boost)
        self.reinforcement_count += 1
        self.episode_last_seen = episode

    def contradict(self, penalty: float = 0.3, episode: int = 0) -> None:
        self.confidence = max(0.0, self.confidence - penalty)
        self.contradiction_count += 1
        self.episode_last_seen = episode

    def decay(self, factor: float) -> None:
        """Apply per-episode multiplicative decay."""
        self.confidence *= factor

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "src_id": self.src_id,
            "tgt_id": self.tgt_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "episode_first_seen": self.episode_first_seen,
            "episode_last_seen": self.episode_last_seen,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "spatial_subtype": self.spatial_subtype,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KGEdge":
        return cls(
            edge_id=d["edge_id"],
            src_id=d["src_id"],
            tgt_id=d["tgt_id"],
            edge_type=EdgeType(d["edge_type"]),
            confidence=d["confidence"],
            episode_first_seen=d["episode_first_seen"],
            episode_last_seen=d["episode_last_seen"],
            reinforcement_count=d["reinforcement_count"],
            contradiction_count=d["contradiction_count"],
            spatial_subtype=d.get("spatial_subtype"),
        )
