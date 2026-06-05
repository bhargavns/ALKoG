"""
Neo4j persistence layer for the Knowledge Graph.

Responsibilities
----------------
- Schema setup: constraints, the vector index for cosine similarity search
- Upsert / delete nodes and edges (called at episode sync)
- Vector similarity search for redundancy detection (delegates to Neo4j)
- Raw Cypher query interface for the agent's Phase-4 relational reasoning

Neo4j data model
----------------
  (:KGNode {node_id, run_name, embedding, affordance_vector, ...})
  -[:SPATIAL | :CAUSAL | :COMPOSITIONAL | :TEMPORAL
     {edge_id, confidence, ...}]->
  (:KGNode)

All nodes and edges carry a `run_name` property so multiple training runs
can coexist in the same database without collision.

Vector index
------------
Neo4j 5.11+ vector index on KGNode.embedding allows O(log n) approximate
nearest-neighbour queries in cosine space.  This is what we use for the
redundancy check instead of brute-force numpy scan — scales to large KGs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np

from alkog.config.kg import Neo4jConfig
from alkog.kg.types import KGEdge, KGNode

log = logging.getLogger(__name__)


class Neo4jStore:
    """
    Thin wrapper around the neo4j Python driver.

    The driver is created once and reused across all queries.
    Call close() when training ends.
    """

    def __init__(self, cfg: Neo4jConfig) -> None:
        self.cfg = cfg
        self._driver = None
        if cfg.enabled:
            self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "neo4j package not installed.  Run: pip install neo4j\n"
                "Or set kg.neo4j.enabled: false in your config to run in-memory only."
            )
        self._driver = GraphDatabase.driver(
            self.cfg.uri,
            auth=(self.cfg.username, self.cfg.password),
            connection_timeout=self.cfg.connection_timeout_seconds,
            max_connection_pool_size=self.cfg.max_connection_pool_size,
        )
        # Verify connectivity eagerly so config errors surface at startup
        self._driver.verify_connectivity()
        log.info(f"Neo4j connected: {self.cfg.uri} / db={self.cfg.database}")

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    @contextmanager
    def _session(self) -> Generator:
        if self._driver is None:
            raise RuntimeError("Neo4j store is not connected (enabled=False or not yet connected).")
        with self._driver.session(database=self.cfg.database) as session:
            yield session

    # ------------------------------------------------------------------
    # Schema setup — call once at the start of each run
    # ------------------------------------------------------------------

    def setup_schema(self, embedding_dim: int) -> None:
        """
        Create uniqueness constraints and the vector index.
        Safe to call repeatedly (uses IF NOT EXISTS).
        """
        with self._session() as s:
            # Uniqueness constraint on node_id
            s.run(
                "CREATE CONSTRAINT kg_node_id IF NOT EXISTS "
                "FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE"
            )
            # Uniqueness constraint on edge_id
            s.run(
                "CREATE CONSTRAINT kg_edge_id IF NOT EXISTS "
                "FOR ()-[r:RELATES]-() REQUIRE r.edge_id IS UNIQUE"
            )
            # Vector index for cosine similarity search on embeddings
            s.run(
                f"CREATE VECTOR INDEX {self.cfg.vector_index_name} IF NOT EXISTS "
                f"FOR (n:KGNode) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{"
                f"  `vector.dimensions`: {embedding_dim},"
                f"  `vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
        log.info("Neo4j schema ready.")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(self, node: KGNode, run_name: str) -> None:
        """Insert or update a KG node.  Uses MERGE on node_id."""
        props = node.to_dict()
        props["run_name"] = run_name
        with self._session() as s:
            s.run(
                """
                MERGE (n:KGNode {node_id: $node_id})
                SET n += $props
                """,
                node_id=node.node_id,
                props=props,
            )

    def delete_node(self, node_id: str, run_name: str) -> None:
        """Remove a node and all its relationships."""
        with self._session() as s:
            s.run(
                "MATCH (n:KGNode {node_id: $node_id, run_name: $run_name}) "
                "DETACH DELETE n",
                node_id=node_id,
                run_name=run_name,
            )

    def bulk_upsert_nodes(self, nodes: list[KGNode], run_name: str) -> None:
        """Batch upsert — more efficient than calling upsert_node in a loop."""
        rows = [{"props": {**n.to_dict(), "run_name": run_name}} for n in nodes]
        with self._session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MERGE (n:KGNode {node_id: row.props.node_id})
                SET n += row.props
                """,
                rows=rows,
            )

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def upsert_edge(self, edge: KGEdge, run_name: str) -> None:
        """Insert or update a relationship between two KG nodes."""
        props = edge.to_dict()
        props["run_name"] = run_name
        rel_type = edge.edge_type.value.upper()
        with self._session() as s:
            s.run(
                f"""
                MATCH (src:KGNode {{node_id: $src_id, run_name: $run_name}})
                MATCH (tgt:KGNode {{node_id: $tgt_id, run_name: $run_name}})
                MERGE (src)-[r:{rel_type} {{edge_id: $edge_id}}]->(tgt)
                SET r += $props
                """,
                src_id=edge.src_id,
                tgt_id=edge.tgt_id,
                edge_id=edge.edge_id,
                props=props,
                run_name=run_name,
            )

    def delete_edge(self, edge_id: str) -> None:
        with self._session() as s:
            s.run(
                "MATCH ()-[r {edge_id: $edge_id}]->() DELETE r",
                edge_id=edge_id,
            )

    def bulk_upsert_edges(self, edges: list[KGEdge], run_name: str) -> None:
        """Batch upsert edges.  Relationships are keyed by edge_id."""
        for edge in edges:
            # Dynamic rel type requires individual calls (Cypher limitation)
            self.upsert_edge(edge, run_name)

    # ------------------------------------------------------------------
    # Vector similarity search
    # ------------------------------------------------------------------

    def find_similar_nodes(
        self,
        embedding: np.ndarray,
        run_name: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar nodes to the given embedding.

        Uses the Neo4j vector index (cosine similarity).
        Returns a list of (node_id, similarity_score) sorted descending.

        Falls back to empty list if the index is not yet populated.
        """
        with self._session() as s:
            result = s.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{self.cfg.vector_index_name}', $top_k, $embedding
                )
                YIELD node, score
                WHERE node.run_name = $run_name
                RETURN node.node_id AS node_id, score
                ORDER BY score DESC
                """,
                top_k=top_k,
                embedding=embedding.tolist(),
                run_name=run_name,
            )
            return [(r["node_id"], r["score"]) for r in result]

    # ------------------------------------------------------------------
    # Cypher query interface (Phase 4 relational reasoning)
    # ------------------------------------------------------------------

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        Execute an arbitrary read-only Cypher query.

        This is the interface the agent uses in Phase 4 to answer relational
        questions: "what is on top of the red box?", "what did I push last?",
        etc.  The query should be scoped to the current run_name.

        Returns a list of record dicts.

        Example queries
        ---------------
        # What objects have I categorised as 'stackable'?
        "MATCH (n:KGNode {run_name: $run_name})
         WHERE n.affordance_vector[2] > 0.5
         RETURN n.node_id, n.label"

        # What objects are causally connected to node X?
        "MATCH (a:KGNode {node_id: $node_id})-[:CAUSAL]->(b:KGNode)
         WHERE a.run_name = $run_name
         RETURN b.node_id, b.label"
        """
        with self._session() as s:
            result = s.run(cypher, **(params or {}))
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Full snapshot load (for resuming a training run)
    # ------------------------------------------------------------------

    def load_snapshot(self, run_name: str) -> tuple[list[KGNode], list[KGEdge]]:
        """
        Load all nodes and edges for a given run from Neo4j.
        Used to restore KG state when resuming training from a checkpoint.
        """
        with self._session() as s:
            node_result = s.run(
                "MATCH (n:KGNode {run_name: $run_name}) RETURN properties(n) AS props",
                run_name=run_name,
            )
            nodes = [KGNode.from_dict(r["props"]) for r in node_result]

            edge_result = s.run(
                """
                MATCH ()-[r]-()
                WHERE r.run_name = $run_name
                RETURN properties(r) AS props
                """,
                run_name=run_name,
            )
            edges = [KGEdge.from_dict(r["props"]) for r in edge_result]

        log.info(f"Loaded snapshot: {len(nodes)} nodes, {len(edges)} edges (run={run_name})")
        return nodes, edges
