"""
Tests for Block 2: KG types and the in-memory graph.

No Neo4j or PyTorch required.  All tests run against the in-memory layer
with neo4j=None and cfg.neo4j.enabled=False.
"""

import numpy as np
import pytest

from alkog.config.kg import EdgeType, KGConfig, Neo4jConfig
from alkog.kg.graph import KnowledgeGraph
from alkog.kg.types import KGEdge, KGNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> KGConfig:
    return KGConfig(
        node_embedding_dim=8,
        affordance_dim=4,
        edge_feature_dim=8,
        max_nodes=10,
        max_edges=20,
        redundancy_similarity_threshold=0.95,
        min_occurrences_for_stable_node=3,
        neo4j=Neo4jConfig(enabled=False),
    )


@pytest.fixture
def kg(cfg) -> KnowledgeGraph:
    return KnowledgeGraph(cfg=cfg, run_name="test_run", neo4j=None)


def rand_embedding(dim: int = 8) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def rand_affordance(dim: int = 4) -> np.ndarray:
    return np.random.rand(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# KGNode
# ---------------------------------------------------------------------------

class TestKGNode:
    def test_create_assigns_uuid(self):
        n = KGNode.create(rand_embedding(), rand_affordance(), episode=0)
        assert len(n.node_id) == 36  # UUID format

    def test_reinforce_updates_embedding(self):
        emb = rand_embedding()
        n = KGNode.create(emb.copy(), rand_affordance(), episode=0)
        old_emb = n.embedding.copy()
        new_emb = rand_embedding()
        n.reinforce(new_emb, rand_affordance(), episode=1, min_occurrences_for_stable=3)
        assert not np.allclose(n.embedding, old_emb)
        assert n.occurrence_count == 2

    def test_stability_gate(self):
        n = KGNode.create(rand_embedding(), rand_affordance(), episode=0)
        assert not n.is_stable
        for i in range(2):
            n.reinforce(rand_embedding(), rand_affordance(), episode=i + 1, min_occurrences_for_stable=3)
        assert n.is_stable

    def test_roundtrip_serialisation(self):
        n = KGNode.create(rand_embedding(), rand_affordance(), episode=5)
        n.label = "chair"
        d = n.to_dict()
        n2 = KGNode.from_dict(d)
        assert n2.node_id == n.node_id
        assert n2.label == "chair"
        assert np.allclose(n2.embedding, n.embedding)


# ---------------------------------------------------------------------------
# KGEdge
# ---------------------------------------------------------------------------

class TestKGEdge:
    def test_create(self):
        e = KGEdge.create("a", "b", EdgeType.CAUSAL, episode=0)
        assert e.edge_type == EdgeType.CAUSAL
        assert e.confidence == 1.0

    def test_decay(self):
        e = KGEdge.create("a", "b", EdgeType.SPATIAL, episode=0)
        e.decay(0.9)
        assert pytest.approx(e.confidence, abs=1e-5) == 0.9

    def test_reinforce_caps_at_one(self):
        e = KGEdge.create("a", "b", EdgeType.TEMPORAL, episode=0)
        e.reinforce(0.5, episode=1)
        assert e.confidence == 1.0

    def test_contradict_reduces_confidence(self):
        e = KGEdge.create("a", "b", EdgeType.CAUSAL, episode=0)
        e.contradict(0.3)
        assert pytest.approx(e.confidence, abs=1e-5) == 0.7

    def test_roundtrip_serialisation(self):
        e = KGEdge.create("src", "tgt", EdgeType.COMPOSITIONAL, episode=2)
        d = e.to_dict()
        e2 = KGEdge.from_dict(d)
        assert e2.edge_id == e.edge_id
        assert e2.edge_type == EdgeType.COMPOSITIONAL


# ---------------------------------------------------------------------------
# KnowledgeGraph — node operations
# ---------------------------------------------------------------------------

class TestKGNodeOperations:
    def test_propose_new_node(self, kg):
        emb = rand_embedding(8)
        aff = rand_affordance(4)
        result, _ = kg.propose_node(emb, aff)
        assert result.is_new
        assert kg.num_nodes == 1

    def test_redundant_proposal_matches_existing(self, kg):
        emb = rand_embedding(8)
        aff = rand_affordance(4)
        result1, _ = kg.propose_node(emb, aff)
        # Propose almost identical embedding (should match)
        noisy_emb = emb + np.random.randn(8).astype(np.float32) * 0.001
        noisy_emb /= np.linalg.norm(noisy_emb)
        result2, _ = kg.propose_node(noisy_emb, aff)
        assert not result2.is_new
        assert result2.node_id == result1.node_id
        assert kg.num_nodes == 1

    def test_orthogonal_proposals_create_distinct_nodes(self, kg):
        # Create two orthogonal embeddings — similarity ~ 0
        emb1 = np.zeros(8, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(8, dtype=np.float32)
        emb2[1] = 1.0
        kg.propose_node(emb1, rand_affordance(4))
        kg.propose_node(emb2, rand_affordance(4))
        assert kg.num_nodes == 2

    def test_max_nodes_prune(self, cfg):
        cfg = cfg.model_copy(update={"max_nodes": 3})
        kg = KnowledgeGraph(cfg=cfg, run_name="test", neo4j=None)
        # Add 4 distinct nodes
        for i in range(4):
            emb = np.zeros(8, dtype=np.float32)
            emb[i % 8] = 1.0
            kg.propose_node(emb, rand_affordance(4))
        assert kg.num_nodes <= 3

    def test_novel_reuse_flag(self, kg):
        emb = rand_embedding(8)
        result, _ = kg.propose_node(emb, rand_affordance(4))
        node_id = result.node_id
        # First check: should return True (novel, first reuse)
        assert kg.check_novel_reuse(node_id) is True
        # Second check: flag cleared, should return False
        assert kg.check_novel_reuse(node_id) is False


# ---------------------------------------------------------------------------
# KnowledgeGraph — edge operations
# ---------------------------------------------------------------------------

class TestKGEdgeOperations:
    def _add_two_nodes(self, kg) -> tuple[str, str]:
        emb1 = np.zeros(8, dtype=np.float32); emb1[0] = 1.0
        emb2 = np.zeros(8, dtype=np.float32); emb2[1] = 1.0
        r1, _ = kg.propose_node(emb1, rand_affordance(4))
        r2, _ = kg.propose_node(emb2, rand_affordance(4))
        return r1.node_id, r2.node_id

    def test_propose_edge(self, kg):
        a, b = self._add_two_nodes(kg)
        eid, is_new = kg.propose_edge(a, EdgeType.CAUSAL, b)
        assert is_new
        assert kg.num_edges == 1

    def test_reinforce_existing_edge(self, kg):
        a, b = self._add_two_nodes(kg)
        eid1, _ = kg.propose_edge(a, EdgeType.CAUSAL, b)
        eid2, is_new = kg.propose_edge(a, EdgeType.CAUSAL, b)
        assert not is_new
        assert eid1 == eid2
        assert kg.edges[eid1].confidence > 1.0 - 1e-5  # capped at 1.0

    def test_contradict_edge(self, kg):
        a, b = self._add_two_nodes(kg)
        eid, _ = kg.propose_edge(a, EdgeType.CAUSAL, b)
        kg.contradict_edge(a, EdgeType.CAUSAL, b)
        assert kg.edges[eid].confidence < 1.0

    def test_missing_node_raises(self, kg):
        emb = np.zeros(8, dtype=np.float32); emb[0] = 1.0
        r, _ = kg.propose_node(emb, rand_affordance(4))
        with pytest.raises(KeyError):
            kg.propose_edge(r.node_id, EdgeType.SPATIAL, "nonexistent")


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_edge_decay_on_episode_end(self, kg):
        emb1 = np.zeros(8, dtype=np.float32); emb1[0] = 1.0
        emb2 = np.zeros(8, dtype=np.float32); emb2[1] = 1.0
        r1, _ = kg.propose_node(emb1, rand_affordance(4))
        r2, _ = kg.propose_node(emb2, rand_affordance(4))
        eid, _ = kg.propose_edge(r1.node_id, EdgeType.TEMPORAL, r2.node_id)
        initial_conf = kg.edges[eid].confidence
        kg.on_episode_end()
        assert kg.edges[eid].confidence < initial_conf

    def test_pruned_edge_removed(self, cfg):
        cfg = cfg.model_copy(update={
            "edge_confidence_decay": 0.01,  # aggressive decay
            "edge_prune_confidence_threshold": 0.5,
        })
        kg = KnowledgeGraph(cfg=cfg, run_name="test", neo4j=None)
        emb1 = np.zeros(8, dtype=np.float32); emb1[0] = 1.0
        emb2 = np.zeros(8, dtype=np.float32); emb2[1] = 1.0
        r1, _ = kg.propose_node(emb1, rand_affordance(4))
        r2, _ = kg.propose_node(emb2, rand_affordance(4))
        kg.propose_edge(r1.node_id, EdgeType.CAUSAL, r2.node_id)
        kg.on_episode_end()
        assert kg.num_edges == 0

    def test_summary(self, kg):
        summary = kg.summary()
        assert "num_nodes" in summary
        assert "num_edges" in summary
        assert summary["episode"] == 0
