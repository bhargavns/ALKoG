"""
Configuration for the Knowledge Graph module.

Storage architecture
--------------------
The KG runs on two layers that serve different purposes:

  In-memory cache   Used every training step.  Holds all current nodes and
                    edges as numpy arrays + dicts.  The GNN encoder reads
                    from here via to_pyg().  Writes are O(1).

  Neo4j store       Persistent graph database.  Synced from the in-memory
                    cache at episode boundaries (or every N episodes).
                    Provides:
                      - Durable storage so training can be resumed
                      - Native vector index for fast cosine similarity search
                        (used for redundancy detection and node matching)
                      - Cypher query interface for the agent's relational
                        reasoning tasks in Phase 4

Neo4j requirements
------------------
Neo4j 5.11+ (Community or Enterprise).  The vector index feature used for
node similarity search requires >= 5.11.

Quickstart with Docker:
  docker run -d \
    --name alkog-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/alkog_password \
    -e NEO4J_PLUGINS='["apoc"]' \
    neo4j:5.20-community
"""

from enum import Enum

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    """
    Pre-defined relationship types.  Keeping this closed keeps the edge
    space tractable, and maps cleanly to natural-language prepositions and
    verbs (Stage 3 symbol binding).
    """
    SPATIAL       = "spatial"        # "A is on top of / next to B"
    CAUSAL        = "causal"         # "Pushing A causes B to move"
    COMPOSITIONAL = "compositional"  # "A is part of B" (plank + wheels = skateboard)
    TEMPORAL      = "temporal"       # "A appears after B"


class Neo4jConfig(BaseModel):
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Bolt URI of the Neo4j instance.",
    )
    username: str = Field(default="neo4j")
    password: str = Field(default="alkog_password")
    database: str = Field(
        default="neo4j",
        description="Neo4j database name.  'neo4j' is the default database in Community edition.",
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Set False to run in-memory only (no Neo4j required). "
            "Useful for unit tests and quick experiments."
        ),
    )
    sync_every_n_episodes: int = Field(
        default=1,
        gt=0,
        description=(
            "Flush the in-memory KG to Neo4j every N episodes. "
            "Higher values reduce DB write pressure but lose more state on crash."
        ),
    )
    vector_index_name: str = Field(
        default="node_embeddings",
        description="Name of the Neo4j vector index used for cosine similarity search.",
    )
    connection_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Driver connection timeout.",
    )
    max_connection_pool_size: int = Field(
        default=10,
        gt=0,
        description="Neo4j driver connection pool size.",
    )


class KGConfig(BaseModel):
    # ------------------------------------------------------------------
    # Embedding dimensions
    # ------------------------------------------------------------------
    node_embedding_dim: int = Field(
        default=256,
        gt=0,
        description="Dimensionality of each node's visual/semantic embedding.",
    )
    affordance_dim: int = Field(
        default=32,
        gt=0,
        description=(
            "Dimensionality of the affordance vector attached to each node. "
            "Encodes what the agent can do with this category of object."
        ),
    )
    edge_feature_dim: int = Field(
        default=64,
        gt=0,
        description="Dimensionality of edge feature vectors (used by the GNN encoder).",
    )

    # ------------------------------------------------------------------
    # Graph size caps
    # ------------------------------------------------------------------
    max_nodes: int = Field(
        default=128,
        gt=0,
        description="Hard cap on number of KG nodes. Oldest low-confidence nodes are pruned.",
    )
    max_edges: int = Field(
        default=512,
        gt=0,
        description="Hard cap on number of KG edges.",
    )

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------
    redundancy_similarity_threshold: float = Field(
        default=0.92,
        gt=0.0,
        le=1.0,
        description=(
            "Cosine similarity above which a proposed new node is considered "
            "redundant with an existing one.  Triggers the redundant_abstraction penalty."
        ),
    )
    min_occurrences_for_stable_node: int = Field(
        default=3,
        gt=0,
        description=(
            "A node must be observed (matched) at least this many times "
            "before it is treated as a stable category."
        ),
    )
    node_prune_occurrence_threshold: int = Field(
        default=1,
        gt=0,
        description=(
            "Nodes with fewer total occurrences than this are candidates for "
            "pruning when max_nodes is reached."
        ),
    )

    # ------------------------------------------------------------------
    # Edge lifecycle
    # ------------------------------------------------------------------
    edge_confidence_decay: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplicative decay applied to an edge's confidence score each "
            "episode it is not reinforced by an observed interaction."
        ),
    )
    edge_prune_confidence_threshold: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Edges below this confidence are pruned from the graph.",
    )
    edge_confidence_boost: float = Field(
        default=0.15,
        gt=0.0,
        description="Amount added to edge confidence on a confirming observation.",
    )

    # ------------------------------------------------------------------
    # GNN encoder
    # ------------------------------------------------------------------
    num_gnn_layers: int = Field(
        default=3,
        gt=0,
        description="Number of message-passing layers in the KG encoder GNN.",
    )
    gnn_hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dimension inside each GNN layer.",
    )
    graph_embedding_dim: int = Field(
        default=512,
        gt=0,
        description=(
            "Output dimensionality of the graph-level embedding produced by "
            "the GNN encoder (via mean pooling over node embeddings)."
        ),
    )

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
