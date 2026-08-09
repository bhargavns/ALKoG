# ALKoG — Active Learning Knowledge Graph Framework

The five-object soft-category perception/PPO experiment is documented in
[`bhargav_dev_ppo/SOFT_CATEGORY_QUICKSTART.md`](bhargav_dev_ppo/SOFT_CATEGORY_QUICKSTART.md).

A simulated knowledge graph-based active learning framework for studying emergent symbolic grounding in multi-agent RL environments.

Agents in a procedurally generated 2D/3D world learn to:
1. **Perceive** — propose productive bounding boxes over their visual field
2. **Represent** — build a structured knowledge graph of entity relationships
3. **Communicate** — develop a shared token vocabulary through interaction with a partner agent
4. **Reason** — answer relational queries using the accumulated KG

## Architecture overview

```
alkog/
├── config/         Block 1 -  Pydantic configs (rewards, KG, agent, training, env)
├── kg/             Block 2 -  Knowledge graph module (nodes, edges, NetworkX backend)
├── agents/         Block 3 -  Token vocabulary + symbol binding
│                   Block 5 -  Neural architecture (visual encoder, GNN, policy)
├── training/       Block 6 -  Reward functions
│                   Block 7 -  PPO training loop + curriculum manager
├── environment/    Block 8 -  Mock Gymnasium environment (no Godot required)
├── evaluation/     Block 9 -  KG Gini impurity, communication success metrics
└── utils/          Logging, W&B wrapper
```

Godot integration (Block 10–13) adds the real simulation as a drop-in replacement
for the mock environment via a WebSocket bridge.

## Setup

```bash
# Requires Python 3.11+
pip install poetry
poetry install

# Validate the config loads correctly:
python scripts/train.py --dry-run

# Run with a custom config:
python scripts/train.py --config configs/default.yaml --run-name my_experiment
```

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| RL algorithm | PPO → GRPO | PPO first for stability; GRPO later for KG reasoning tasks |
| Visual encoder | ResNet18 pretrained | Fast convergence on early curriculum; upgrade to ResNet50 for Phase 3-4 |
| KG edge types | Pre-defined enum | Keeps the relational space tractable; 4 types cover the proposal's examples |
| Agent communication | Finite token vocab, no pre-assigned meanings | Mirrors the proposal's emergent-language design |
| Simulation backend | Godot 4 via WebSocket | Swappable with mock env; identical Python interface either way |

## Reward structure

| Signal | Value | When |
|---|---|---|
| Task completion | +1.0 | Episode objective reached |
| Communication success | +0.5 | Partner takes correct action after utterance |
| Predictive accuracy | +0.2 | Agent correctly predicts interaction outcome |
| Novel abstraction | +0.1 | New KG node reused in a future episode |
| Time step cost | −0.01 | Every step |
| Failed interaction | −0.3 | Touching dangerous object / falling |
| Communication failure | −0.2 | Partner takes wrong action after utterance |
| Redundant abstraction | −0.1 | New node too similar to existing node |
| KG inconsistency | −0.15 | KG prediction contradicted by observation |

## Curriculum

| Phase | Agents | Objects | Primary reward driver |
|---|---|---|---|
| 1 — Solo Exploration | 1 | 2-3 | Navigation |
| 2 — Affordance Discovery | 1 | 5-8 | Interaction |
| 3 — Partner Communication | 2 | 5-8 | Communication success |
| 4 — Compositional Reasoning | 2 | 8-15 | Relational tasks |
