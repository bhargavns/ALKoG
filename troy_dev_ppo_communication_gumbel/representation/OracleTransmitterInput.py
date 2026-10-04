"""Simulator-state triples: no images, detector, classifier, or box relations.

All task objects are available regardless of visibility. Caged lions have an
exact (lion, inside, cage) relation; loose lions and their cage are unary nodes.
Identity is shared between caged and loose lions. Static world anchors are exact
and converted to current egocentric offsets by the existing input implementation.
"""
import torch

from representation.TransmitterInput import TransmitterInput, _merge_with_receiver_slot
from representation.Grounding import TRIPLE_PAD
from environment.KGWorldEnv import CAMERA_NAMES, POS_SCALE


class OracleTransmitterInput(TransmitterInput):
    def __init__(self, vocabulary, env, body_to_node, k_triples):
        required = 5 if env.resources else 3
        if k_triples < required:
            raise ValueError(f"oracle_state requires k_triples >= {required}")
        super().__init__(vocabulary, None, body_to_node['receiver'], k_triples)
        self.env = env
        self.body_to_node = body_to_node

    def refresh(self, obs):
        nodes = self.body_to_node
        entries = [(nodes['food'], TRIPLE_PAD, TRIPLE_PAD)]
        if self.env.lion_caged:
            entries.append((nodes['lion'], self.vocabulary.relation_index('inside'), nodes['cage']))
        else:
            entries.extend((nodes[name], TRIPLE_PAD, TRIPLE_PAD) for name in ('lion', 'cage'))
        if self.env.resources:
            entries.extend((nodes[name], TRIPLE_PAD, TRIPLE_PAD) for name in ('water', 'poison'))
        entries.sort()
        triples = torch.full((self.k_triples, 3), TRIPLE_PAD, dtype=torch.long)
        deltas = torch.zeros(self.k_triples, 2, 2)
        positions = {node: torch.as_tensor(
            self.env.model.body(name).pos[:2].copy() / POS_SCALE,
            dtype=torch.float32) for name, node in nodes.items()}
        agent = torch.as_tensor(obs[:2], dtype=torch.float32)
        for i, (src, rel, dst) in enumerate(entries):
            triples[i] = torch.tensor((src, rel, dst))
            deltas[i, 0] = positions[src] - agent
            if dst != TRIPLE_PAD:
                deltas[i, 1] = positions[dst] - agent
        self.triples, self.anchors, self.mask = _merge_with_receiver_slot(
            self.mem, triples, deltas, obs[:2], self.receiver_node)
        # Optional videos can render the scene, but have no perception overlays.
        self.boxes = [{} for _ in CAMERA_NAMES]
        self.relations = [[] for _ in CAMERA_NAMES]
        return self.boxes, self.relations
