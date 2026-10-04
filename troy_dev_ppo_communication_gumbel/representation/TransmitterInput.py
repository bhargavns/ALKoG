"""Visual scene triples with persistent anchors and a live receiver slot."""
import torch
from representation.Grounding import TRIPLE_PAD, AnchorMemory, assemble_triples_geo, deltas_from_anchors


def strip_concept(per_frame, node_id):
    """Drop one concept, and every relation touching it, from a perception pass.

    The receiver is fed to the transmitter as an oracle position instead, so it
    must not also arrive through SAM -- otherwise it would occupy a second slot
    carrying a stale anchor, and could form relations that are stale too.
    """
    out = []
    for boxes, sims, deltas, relations in per_frame:
        out.append(
            (
                {c: b for c, b in boxes.items() if c != node_id},
                {c: s for c, s in sims.items() if c != node_id},
                {c: d for c, d in deltas.items() if c != node_id},
                [r for r in relations if r[0] != node_id and r[2] != node_id],
            )
        )
    return out


def _merge_with_receiver_slot(anchor_mem, triples, deltas, agent_xy, receiver_node):
    """AnchorMemory.merge, plus a reserved final slot holding the receiver.

    The receiver enters as a lone (node, PAD, PAD) entry in slot k, so the
    policy sees k_triples SAM-derived entries plus one position-only receiver
    entry. Its anchor is a placeholder here -- TransmitterInput.deltas
    overwrites anchors[-1, 0] from the env every step.
    """
    tri, anchors, mask = anchor_mem.merge(triples, deltas, agent_xy)
    k = tri.shape[0]
    out_tri = torch.full((k + 1, 3), TRIPLE_PAD, dtype=torch.long)
    out_tri[:k] = tri
    out_tri[k, 0] = receiver_node
    out_anchors = torch.zeros(k + 1, 2, 2, dtype=anchors.dtype)
    out_anchors[:k] = anchors
    out_mask = torch.zeros(k + 1, 2, 1, dtype=mask.dtype)
    out_mask[:k] = mask
    out_mask[k, 0] = 1.0
    return out_tri, out_anchors, out_mask


class TransmitterInput:
    """The one implementation of the transmitter's per-step policy input.

    Two things drive the transmitter -- the trainer and the video recorder --
    and if they build its input differently the video depicts a state the
    policy never saw. So every step that could drift lives here and nowhere
    else: stripping the receiver out of perception, assembling triples at the
    right k, reserving the receiver slot, and refreshing that slot's anchor
    from the env each step.

    Build these through a single zero-argument factory shared by both callers
    (see communication/train_communication.py) so there is also only one place the
    constructor arguments are chosen.
    """

    def __init__(self, vocabulary, per_frame_fn, receiver_node, k_triples):
        self.vocabulary = vocabulary
        self.per_frame_fn = per_frame_fn
        self.receiver_node = receiver_node
        self.k_triples = k_triples
        self.mem = AnchorMemory()
        self.triples = None
        self.anchors = None
        self.mask = None
        self.boxes = None
        self.relations = None

    def reset(self):
        """New episode: remembered anchors describe the previous layout."""
        self.mem.reset()

    def refresh(self, obs):
        """Run a perception pass and rebuild the slots.

        Returns (boxes, relations) per camera for the video overlay; the
        trainer ignores them. The receiver is stripped HERE, so no caller can
        forget to -- it reaches the policy only through its reserved slot.
        """
        per_frame = strip_concept(self.per_frame_fn(), self.receiver_node)
        self.boxes = [pf[0] for pf in per_frame]
        self.relations = [pf[3] for pf in per_frame]
        triples, deltas = assemble_triples_geo(self.vocabulary, per_frame, k=self.k_triples)
        self.triples, self.anchors, self.mask = _merge_with_receiver_slot(
            self.mem, triples, deltas, obs[:2], self.receiver_node
        )
        return self.boxes, self.relations

    def deltas(self, obs, env):
        """Agent-frame offsets, with the receiver slot refreshed from the env.

        The static objects keep the anchors their last SAM pass measured; only
        the receiver -- the one thing that moves -- is re-read every step.
        """
        self.anchors[-1, 0] = torch.as_tensor(
            env.receiver_anchor(), dtype=self.anchors.dtype
        )
        return deltas_from_anchors(self.anchors, self.mask, obs)
