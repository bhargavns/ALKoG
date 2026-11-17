"""Checks for the grammar-pressures event world, sampler and channel."""
import torch
from experiments.grammar_pressures.world import World, PARTITIONS, NOVEL, S_ONLY, A_ONLY, P_ONLY, NC, NT
from experiments.grammar_pressures.agents import Speaker, Listener, strip, pack
from experiments.grammar_pressures.train import decision_weights

W = World()


def test_partitions_keep_every_primitive_in_training():
    train = W.events[W.partition == 0]
    assert len(W.events) == 2304 and (W.partition == 0).sum() == 840
    ents = set(train[:, 2]) | set(train[train[:, 0] == 2, 3])
    assert {e // NC for e in ents} == set(range(NT)) and {e % NC for e in ents} == set(range(NC))
    assert not ents & NOVEL                                        # novel descriptions never occur in training
    tr = train[train[:, 0] == 2]
    assert not (set(tr[:, 2]) | set(tr[:, 3])) & S_ONLY            # seen only as sole participant
    assert not set(tr[:, 3]) & A_ONLY and not set(tr[:, 2]) & P_ONLY


def test_role_reversal_is_a_different_event_with_the_same_entities():
    i = W.lookup[(2, 1, 5, 9)]; j = int(W.swap[i])
    assert tuple(W.events[j]) == (2, 1, 9, 5) and not torch.equal(W.features[i], W.features[j])
    assert torch.equal(W.features[i].sum(0)[:NT + NC], W.features[j].sum(0)[:NT + NC])


def test_sampler_is_balanced_and_stays_inside_its_partition():
    g = torch.Generator().manual_seed(0)
    for name in PARTITIONS:
        for n_cand, focus in ((4, (1, 2, 3)), (16, (6,))):
            b = W.sample(32, name, g, n_cand, focus); c = b['cand_ids'].reshape(32, n_cand, n_cand)
            assert (c == c[:, :1]).all()                           # the same rooms for every target of a set
            assert all(len(set(r.tolist())) == n_cand for r in c[:, 0])
            assert (W.part_t[c] == PARTITIONS.index(name)).all() and (W.val_t[c] == W.val_t[c[:, :1, :1]]).all()
            assert torch.equal(b['target'].reshape(32, n_cand), torch.arange(n_cand).expand(32, -1))
            assert torch.equal(b['goal'], c[:, 0].flatten())


def test_focus_one_contrasts_a_single_slot_unless_padded():
    """With focus 1 every distractor edits the same slot of the anchor.  When the
    partition leaves fewer than three such neighbours the set is padded with
    unrelated events, so only part of the sets are pure one-contrast contexts."""
    g = torch.Generator().manual_seed(1); c = W.sample(300, 'train', g, 4, (1,))['cand_ids'].reshape(300, 4, 4)[:, 0]
    pure = 0
    for row in c.tolist():
        diffs = set()
        for i in row:
            for j in row:
                a, d = W.slots(i), W.slots(j); diffs |= {k for k in a if a[k] != d[k]}
        pure += len(diffs) == 1
    assert 0.3 < pure / 300 < 0.9


def test_silent_channel_is_exactly_chance():
    torch.manual_seed(0); g = torch.Generator().manual_seed(2); l = Listener()
    b = W.sample(64, 'train', g, 4); silent = torch.full((len(b['target']), 6), l.eos)
    assert float((l(b['candidates'], silent).argmax(-1) == b['target']).float().mean()) == 0.25


def test_listener_sees_only_public_rooms_and_integers():
    torch.manual_seed(0); g = torch.Generator().manual_seed(3); s, l = Speaker(), Listener(); b = W.sample(16, 'train', g, 4)
    tok = s(b['private'], b['candidates'], mode='sample')['tokens']
    assert tok.dtype == torch.long and int(tok.max()) <= s.eos
    first = (tok == s.eos).float().argmax(1)
    for row, f in zip(tok, first):
        if (row == s.eos).any(): assert (row[f:] == s.eos).all()   # nothing after the end of an utterance
    assert strip(pack(strip(tok, s.eos), s.eos, 6), s.eos) == strip(tok, s.eos)


def test_blind_speaker_ignores_the_rooms():
    torch.manual_seed(0); g = torch.Generator().manual_seed(4); s = Speaker(); b = W.sample(16, 'train', g, 4)
    a = s(b['private'])['tokens']; c = s(b['private'][torch.arange(64)], None)['tokens']
    assert torch.equal(a, c) and torch.equal(a.reshape(16, 4, 6)[:, 0], s(W.features[b['goal'].reshape(16, 4)[:, 0]])['tokens'])


def test_decision_weights_form_a_distribution_over_heard_prefixes():
    alive = torch.tensor([[1., 1., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]])
    for h in (0., 0.3):
        w = decision_weights(alive, h)
        assert torch.allclose(w.sum(1), alive[:, 0]) and (w >= 0).all()
    assert torch.equal(decision_weights(alive, 0.)[0], torch.tensor([0., 0., 1., 0.]))
