"""Discrete forward semantics, categorical gradients, and faithful PPO replay."""
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic
from communication.GumbelPPOTrainer import evaluate_records


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError): StraightThroughGumbelChannel(temperature)


def test_discrete_forward_soft_backward_and_temperature():
    logits = torch.tensor([[.2, -.1, .3, -.4]], requires_grad=True)
    noise = torch.tensor([[.1, .2, -.3, .4]])
    c = StraightThroughGumbelChannel(1.2)
    msg = c(logits, noise)
    assert set(msg.detach().flatten().tolist()) <= {0., 1.}
    assert msg.sum() == 1
    torch.testing.assert_close(msg, c(logits, noise), rtol=0, atol=0)
    torch.testing.assert_close(msg, StraightThroughGumbelChannel(.5)(logits, noise))
    weights = torch.tensor([[1., 2., -1., 4.]])
    g = torch.autograd.grad((msg*weights).sum(), logits)[0]
    expected = torch.autograd.grad((c.soft(logits, noise)*weights).sum(), logits)[0]
    torch.testing.assert_close(g, expected)
    assert g.abs().sum() > 0


def test_sampled_channel_matches_categorical_distribution():
    torch.manual_seed(9)
    logits=torch.tensor([0., 1., -.5, .5]).expand(30000, -1)
    msg=StraightThroughGumbelChannel()(logits)
    torch.testing.assert_close(msg.mean(0), logits[0].softmax(-1), atol=.012, rtol=0)


def test_receiver_symbol_equivalence_and_gradients():
    rx=CurrentMessageReceiverActorCritic()
    messages=torch.nn.functional.one_hot(torch.arange(4),4).float().requires_grad_()
    dist,value=rx.distribution_messages(messages)
    embedded = rx.token_embeddings(torch.arange(4))
    torch.testing.assert_close(dist.probs,rx.actor(embedded).softmax(-1))
    torch.testing.assert_close(value,rx.critic(embedded).squeeze(-1))
    value.sum().backward(retain_graph=True)
    assert messages.grad.abs().sum()>0
    messages.grad.zero_()
    dist.log_prob(torch.arange(4)).sum().backward()
    assert messages.grad.abs().sum()>0


class Sender(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_head=nn.Linear(4,4)
    def forward(self, obs, triples, deltas):
        return self.message_head(obs)


@pytest.mark.parametrize('detach', [False,True])
def test_rollout_replay_and_channel_detachment(detach):
    torch.manual_seed(11)
    tx,rx,c=Sender(),CurrentMessageReceiverActorCritic(),StraightThroughGumbelChannel()
    records=dict(obs=torch.randn(4,4),triples=torch.zeros(4,1,3),deltas=torch.zeros(4,1,2,2),noise=c.sample_noise((4,4),'cpu'))
    with torch.no_grad():
        old,_,_,_=evaluate_records(tx,rx,c,records)
        actions=old.sample();old_logp=old.log_prob(actions)
    dist,_,_,messages=evaluate_records(tx,rx,c,records,detach_channel=detach)
    torch.testing.assert_close(dist.log_prob(actions),old_logp,atol=0,rtol=0)
    assert set(messages.detach().flatten().tolist()) <= {0.,1.}
    dist.log_prob(actions).sum().backward()
    g=tx.message_head.weight.grad
    assert (g is None) if detach else (g is not None and g.abs().sum()>0)


@pytest.mark.parametrize('detach', [False, True])
def test_gumbel_ppo_updates_and_replays_across_horizons(detach):
    from helpers import ToyEnv, ToyInput
    from communication.GumbelPPOTrainer import GumbelPPOTrainer
    torch.manual_seed(21)
    t=GumbelPPOTrainer(Sender(),CurrentMessageReceiverActorCritic(),ToyEnv(length=13),ToyInput,
        device='cpu',horizon=8,epochs=2,minibatch_size=4,n_symbols=4,detach_channel=detach)
    before={k:v.clone() for k,v in t.transmitter.state_dict().items()}
    batch,stats=t.collect_rollout()
    dist,_,_,_=evaluate_records(t.transmitter,t.receiver,t.channel,batch['records'])
    torch.testing.assert_close(dist.log_prob(batch['actions']),batch['logp'],atol=1e-6,rtol=1e-5)
    assert batch['records']['noise'].shape[-1]==4
    assert stats['symbol_metric']=='sampled_discrete'
    pending_noise=t._pending_record['noise'].clone()
    pending_id=t._pending_record['id']
    updated=t.update(batch)
    changed=any(not torch.equal(v,before[k]) for k,v in t.transmitter.state_dict().items())
    assert changed != detach
    assert updated['tx_actor_grad_norm']==0 if detach else updated['tx_actor_grad_norm']>0
    batch2,_=t.collect_rollout()
    assert t.last_record_ids[0] == pending_id
    torch.testing.assert_close(batch2['records']['noise'][0],pending_noise)
    dist,_,_,_=evaluate_records(t.transmitter,t.receiver,t.channel,batch2['records'])
    torch.testing.assert_close(dist.log_prob(batch2['actions']),batch2['logp'],atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('symbols', [4, 8, 16])
def test_diagnostic_vocabulary_sizes(symbols):
    torch.manual_seed(42)
    c=StraightThroughGumbelChannel(n_symbols=symbols)
    rx=CurrentMessageReceiverActorCritic(n_symbols=symbols)
    logits=torch.randn(4,symbols,requires_grad=True)
    messages=c(logits)
    assert messages.shape==(4,symbols)
    assert set(messages.detach().flatten().tolist())<={0.,1.}
    torch.testing.assert_close(messages.sum(-1),torch.ones(4))
    dist,_=rx.distribution_messages(messages)
    assert dist.probs.shape==(4,4)
    expected=rx.actor(rx.token_embeddings(messages.argmax(-1))).softmax(-1)
    torch.testing.assert_close(dist.probs,expected)
    (-dist.log_prob(torch.arange(4)).mean()).backward()
    assert logits.grad.abs().sum()>0
    assert rx.token_embeddings.weight.shape==(symbols,4)
    assert sum(p.numel() for p in rx.parameters())==65+4*symbols
