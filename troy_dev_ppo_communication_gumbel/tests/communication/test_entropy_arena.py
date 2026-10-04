import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from communication.GumbelPPOTrainer import GumbelPPOTrainer, evaluate_records
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic
from evaluation.CommunicationEvaluation import CommunicationActor
from communication.SenderEntropy import entropy_coefficient, sender_entropy
from helpers import ToyEnv,ToyInput

class Sender16(nn.Module):
    def __init__(self):
        super().__init__();self.message_head=nn.Linear(4,16)
    def forward(self,obs,triples,deltas):return self.message_head(obs)

def trainer(beta,lr=3e-4):
    torch.manual_seed(19)
    return GumbelPPOTrainer(Sender16(),CurrentMessageReceiverActorCritic(n_symbols=16),ToyEnv(length=13),ToyInput,
        device='cpu',horizon=8,epochs=1,minibatch_size=8,learning_rate=lr,n_symbols=16,
        sender_entropy_beta=beta,total_iterations=800)

def test_800_iteration_schedule():
    for it,beta in [(1,.1),(200,.1),(400,.05),(600,0),(800,0)]:
        assert entropy_coefficient(it,800,.1,'annealed')==pytest.approx(beta)

def test_matched_initialization_rollout_and_entropy_update():
    t=trainer(.1);b,stats=t.collect_rollout()
    c=trainer(0);bc,_=c.collect_rollout()
    for key in ('actions','logp','values','advantages','returns'):torch.testing.assert_close(b[key],bc[key],atol=0,rtol=0)
    for k,v in t.transmitter.state_dict().items():torch.testing.assert_close(v,c.transmitter.state_dict()[k],atol=0,rtol=0)
    for k,v in t.receiver.state_dict().items():torch.testing.assert_close(v,c.receiver.state_dict()[k],atol=0,rtol=0)
    assert len(stats['symbol_fracs'])==16
    dist,_,_,messages=evaluate_records(t.transmitter,t.receiver,t.channel,b['records'])
    torch.testing.assert_close(dist.log_prob(b['actions']),b['logp'])
    assert messages.shape[-1]==16
    result=t.update(b);control=c.update(bc)
    assert result['sender_entropy_beta']==.1 and result['tx_entropy_grad_norm']>0
    assert control['sender_entropy_beta']==0 and control['tx_entropy_grad_norm']==0
    assert any(not torch.equal(v,c.transmitter.state_dict()[k]) for k,v in t.transmitter.state_dict().items())

def test_entropy_counts_each_decision_once_across_rollouts():
    t=trainer(.1,lr=0.)
    t.collect_rollout()
    batch,_=t.collect_rollout()
    r=batch['records']
    assert len(r['noise']) == t.horizon
    expected=float(sender_entropy(t.transmitter(r['obs'],r['triples'],r['deltas'])).detach())
    stats=t.update(batch)
    assert stats['sender_entropy']==pytest.approx(expected,abs=1e-6)
    t.iteration=600
    stats=t.update(batch)
    assert stats['sender_entropy_beta']==0 and stats['tx_entropy_grad_norm']==0

@pytest.mark.parametrize('mode',['hard','sampled','constant','random','shuffle'])
def test_evaluation_preserves_16_symbol_ids(mode):
    tx=Sender16();rx=CurrentMessageReceiverActorCritic(n_symbols=16)
    with torch.no_grad():
        tx.message_head.weight.zero_();tx.message_head.bias.fill_(-100);tx.message_head.bias[15]=100
    actor=CommunicationActor(tx,rx,'cpu',mode=mode,shuffled_messages=[15,14] if mode=='shuffle' else None)
    seen=[]
    for _ in range(80):
        action,symbol,value,_=actor.act(np.zeros(4),ToyInput.triples,torch.zeros(1,2,2))
        assert 0<=action<4 and 0<=symbol<16
        seen.append(symbol)
    if mode in ('hard','sampled'):assert set(seen)=={15}
    elif mode=='constant':assert set(seen)=={0}
    elif mode=='shuffle':assert seen[:2]==[15,14]
    else:assert max(seen)>3
