import sys
from pathlib import Path
import pytest
import torch
from communication.SenderEntropy import entropy_coefficient, sender_entropy


def test_annealing_boundaries():
    for update,expected in [(1,.1),(500,.1),(1000,.05),(1500,0),(2000,0)]:
        assert entropy_coefficient(update,2000,.1,'annealed')==pytest.approx(expected)
        assert entropy_coefficient(update,2000,.1,'constant')==.1


def test_bonus_pushes_against_confidence():
    logits=torch.tensor([[3.,0.,0.,0.]],requires_grad=True)
    (-sender_entropy(logits)).backward()
    assert logits.grad[0,0]>0
    assert (logits.grad[0,1:]<0).all()
    assert sender_entropy(torch.zeros(2,4))==pytest.approx(1.38629436)


@pytest.mark.parametrize('beta',[-1,float('nan'),float('inf')])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):entropy_coefficient(1,2000,beta,'annealed')
