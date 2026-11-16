"""Ensure admission cannot select models using held-out meanings."""
import pytest
import torch
from torch import nn
from experiments.grammar_arena.quality_control import training_admission
from experiments.grammar_food.task import held_out

class RecordingPair(nn.Module):
    def __init__(self):super().__init__();self.dummy=nn.Parameter(torch.zeros(1));self.seen=None
    def speak(self,facts,greedy):self.seen=facts.clone();return (facts,)
    def listen(self,words):return [torch.nn.functional.one_hot(words[:,a],n).float()*10 for a,n in enumerate((4,4,2,2))]

@pytest.mark.parametrize('split,shift',[('context',0),('food_pair',1)])
def test_admission_forwards_only_training_meanings_and_covers_primitives(split,shift):
    pair=RecordingPair();score=training_admission(pair,dict(split=split,shift=shift))
    assert score['exact']==1. and len(pair.seen)==48
    assert not held_out(pair.seen,split,shift).any()
    for a,n in enumerate((4,4,2,2)):assert len(pair.seen[:,a].unique())==n
