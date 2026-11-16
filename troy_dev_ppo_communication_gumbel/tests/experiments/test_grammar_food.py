"""Independent information-boundary, reward, split, and learning checks."""
import itertools
import numpy as np
import torch
import pytest
from experiments.grammar_food.task import *
from experiments.grammar_food.models import Pair
from experiments.grammar_food.evaluate import mi,edit

def test_splits_withhold_sixteen_complete_meanings_and_cover_primitives():
    facts=all_facts()
    for split in ('context','food_pair'):
        for shift in range(4):
            test=held_out(facts,split,shift)
            assert int(test.sum())==16
            for a,n in enumerate(ARITIES):assert len(facts[~test,a].unique())==n
            samples=Sampler(13,split=split,shift=shift).sample(10000)
            assert not held_out(samples['facts'],split,shift).any()

def test_exhaustive_contexts_and_no_private_order_marker():
    batch=exhaustive_contexts();n=len(batch['facts']);assert n==27648
    for meaning in all_facts():
        mask=(batch['facts']==meaning).all(-1)
        assert int(mask.sum())==432
        assert torch.bincount(batch['target'][mask],minlength=4).tolist()==[108]*4
    chosen=batch['candidates'][torch.arange(n),batch['target']]
    assert torch.equal(chosen,batch['facts'][:,:2])
    assert len(torch.unique(batch['candidates'].reshape(n,-1),dim=0))==864

@pytest.mark.parametrize('channel',['flat','sequence','modular','slots','tied_slots','silent'])
def test_receiver_is_invariant_to_private_state_fields(channel):
    torch.manual_seed(23);model=Pair(channel);batch=Sampler(24).sample(64)
    words=model.speak(batch['facts'],True)[0]
    before=model.action_distributions(batch,words)
    altered={**batch,'facts':1-batch['facts'],'target':(batch['target']+1)%4}
    after=model.action_distributions(altered,words)
    for a,b in zip(before[:3],after[:3]):assert torch.equal(a,b)

def test_sender_does_not_observe_food_permutation_or_route_layout():
    torch.manual_seed(24);model=Pair('sequence');facts=all_facts()
    a=model.speak(facts,True)[0];b=model.speak(facts,True)[0]
    assert torch.equal(a,b)
    # The signature itself accepts facts only; there is no context input.
    assert 'candidates' not in model.speak.__code__.co_varnames

def test_sampled_ppo_replay_does_not_resample_messages_or_actions():
    torch.manual_seed(31);model=Pair('sequence');batch=Sampler(32).sample(128)
    batch['query_game']=torch.rand(128)<.5
    first=model.act(batch);second=model.act(batch,saved=first)
    for key in ('words','food','route','answer','logp'):assert torch.equal(first[key],second[key])

def test_both_agents_get_gradients_from_sampled_reward():
    torch.manual_seed(41);model=Pair('sequence');batch=Sampler(42).sample(1024)
    batch['query_game']=torch.zeros(1024,dtype=torch.bool)
    chosen=model.act(batch);reward=outcome(batch,chosen['food'],chosen['route'])['reward']
    loss=-(chosen['logp']*(reward-reward.mean())).mean();loss.backward()
    assert model.sender_output.weight.grad.abs().sum()>0
    assert model.receiver_cell.weight_ih.grad.abs().sum()>0
    assert model.interpret[0].weight.grad.abs().sum()>0

def test_modular_symbols_do_not_change_with_other_fields():
    torch.manual_seed(51);model=Pair('modular');facts=all_facts()
    baseline=model.speak(facts,True)[0]
    for a,n in enumerate(ARITIES):
        changed=facts.clone();changed[:,a]=(changed[:,a]+1)%n
        words=model.speak(changed,True)[0]
        for j in range(4):
            if j!=a:assert torch.equal(words[:,j],baseline[:,j])

def test_vector_reward_matches_independent_cell_execution():
    batch=Sampler(61).sample(256)
    for i in range(64):
        for food,route in itertools.product(range(4),range(3)):
            scalar=execute_path(batch['facts'][i].tolist(),batch['candidates'][i].tolist(),int(batch['swap'][i]),food,route)
            selected={k:v[i:i+1] for k,v in batch.items()}
            vector=outcome(selected,torch.tensor([food]),torch.tensor([route]))
            for k in ('success','hazard'):assert scalar[k]==bool(vector[k][0])
            assert scalar['steps']==float(vector['steps'][0])
            assert abs(scalar['reward']-float(vector['reward'][0]))<1e-6

def test_public_paths_are_legal_and_use_exactly_one_gate():
    wall={(5,y) for y in range(HEIGHT)}-set(GATES)
    for food,routes in enumerate(PATHS):
        for j,path in enumerate(routes):
            assert path[0]==START and path[-1]==GOALS[food]
            assert not set(path)&wall
            assert set(path)&set(GATES)=={GATES[j]}
            assert all(abs(x[0]-y[0])+abs(x[1]-y[1])==1 for x,y in zip(path,path[1:]))

def test_null_wire_cannot_depend_on_private_facts():
    model=Pair('silent');words,lp,h=model.speak(all_facts(),False)
    assert torch.unique(words).tolist()==[4]
    assert torch.equal(lp,torch.zeros(64))

def test_information_and_edit_metrics_on_known_codes():
    facts=all_facts().numpy()
    assert abs(mi(facts[:,0],facts[:,0])-2)<1e-10
    assert abs(mi(facts[:,0],facts[:,1]))<1e-10
    assert edit([0,1,2,3],[0,1,2,3])==0
    assert edit([0,1,2,3],[3,2,1,0])==4

def test_optional_gate_inputs_contain_only_visibility_and_cost():
    from experiments.grammar_food.optional_tool import Gate
    gate=Gate();obs=torch.tensor([[0.,.01],[1.,.4]])
    assert gate.actor(obs).shape==(2,3)
    assert gate.critic(obs).shape==(2,1)

def test_new_maze_paths_do_not_cross_unselected_gates():
    from experiments.grammar_food.transfer import generate_layouts
    layouts=generate_layouts(4,123)
    for layout in layouts:
        for goal,routes in zip(layout['goals'],layout['paths']):
            for gate,path in zip(layout['gates'],routes):
                assert path[0]==layout['start'] and path[-1]==goal
                assert not set(path)&set(layout['blocked'])
                assert set(path)&set(layout['gates'])=={gate}
                assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))
