"""Language games cannot leak target labels or enforce semantic word positions."""
import pytest
import torch
from experiments.spontaneous_grammar.task import build_cache,Game,split_mask,grid_execute,ROOMS,START
from experiments.spontaneous_grammar.models import Pair,intervene

@pytest.fixture(scope='module')
def caches(tmp_path_factory):
    root=tmp_path_factory.mktemp('role_free')
    out={}
    for domain in ('attributes','binding'):
        path=root/f'{domain}.pt';build_cache(domain,1,path);out[domain]=path
    return out

@pytest.mark.parametrize('domain',['attributes','binding'])
def test_rectangles_are_valid_target_balanced_and_split_clean(caches,domain):
    game=Game(caches[domain],123);d=game.data
    assert not d['held'][d['train']].any();assert d['held'][d['test']].all()
    for partition in ('train','test'):
        batch=game.sample(128,partition)
        for i in range(0,128,4):
            assert all(torch.equal(batch['candidates'][i],batch['candidates'][j]) for j in range(i,i+4))
            assert batch['target'][i:i+4].tolist()==[0,1,2,3]
            assert torch.equal(batch['private'][i:i+4],batch['candidates'][i])

@pytest.mark.parametrize('domain',['attributes','binding'])
def test_constant_wire_has_exact_chance_without_private_data(caches,domain):
    torch.manual_seed(1);game=Game(caches[domain],2);batch=game.sample(128,'test');model=Pair(domain,32)
    words=model.speaker(batch['private'])['tokens'];words=intervene(words,'blocked',8)
    logits=model.listener(batch['candidates'],tokens=words)
    assert float((logits.argmax(-1)==batch['target']).float().mean())==.25
    assert not hasattr(model.listener,'interpret') and not hasattr(model.speaker,'modular_heads')

@pytest.mark.parametrize('domain',['attributes','binding'])
def test_differentiable_wire_is_exact_integer_forward_and_matches_integer_listener(caches,domain):
    torch.manual_seed(3);batch=Game(caches[domain],4).sample(64);model=Pair(domain,32)
    out=model.speaker(batch['private'],mode='gumbel')
    assert set(out['wire'].detach().unique().tolist())<={0.,1.}
    a=model.listener(batch['candidates'],wire=out['wire'])
    b=model.listener(batch['candidates'],tokens=out['tokens'])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    loss=torch.nn.functional.cross_entropy(a,batch['target']);loss.backward()
    assert model.speaker.output.weight.grad.abs().sum()>0
    assert model.listener.cell.weight_ih.grad.abs().sum()>0

def test_end_marker_blocks_all_later_symbols():
    model=Pair('attributes',32);a=torch.tensor([[1,8,2,3,4,5,6,7]]);b=torch.tensor([[1,8,7,7,7,7,7,7]])
    torch.testing.assert_close(model.listener.message(tokens=a),model.listener.message(tokens=b),rtol=0,atol=0)

def test_recurrent_ppo_replays_same_message(caches):
    torch.manual_seed(5);batch=Game(caches['attributes'],6).sample(64);model=Pair('attributes',32)
    first=model.speaker(batch['private'],mode='sample');second=model.speaker(batch['private'],mode='sample',saved=first['tokens'])
    assert torch.equal(first['tokens'],second['tokens']);torch.testing.assert_close(first['logp'],second['logp'],rtol=0,atol=0)

def test_binding_encoder_ignores_array_order_but_not_physical_positions(caches):
    torch.manual_seed(7);batch=Game(caches['binding'],8).sample(64);model=Pair('binding',32)
    before=model.speaker.encoder(batch['private']);changed=model.speaker.encoder(batch['private'].flip(1))
    torch.testing.assert_close(before,changed,rtol=1e-6,atol=1e-6)
    moved=batch['private'].clone();moved[:,:2,-2:]=moved[:,:2,-2:].flip(1)
    assert not torch.allclose(before,model.speaker.encoder(moved))

def test_binding_role_swap_of_heldout_is_training_and_primitives_cover_both_roles(caches):
    d=torch.load(caches['binding'],weights_only=False);facts=d['facts'];held=d['held']
    swapped=facts[held].clone();swapped[:,:3]=facts[held,3:6];swapped[:,3:6]=facts[held,:3]
    assert not split_mask(swapped,'binding',1).any()
    for offset in (0,3):
        for a,n in enumerate((6,4,2)):assert len(facts[~held,offset+a].unique())==n

def test_three_object_extension_is_representable_but_never_training_data(caches):
    game=Game(caches['binding'],9);regular=game.sample(64);longer=game.sample_longer(16)
    assert (regular['private'].abs().sum(-1)>0).sum(-1).unique().tolist()==[2]
    assert (longer['private'].abs().sum(-1)>0).sum(-1).unique().tolist()==[3]
    assert longer['candidates'].shape==(64,4,3,14)
    assert Pair('binding',32).listener(longer['candidates'],tokens=torch.tensor([[1,8,9,9,9,9,9,9]]).repeat(64,1)).shape==(64,4)

def test_grid_routes_are_real_cardinal_paths_to_all_destinations():
    for room in range(4):
        path=grid_execute(room);assert path[0]==START and path[-1]==ROOMS[room] and len(path)==11
        assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))

@pytest.mark.parametrize('domain',['attributes','binding'])
def test_parallel_slots_are_unassigned_and_wire_is_discrete(caches,domain):
    model=Pair(domain,96,parallel=True,anchored=True);batch=Game(caches[domain],31).sample(64)
    out=model.speaker(batch['private'],mode='gumbel')
    assert out['tokens'].shape==(64,8) and out['tokens'].max()<8
    assert out['length'].unique().tolist()==[8.]
    assert set(out['wire'].detach().unique().tolist())<={0.,1.}
    a=model.listener(batch['candidates'],wire=out['wire'])
    b=model.listener(batch['candidates'],tokens=out['tokens'])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.nn.functional.cross_entropy(a,batch['target']).backward()
    assert model.speaker.parallel.weight.grad.abs().sum()>0
    assert model.speaker.parallel.in_features==96
    blocked=intervene(out['tokens'],'blocked',8)
    assert (model.listener(batch['candidates'],tokens=blocked).argmax(-1)==batch['target']).float().mean()==.25

@pytest.mark.parametrize('domain',['attributes','binding'])
def test_grounded_encoder_preserves_perceptual_distances_without_word_labels(caches,domain):
    from experiments.spontaneous_grammar.models import GroundedEncoder
    scene=Game(caches[domain],32).sample(64)['private'];encoder=GroundedEncoder(domain,96)
    z=encoder(scene);assert z.shape==(64,96)
    if domain=='attributes':
        torch.testing.assert_close((z[1:]-z[:-1]).norm(dim=-1),(scene[1:]-scene[:-1]).flatten(1).norm(dim=-1),rtol=1e-5,atol=1e-5)
    else:torch.testing.assert_close(z,encoder(scene.flip(1)),rtol=0,atol=0)

def test_attention_uses_shared_unassigned_symbol_head(caches):
    model=Pair('binding',96,anchored=True,attention=True);batch=Game(caches['binding'],33).sample(64)
    torch.testing.assert_close(model.speaker(batch['private'])['tokens'],model.speaker(batch['private'].flip(1))['tokens'])
    assert model.speaker.output.out_features==9 and not hasattr(model.speaker,'heads')

def test_minimum_length_is_only_training_constraint(caches):
    model=Pair('attributes',32);scene=Game(caches['attributes'],34).sample(16)['private']
    with torch.no_grad():model.speaker.output.bias[8]=100.
    assert (model.speaker(scene)['tokens'][:,0]==8).all()
    assert (model.speaker(scene,min_length=8)['tokens']<8).all()

def test_rewrite_probe_uses_local_substitution_not_fixed_word_positions():
    from experiments.spontaneous_grammar.evaluate import rewrite_fragment,replace_fragment,edit
    assert rewrite_fragment((1,2,3),(1,4,3))==((2,),(4,))
    assert replace_fragment((5,2,6),(2,),(4,))==(5,4,6)
    assert replace_fragment((5,6),(2,),(4,)) is None
    assert edit((1,2,3),(1,4,3))==1

def test_training_only_role_transform_never_admits_heldout(caches):
    game=Game(caches['binding'],35);facts=game.data['facts'][~game.data['held']]
    swapped=facts.clone();swapped[:,:3]=facts[:,3:6];swapped[:,3:6]=facts[:,:3]
    eligible=~split_mask(swapped,'binding',1)
    assert eligible.any() and (~eligible).any()
    assert not split_mask(swapped[eligible],'binding',1).any()

def test_new_arena_bridge_does_not_read_private_target_or_status():
    from experiments.spontaneous_grammar.arena_transfer import select
    from experiments.grammar_arena.evaluate import cases
    torch.manual_seed(41);model=Pair('attributes',96,anchored=True,parallel=True)
    batch,geometry,yaw=cases('food_pair',1,1,42,'standard')
    words=torch.randint(8,(64,8));before=select(model,batch,words,geometry,yaw)
    changed=dict(batch);changed['facts']=torch.flip(batch['facts'],[0]);changed['target']=(batch['target']+1)%4
    after=select(model,changed,words,geometry,yaw)
    import numpy as np
    for key in before:np.testing.assert_array_equal(before[key],after[key])

def test_new_arena_hypothesis_retrieval_realizes_a_known_scene_fixture():
    from experiments.spontaneous_grammar.arena_transfer import select,new_facts
    from experiments.spontaneous_grammar.task import features
    from experiments.grammar_arena.evaluate import cases
    class ExactSceneFixture:
        # A test-only recognizer verifies hypothesis construction. No model
        # training uses this fixture or its deliberately labeled test message.
        def listener(self,candidates,tokens):
            target=features(tokens[:,:5],'attributes')
            return -100.*(candidates-target[:,None]).square().sum((-1,-2))
    batch,geometry,yaw=cases('food_pair',1,1,43,'standard');meaning=new_facts(batch['facts'])
    words=torch.cat((meaning,torch.zeros(64,3,dtype=torch.long)),-1)
    selection=select(ExactSceneFixture(),batch,words,geometry,yaw)
    assert (torch.tensor(selection['food'])==batch['target']).all()
    for i,route in enumerate(selection['route']):
        dangerous_gate=0 if int(batch['swap'][i])==int(batch['facts'][i,2]) else 1
        assert not (batch['facts'][i,3] and int(route)==dangerous_gate)

def test_transfer_common_subdomain_has_twelve_new_heldout_meanings():
    from experiments.spontaneous_grammar.arena_transfer import new_facts
    from experiments.grammar_food.task import all_facts
    facts=new_facts(all_facts());assert facts.shape==(64,5)
    assert int(split_mask(facts,'attributes',1).sum())==12
