import itertools
import torch
from experiments.communication_options.models import (
    PrivateGoal,RouteSafety,Scouting,Complementary,silent_route_ceiling,broadcast_bound)


@torch.no_grad()
def test_private_goal_indistinguishability_bound():
    model=PrivateGoal('silent')
    for hard in (False,True):
        assert abs(float(model(hard)['success'])-.25)<1e-6
    assert float(PrivateGoal('fixed')(True)['success'])==1


@torch.no_grad()
def test_route_bound_and_event_control():
    model=RouteSafety('silent')
    with torch.no_grad(): model.receiver[:,0]=10;model.receiver[:,1]=-10
    assert abs(float(model(True)['success'])-silent_route_ceiling())<1e-6
    event=RouteSafety('fixed_event')
    assert abs(float(event(True)['success'])-1)<1e-6
    assert float(event(True)['packets'])<1.1
    assert float(event(True,'blocked')['success'])<.4
    assert float(RouteSafety('oracle')(True,'blocked')['success'])==1


@torch.no_grad()
def test_route_exact_recursion_matches_independent_status_path_enumeration():
    torch.manual_seed(4);model=RouteSafety('learned')
    paths=torch.tensor(list(itertools.product((0,1),repeat=9)))
    p=torch.full((len(paths),),.5)
    for t in range(1,9):p*=torch.where(paths[:,t]!=paths[:,t-1],.08,.92)
    h=torch.zeros(len(paths),dtype=torch.long);alive=torch.ones(len(paths),dtype=torch.bool)
    packets=torch.zeros(len(paths));sender=model.sender.argmax(-1);dec=model.receiver.argmax(-1)
    for t in range(9):
        z=sender[t,paths[:,t],h];packets+=alive*(z>0)
        h=torch.where(z>0,z,h)
        if t in (0,4,8):alive &= dec[h]==paths[:,t]
    result=model(True)
    torch.testing.assert_close(result['success'],(p*alive).sum())
    torch.testing.assert_close(result['packets'],(p*packets).sum())


@torch.no_grad()
def test_scouting_receiver_has_no_target_observation_when_hidden():
    model=Scouting('silent')
    with torch.no_grad():
        model.first[:]=-20
        for cost in range(3):
            for goal in range(4):model.first[cost,goal,goal]=20
            model.first[cost,4,4]=20
    r=model(True)
    assert abs(float(r['utility'])-.91)<1e-6
    assert abs(float(r['success'])-1)<1e-6
    assert abs(float(r['scouts'])-.5)<1e-6


@torch.no_grad()
def test_complementary_request_is_needed_and_bit_accounting_is_fair():
    assert float(Complementary('silent')(True)['success'])==.5
    r=Complementary('fixed_dialogue')(True)
    assert float(r['success'])==1
    assert float(r['bits'])==5
    assert abs(float(r['utility'])-.9)<1e-6
    assert abs(float(Complementary('fixed_dialogue')(True,'remove_query')['success'])-.5625)<1e-6
    assert .92<broadcast_bound()<.94


def test_all_learned_models_have_finite_gradients():
    for model in (PrivateGoal(),RouteSafety(),Scouting(),Complementary()):
        loss=-model()['utility'];loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in model.parameters())


def test_receiver_planner_follows_selected_public_gate():
    from experiments.communication_options.navigation import room_route
    for choice in (0,1):
        path=room_route(choice)
        assert path[0]==(0,3) and path[-1]==(6,3)
        assert (3,1 if choice==0 else 5) in path
        assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))
        assert all(x!=3 or y in (1,5) for x,y in path)


def test_receiver_planner_does_not_pass_through_blocked_cells():
    from experiments.communication_options.navigation import plan
    path=plan((0,0),(2,0),{(1,0)},3,3)
    assert len(path)==5 and (1,0) not in path


@torch.no_grad()
def test_separate_route_caches_match_reliable_channel():
    from experiments.communication_options.interventions import route_intervention
    torch.manual_seed(19)
    for method in ('learned','memoryless','continuous','opening','silent','fixed_event'):
        model=RouteSafety(method)
        for hard in (False,True):
            fast=model(hard);joint=route_intervention(model,hard,'normal')
            for key in fast:torch.testing.assert_close(fast[key],joint[key])


@torch.no_grad()
def test_packet_loss_has_no_sender_acknowledgment():
    from experiments.communication_options.interventions import route_intervention
    torch.manual_seed(37);model=RouteSafety()
    n=100000;gen=torch.Generator().manual_seed(81031)
    state=torch.randint(2,(n,),generator=gen)
    hs=torch.zeros(n,dtype=torch.long);hr=hs.clone()
    active=torch.ones(n,dtype=torch.bool);packets=torch.zeros(n)
    sender=model.sender.argmax(-1);decoder=model.receiver.argmax(-1)
    for t in range(9):
        z=sender[t,state,hs];packets+=active*(z>0)
        hs=torch.where(z>0,z,hs)  # sender knows intended words only
        delivered=torch.where(torch.rand(n,generator=gen)<.25,0,z)
        hr=torch.where(delivered>0,delivered,hr)
        if t in (0,4,8):active &= decoder[hr]==state
        if t<8:state=torch.where(torch.rand(n,generator=gen)<.08,1-state,state)
    joint=route_intervention(model,True,'drop25')
    assert abs(float(joint['success'])-float(active.float().mean()))<.01
    assert abs(float(joint['packets'])-float(packets.mean()))<.02
