"""Independent actor boundaries, dynamics, collision and time-limit checks."""
import inspect
import numpy as np
import pytest
import torch
from experiments.grammar_arena.arena import FoodArena,segment_aabb,segment_distance,GATE_Y
from experiments.grammar_arena.bridge import choose,public_lengths
from experiments.grammar_arena.train_navigation import gae,Navigator
from experiments.grammar_food.models import Pair
from experiments.grammar_food.task import Sampler
from environment.KGWorldEnv import CommunicationKGWorldEnv

GEOMETRY=dict(scale=1.,translation=[0,0],rotation=0.)
FACTS=[0,0,0,1]
CANDIDATES=[[0,0],[0,1],[1,0],[1,1]]

def new_env(facts=FACTS):
    env=FoodArena(123);env.reset_task(facts,CANDIDATES,0,GEOMETRY,0.)
    env.select(0,1);return env

def test_original_arena_is_inherited_and_body_position_updates():
    env=new_env();assert isinstance(env,CommunicationKGWorldEnv)
    old=env.receiver_xy.copy();obs,reward,term,trunc,info=env.step(0)
    np.testing.assert_allclose(env.receiver_xy-old,[.2,0],atol=1e-8)
    np.testing.assert_allclose(env.data.xpos[env.receiver_id,:2],env.receiver_xy,atol=1e-8)
    assert env.model.body('agent').id>=0 and env.model.camera('cam_front').id>=0
    assert env.renderer is None
    env.close()

def test_swept_wall_rejects_motion_in_all_four_orientations():
    for theta in (0,np.pi/2,np.pi,3*np.pi/2):
        env=new_env();env._geometry(rotation=theta)
        old=env.world([-.4,2]);env._set_receiver(old,theta)
        env.step_size=1.;env._apply_action(0)
        np.testing.assert_allclose(env.receiver_xy,old,atol=1e-8);assert env.bump
        env.close()

def test_selected_opening_can_be_crossed_without_wall_teleportation():
    env=new_env();env._set_receiver([-.4,GATE_Y[1]],0);env.step_size=.8
    env._apply_action(0);np.testing.assert_allclose(env.receiver_xy,[.4,GATE_Y[1]])
    assert not env.bump;env.close()

def test_free_lion_penalty_occurs_on_swept_segment_before_food():
    env=new_env();env._set_receiver([-1,GATE_Y[0]],0);env.step_size=2.
    _,reward,term,trunc,info=env.step(0)
    assert term and info['hazard'] and not info['reached'] and reward==-50.01
    env.close()

def test_caged_lion_is_harmless_and_other_animal_is_caged():
    env=new_env([0,0,0,0]);env._set_receiver([-1,GATE_Y[0]],0);env.step_size=2.
    _,_,term,_,info=env.step(0);assert not term and not info['hazard']
    env.close()
    env=new_env([0,0,1,1]);env._set_receiver([-1,GATE_Y[0]],0);env.step_size=2.
    _,_,term,_,info=env.step(0);assert not term and not info['hazard'];env.close()

def test_wrong_food_really_terminates_and_does_not_receive_bonus():
    env=new_env();env._set_receiver(env.food_xy[1]-[1,0],0);env.step_size=1.
    _,reward,term,_,info=env.step(0)
    assert term and info['food']==1 and not info['reached'] and not info['hazard'] and reward==-.01
    env.close()

def test_motor_observation_does_not_include_private_labels_or_messages():
    env=new_env();before=env.observation().copy();env.facts[:]=1;env.target=3
    np.testing.assert_array_equal(before,env.observation());assert before.shape==(3,)
    env.close()

def test_collector_choices_invariant_to_private_facts_for_fixed_wire():
    torch.manual_seed(12);model=Pair('modular');batch=Sampler(13).sample(32)
    words=model.speak(batch['facts'],True)[0]
    geometries=[GEOMETRY]*32;yaws=np.linspace(-np.pi,np.pi,32)
    original=choose(model,batch,words,geometries,yaws)
    altered={**batch,'facts':1-batch['facts'],'target':(batch['target']+1)%4}
    changed=choose(model,altered,words,geometries,yaws)
    for key in original:np.testing.assert_array_equal(original[key],changed[key])

def test_public_lengths_are_rigid_transform_invariant_with_matching_yaw():
    a=public_lengths(GEOMETRY,.3)
    b=public_lengths(dict(scale=1.,translation=[.3,-.2],rotation=np.pi/2),.3+np.pi/2)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-9)

def test_time_limit_bootstraps_last_state_but_cuts_reset_advantage():
    rewards=torch.tensor([[1.],[10.]])
    terminal=torch.tensor([[0.],[1.]]);done=torch.tensor([[1.],[1.]])
    values=torch.tensor([[2.],[3.]]);next_values=torch.tensor([[4.],[100.]])
    adv,ret=gae(rewards,terminal,done,values,next_values,gamma=.5,lam=1.)
    torch.testing.assert_close(adv,torch.tensor([[1.],[7.]]))
    torch.testing.assert_close(ret,torch.tensor([[3.],[10.]]))

def test_swept_helpers_handle_stationary_and_crossing_segments():
    assert segment_aabb([-1,0],[1,0],[-.1,-1],[.1,1])
    assert not segment_aabb([-1,2],[1,2],[-.1,-1],[.1,1])
    assert segment_aabb([0,0],[0,0],[-.1,-1],[.1,1])
    assert not segment_aabb([2,0],[2,0],[-.1,-1],[.1,1])
    assert segment_distance(np.array([-1,0]),np.array([1,0]),np.zeros(2))==0.

def test_waypoint_controller_has_no_teacher_or_private_state_inputs():
    model=Navigator();dist,value=model(torch.zeros(5,3));assert dist.logits.shape==(5,4) and value.shape==(5,)
    assert 'facts' not in inspect.signature(model.forward).parameters
