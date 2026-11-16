import torch
from experiments.event_communication.study import Grid, Pair, rollout, replay, targets


def test_receiver_observation_cannot_reveal_private_status():
    env = Grid(32, 4)
    before = env.observation().clone()
    env.risk = 1-env.risk
    torch.testing.assert_close(before, env.observation())


def test_hazard_flip_and_absorbing_terminal():
    env = Grid(1, 0)
    env.pos[:] = torch.tensor([4, 3]); env.lion[:] = torch.tensor([5, 3])
    env.risk[:] = 1; env.changes[:] = False
    reward, active, event = env.step(torch.tensor([0]), torch.tensor([1.]), .05)
    assert env.caught.item() and active.item() and reward.item() < -7
    pos = env.pos.clone()
    reward, active, _ = env.step(torch.tensor([0]), torch.tensor([1.]), .05)
    assert not active.item() and reward.item() == 0
    torch.testing.assert_close(pos, env.pos)
    env = Grid(1, 0); env.changes[:] = True; env.flip[:] = 1
    risk = env.risk.clone()
    env.step(torch.tensor([2]), torch.tensor([0.]), .05)
    torch.testing.assert_close(env.risk, 1-risk)


def test_exact_temporal_policy_replay_and_sender_gradient():
    torch.manual_seed(31)
    for memory in ('gru', 'graph', 'memoryless'):
        model = Pair(memory)
        with torch.no_grad():
            batch = rollout(model, 'gru', Grid(16, 7), .05)
        lp, values, q, belief = replay(model, 'gru', batch['replay'])
        torch.testing.assert_close(lp, batch['logp'])
        torch.testing.assert_close(values, batch['value'])
        (-lp[batch['mask']].mean()).backward()
        assert model.gate[0].weight.grad.abs().sum() > 0
        assert model.encoder[0].weight.grad.abs().sum() > 0


def test_memory_survives_silence_and_navigation_is_contextual():
    model = Pair('graph')
    h = model.initial(2, 'cpu')
    h[:, 0] = torch.tensor([.1, .9])
    new, belief = model.receive(h, torch.tensor([1, 3]), torch.zeros(2))
    torch.testing.assert_close(belief, h[:, 0])
    a = model.actor(torch.cat((torch.zeros(2, 6), belief[:, None]), -1))
    b = model.actor(torch.cat((torch.ones(2, 6), belief[:, None]), -1))
    assert not torch.allclose(a, b)


def test_terminal_advantages_do_not_include_absorbing_tail():
    batch = dict(reward=torch.tensor([[2.], [0.], [0.]]),
                 value=torch.zeros(3, 1), mask=torch.tensor([[True], [False], [False]]))
    adv, returns = targets(batch)
    torch.testing.assert_close(adv[:, 0], torch.tensor([2., 0., 0.]))


def test_fork_cost_isolated_when_partner_ignores_messages():
    import copy
    from experiments.event_communication.paired_gate import act
    model = Pair('graph')
    with torch.no_grad():
        for p in model.actor.parameters():
            p.zero_()
        model.actor[-1].bias[0] = 1
    env = Grid(2, 41, scenario='static')
    env.risk[:] = 0
    initial_position = env.pos.clone()
    totals=[]
    for choice in (0., 1.):
        fork=copy.deepcopy(env)
        h=model.initial(2, 'cpu')
        total=torch.zeros(2)
        with torch.no_grad():
            for t in range(24):
                h,reward=act(model,fork,h,torch.full((2,),choice if t==0 else 0.))
                total+=reward
        assert fork.reached.all()
        totals.append(total)
    torch.testing.assert_close(totals[1]-totals[0],torch.full((2,),-.05),atol=1e-5,rtol=1e-5)
    assert env.t == 0 and not env.done.any()
    torch.testing.assert_close(initial_position,env.pos)


def test_rotated_actions_follow_public_coordinate_frame():
    env=Grid(1,2,scenario='rotated')
    env.pos[:]=torch.tensor([2,3]); env.changes[:]=False
    before=env.observation().clone()
    env.step(torch.tensor([0]),torch.zeros(1),.05)
    after=env.observation()
    # Public east advances own x and reduces both eastward relative offsets.
    torch.testing.assert_close(after[0,4]-before[0,4],torch.tensor(.1))
    torch.testing.assert_close(after[0,0]-before[0,0],torch.tensor(-.1))
