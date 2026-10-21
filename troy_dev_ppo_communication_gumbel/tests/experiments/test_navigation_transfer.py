"""First MuJoCo transfer: geometry, reward and temporal-credit contracts."""
import importlib
import sys

import numpy as np
import torch


def module(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')
    return importlib.import_module('experiments.entity_safety.navigation')


def test_mujoco_transition_moves_receiver_and_rewards_safe_goal(monkeypatch):
    nav = module(monkeypatch)
    env = nav.OracleNavigationEnv(1)
    env.reset(case=1)  # caged lion
    env.model.body_pos[env.receiver, :2] = env.target_xy + np.array([.21, 0])
    env.previous_distance = .21
    before = env.xy
    _, reward, done, info = env.step(1)
    assert np.allclose(env.xy, before + [-nav.STEP, 0])
    assert done and info['reached'] and not info['caught'] and reward > 0
    assert np.allclose(env.data.xpos[env.receiver, :2], env.xy)


def test_loose_predator_contact_is_a_failure(monkeypatch):
    nav = module(monkeypatch)
    env = nav.OracleNavigationEnv(2)
    env.reset(case=0)
    env.model.body_pos[env.receiver, :2] = env.target_xy + np.array([.21, 0])
    env.previous_distance = .21
    _, reward, done, info = env.step(1)
    assert done and info['caught'] and not info['reached'] and reward < 0


def test_public_movement_observation_does_not_reveal_private_safety(monkeypatch):
    nav = module(monkeypatch)
    env = nav.OracleNavigationEnv(3)
    observation = env.reset(case=0)
    env.cages = ~env.cages
    assert np.array_equal(observation, env.observation())
    assert env.episode().receiver_input().shape == (1, 3, 5)


def test_terminal_mask_prevents_cross_episode_credit(monkeypatch):
    nav = module(monkeypatch)
    advantage, returns = nav.advantages(torch.tensor([1., 10.]), torch.tensor([1., 1.]),
                                       torch.zeros(2), torch.tensor(100.))
    assert torch.equal(advantage, torch.tensor([1., 10.]))
    assert torch.equal(returns, advantage)
    continuing, _ = nav.advantages(torch.tensor([0., 1.]), torch.tensor([0., 1.]),
                                   torch.zeros(2), torch.tensor(0.), gamma=1., decay=1.)
    assert torch.equal(continuing, torch.tensor([1., 1.]))


def test_simulator_runner_rejects_failed_communication_gate_before_output(tmp_path, monkeypatch):
    import json
    import pytest
    from experiments.entity_safety.run_navigation import main
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'accepted': False}))
    output = tmp_path / 'not_created'
    monkeypatch.setattr(sys, 'argv', ['run_navigation', '--acceptance', str(gate), '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


def test_simulator_runner_rejects_changed_accepted_checkpoint(tmp_path, monkeypatch):
    import json
    import pytest
    from experiments.entity_safety.run_navigation import main
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'changed')
    gate = tmp_path / 'gate.json'
    row = dict(accepted=True, checkpoint=str(checkpoint), checkpoint_sha256='old_hash')
    gate.write_text(json.dumps(dict(accepted=True, runs={str(seed): row for seed in range(5)})))
    output = tmp_path / 'not_created'
    monkeypatch.setattr(sys, 'argv', ['run_navigation', '--acceptance', str(gate), '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
