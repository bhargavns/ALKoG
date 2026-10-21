"""First oracle MuJoCo transfer: frozen safety code and learned temporal control.

Like the original troy_dev arena, movement is kinematic (position updates plus
mj_forward), not torque-driven locomotion. The goal marker is public. The message
communicates the safety response for that goal, not its location or species.
"""
import os
import sys
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')

import numpy as np
import mujoco
import torch
from torch import nn
from torch.distributions import Categorical

from .task import Episodes, SHEEP
from communication.CommunicationChannel import StraightThroughGumbelChannel

ARENA = 3.
STEP = .2
REACH = .25
MAX_STEPS = 80

XML = '''<mujoco model="entity_safety_transfer">
  <option gravity="0 0 0"/>
  <visual><global offwidth="512" offheight="512"/></visual>
  <worldbody>
    <light pos="0 0 6"/>
    <camera name="top" pos="0 0 9"/>
    <geom name="floor" type="plane" size="3.2 3.2 .1" rgba=".15 .18 .2 1"/>
    <body name="lion"><geom type="sphere" size=".15" rgba=".9 .65 .15 1"/></body>
    <body name="tiger"><geom type="sphere" size=".15" rgba="1 .35 .05 1"/></body>
    <body name="sheep"><geom type="sphere" size=".15" rgba=".95 .95 .95 1"/></body>
    <body name="receiver"><geom type="sphere" size=".1" rgba=".1 .65 1 1"/></body>
    <body name="goal"><geom type="cylinder" size=".3 .015" rgba=".1 1 .3 .35"/></body>
  </worldbody>
</mujoco>'''


class OracleNavigationEnv:
    """Three static animal markers, a public goal, and a moving receiver.

    Initial straight paths to the goal are clear of distractors. This first
    transfer tests elementary control, not planning around hidden hazards.
    Caging/safety are oracle flags, as in the original arena; marker colors and
    these flags are never input to the low-level movement policy.
    """

    def __init__(self, seed=0):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.bodies = [self.model.body(name).id for name in ('lion', 'tiger', 'sheep')]
        self.receiver = self.model.body('receiver').id
        self.goal = self.model.body('goal').id
        self.steps = 0

    @property
    def xy(self):
        return self.model.body_pos[self.receiver, :2].copy()

    @property
    def dangerous(self):
        return self.entity != SHEEP and not self.cages[self.entity]

    def reset(self, case=None, force_safe=False):
        if case is None:
            case = int(self.rng.integers(6))
        self.entity = case // 2
        self.cages = self.rng.integers(2, size=3).astype(bool)
        self.cages[self.entity] = bool(case % 2)
        if force_safe:
            self.cages[self.entity] = True
        # Goal and start independent of category/state, within the audited range.
        while True:
            start = self.rng.uniform(-2.5, 2.5, 2)
            goal = self.rng.uniform(-2.5, 2.5, 2)
            if 1.5 < np.linalg.norm(goal - start) < 4.5:
                break
        points = []
        segment = goal - start
        for entity in range(3):
            if entity == self.entity:
                point = goal
            else:
                for _ in range(10000):
                    point = self.rng.uniform(-2.6, 2.6, 2)
                    t = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0, 1)
                    if (np.linalg.norm(point - start - t * segment) > 1.
                            and all(np.linalg.norm(point - p) > .8 for p in points)):
                        break
                else:
                    raise RuntimeError('Could not sample a clear-path scene')
            points.append(point)
        for body, point in zip(self.bodies, points):
            self.model.body_pos[body] = [*point, .15]
        self.model.body_pos[self.receiver] = [*start, .1]
        self.model.body_pos[self.goal] = [*goal, .02]
        self.target_xy = goal.copy()
        self.steps = 0
        self.previous_distance = np.linalg.norm(goal - start)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self):
        # Public target offset and own position; no category, cage or safety label.
        return ((self.target_xy - self.xy) / (2 * ARENA)).astype(np.float32)

    def episode(self):
        # The frozen communication decoder retains its original candidate interface.
        return Episodes(torch.arange(3)[None],
                        torch.tensor(self.model.body_pos[self.bodies, :2] / ARENA,
                                     dtype=torch.float32)[None],
                        torch.tensor(self.cages)[None], torch.tensor([self.entity]))

    def step(self, action):
        movement = np.array(((1, 0), (-1, 0), (0, 1), (0, -1)))[int(action)] * STEP
        self.model.body_pos[self.receiver, :2] = np.clip(self.xy + movement, -ARENA, ARENA)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        distance = np.linalg.norm(self.target_xy - self.xy)
        reward = float(2 * (self.previous_distance - distance) - .01)
        self.previous_distance = distance
        danger = [i for i in range(3) if i != SHEEP and not self.cages[i]]
        caught = any(np.linalg.norm(self.xy - self.model.body_pos[self.bodies[i], :2]) < REACH
                     for i in danger)
        reached = distance < REACH and not caught
        terminated = caught or reached
        timeout = self.steps >= MAX_STEPS and not terminated
        reward += -10 if caught else 5 if reached else 0
        return self.observation(), reward, terminated or timeout, dict(
            reached=bool(reached), caught=bool(caught), timeout=bool(timeout))


@torch.no_grad()
def safety_decision(sender, receiver, env):
    episode = env.episode()
    channel = StraightThroughGumbelChannel(n_symbols=sender.n_symbols)
    message = channel(sender(episode.sender_input()), hard=True)
    dist, _ = receiver.distribution_messages(message, episode.receiver_input())
    response = int(dist.logits.argmax(-1)[0] % 2)
    return response, message.argmax(-1)[0].tolist()


class MovementActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))
        self.critic = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observations):
        return Categorical(logits=self.actor(observations)), self.critic(observations).squeeze(-1)


def advantages(rewards, dones, values, bootstrap, gamma=.99, decay=.95):
    output = torch.zeros_like(rewards)
    running = 0.
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        continuation = 1 - dones[t]
        delta = rewards[t] + gamma * next_value * continuation - values[t]
        running = delta + gamma * decay * continuation * running
        output[t] = running
    return output, output + values


def train_movement(seed, iterations=100, horizon=512):
    """PPO over actual ordered MuJoCo transitions; safety communication is frozen.

    Movement is trained on safe-goal episodes. Warnings terminate before movement
    at evaluation. All six target cases are evaluated separately afterward.
    """
    torch.manual_seed(seed)
    env = OracleNavigationEnv(seed)
    policy = MovementActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    observation = env.reset(force_safe=True)
    curve = []
    for iteration in range(iterations):
        states, actions, logps, values, rewards, dones = [], [], [], [], [], []
        completed, successes = 0, 0
        for _ in range(horizon):
            state = torch.tensor(observation)
            with torch.no_grad():
                distribution, value = policy(state)
                action = distribution.sample()
            observation, reward, done, info = env.step(action.item())
            states.append(state)
            actions.append(action)
            logps.append(distribution.log_prob(action))
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))
            if done:
                completed += 1
                successes += int(info['reached'])
                observation = env.reset(force_safe=True)
        with torch.no_grad():
            _, bootstrap = policy(torch.tensor(observation))
        states, actions, old_logp, values = map(torch.stack, (states, actions, logps, values))
        advantage, returns = advantages(torch.tensor(rewards), torch.tensor(dones), values, bootstrap)
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        for _ in range(4):
            distribution, prediction = policy(states)
            ratio = (distribution.log_prob(actions) - old_logp).exp()
            loss = (-torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
                    + .5 * (prediction - returns).square().mean() - .01 * distribution.entropy().mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
        curve.append(dict(iteration=iteration + 1, environment_steps=(iteration + 1) * horizon,
                          completed_episodes=completed, reached_episodes=successes, loss=loss.item()))
    return policy, curve


@torch.no_grad()
def evaluate_navigation(sender, receiver, movement, seed, layouts_per_case=100):
    env = OracleNavigationEnv(seed)
    records = []
    for layout in range(layouts_per_case):
        for case in range(6):
            observation = env.reset(case)
            response, message = safety_decision(sender, receiver, env)
            expected_warn = env.dangerous
            record = dict(layout=layout, case=case, target_entity=case // 2,
                          caged=bool(case % 2), expected_warn=bool(expected_warn),
                          response='WARN' if response else 'APPROACH', message=message,
                          candidates=env.model.body_pos[env.bodies, :2].tolist(),
                          cages=env.cages.tolist(), goal=env.target_xy.tolist(),
                          positions=[env.xy.tolist()], actions=[], rewards=[])
            if response:
                record.update(success=bool(expected_warn), false_warning=not bool(expected_warn),
                              reached=False, caught=False, timeout=False)
            else:
                done = False
                while not done:
                    distribution, _ = movement(torch.tensor(observation))
                    action = int(distribution.logits.argmax(-1))
                    observation, reward, done, info = env.step(action)
                    record['positions'].append(env.xy.tolist())
                    record['actions'].append(action)
                    record['rewards'].append(reward)
                record.update(success=bool(info['reached'] and not expected_warn),
                              false_warning=False, **info)
            records.append(record)
    return dict(episodes=len(records), successes=sum(x['success'] for x in records),
                safety_response_errors=sum((x['response'] == 'WARN') != x['expected_warn'] for x in records),
                catches=sum(x['caught'] for x in records),
                timeouts=sum(x['timeout'] for x in records),
                success_rate=sum(x['success'] for x in records) / len(records)), records
