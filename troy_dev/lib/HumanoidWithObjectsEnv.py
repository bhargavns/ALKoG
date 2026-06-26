import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv, DEFAULT_CAMERA_CONFIG

_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid_with_objects.xml")
)

LION_CATCH_RADIUS = 0.8
FOOD_REACH_RADIUS = 0.8


class HumanoidWithObjectsEnv(HumanoidEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        food_progress_reward_weight=1.0,
        lion_avoidance_reward_weight=0.75,
        food_reach_bonus=25.0,
        lion_catch_penalty=-25.0,
        terminate_on_food_reach=False,
        terminate_on_lion_catch=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            food_progress_reward_weight,
            lion_avoidance_reward_weight,
            food_reach_bonus,
            lion_catch_penalty,
            terminate_on_food_reach,
            terminate_on_lion_catch,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._food_progress_reward_weight = food_progress_reward_weight
        self._lion_avoidance_reward_weight = lion_avoidance_reward_weight
        self._food_reach_bonus = food_reach_bonus
        self._lion_catch_penalty = lion_catch_penalty
        self._terminate_on_food_reach = terminate_on_food_reach
        self._terminate_on_lion_catch = terminate_on_lion_catch
        self._prev_dist_food = None
        self._prev_dist_lion = None

        obs_shape = 376 if exclude_current_positions_from_observation else 378
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            _XML_PATH,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Gymnasium Humanoid observation composition can vary by version;
        # derive the true shape from the live simulator state.
        actual_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=actual_obs.shape,
            dtype=np.float64,
        )

    def _distance_metrics(self):
        torso_xy = self.data.body("torso").xpos[:2]
        lion_xy = self.data.body("lion").xpos[:2]
        food_xy = self.data.body("food").xpos[:2]

        dist_lion = float(np.linalg.norm(torso_xy - lion_xy))
        dist_food = float(np.linalg.norm(torso_xy - food_xy))
        return dist_lion, dist_food

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        dist_lion, dist_food = self._distance_metrics()
        self._prev_dist_lion = dist_lion
        self._prev_dist_food = dist_food
        info["dist_to_lion"] = dist_lion
        info["dist_to_food"] = dist_food
        info["lion_caught"] = dist_lion < LION_CATCH_RADIUS
        info["food_reached"] = dist_food < FOOD_REACH_RADIUS
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)

        dist_lion, dist_food = self._distance_metrics()

        prev_dist_lion = dist_lion if self._prev_dist_lion is None else self._prev_dist_lion
        prev_dist_food = dist_food if self._prev_dist_food is None else self._prev_dist_food

        # Reward moving toward food and away from lion between timesteps.
        food_progress = prev_dist_food - dist_food
        lion_avoidance = dist_lion - prev_dist_lion

        shaped_reward = (
            self._food_progress_reward_weight * food_progress
            + self._lion_avoidance_reward_weight * lion_avoidance
        )

        lion_caught = dist_lion < LION_CATCH_RADIUS
        food_reached = dist_food < FOOD_REACH_RADIUS

        if food_reached:
            shaped_reward += self._food_reach_bonus
            if self._terminate_on_food_reach:
                terminated = True

        if lion_caught:
            shaped_reward += self._lion_catch_penalty
            if self._terminate_on_lion_catch:
                terminated = True

        reward = float(base_reward + shaped_reward)

        self._prev_dist_lion = dist_lion
        self._prev_dist_food = dist_food

        info["dist_to_lion"] = dist_lion
        info["dist_to_food"] = dist_food
        info["lion_caught"] = lion_caught
        info["food_reached"] = food_reached
        info["base_reward"] = float(base_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["total_reward"] = reward

        return obs, reward, terminated, truncated, info
