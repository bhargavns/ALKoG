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
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

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

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        torso_xy = self.data.body("torso").xpos[:2]
        lion_xy = self.data.body("lion").xpos[:2]
        food_xy = self.data.body("food").xpos[:2]

        dist_lion = float(np.linalg.norm(torso_xy - lion_xy))
        dist_food = float(np.linalg.norm(torso_xy - food_xy))

        info["dist_to_lion"] = dist_lion
        info["dist_to_food"] = dist_food
        info["lion_caught"] = dist_lion < LION_CATCH_RADIUS
        info["food_reached"] = dist_food < FOOD_REACH_RADIUS

        return obs, reward, terminated, truncated, info
