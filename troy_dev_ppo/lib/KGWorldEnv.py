import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "kg_world.xml")
)

CAMERA_NAMES = ["cam_front", "cam_left", "cam_back", "cam_right"]

FOOD_REACH_RADIUS = 0.8
LION_CATCH_RADIUS = 0.8

STEP_PENALTY = -0.01
FOOD_REWARD = 10.0
LION_PENALTY = -50.0
# potential-based shaping: reward getting closer to food, so the sparse +10
# is discoverable; policy-invariant and carries no caged/loose information
FOOD_SHAPING = 0.5

ARENA_HALF = 7.0  # object placement stays inside this
POS_SCALE = 8.0  # normalization for positions in the observation

# ground-plane range estimates are clamped here (arena diagonal is ~19.8)
MAX_GROUND_RANGE = 20.0


class KGWorldEnv(gym.Env):
    """Flat arena with a velocity-controlled ball agent, a lion, a cage, and food.

    Observation (13,): agent xy, yaw cos/sin, qvel(vx, vy, wyaw),
    lion/food/cage positions relative to the agent (world frame, /POS_SCALE).
    The observation deliberately does NOT contain the caged/loose flag --
    the agent can only learn that distinction through the KG symbol triples.

    Action (3,): world-frame [vx, vy, yaw_rate] in [-1, 1], scaled to ctrlrange.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_size=512,
        frame_skip=5,
        max_steps=300,
        lion_caged_prob=0.5,
        food_near_lion_prob=0.0,  # prob. food spawns on the lion (conflict layout)
        food_near_lion_offset=0.0,  # 0 = exact overlap (oracle); >0 = perceivable gap
        fixed_cage_state=None,  # True/False to pin the scenario, None to randomize
        seed=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.render_size = render_size

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.lion_caged_prob = lion_caged_prob
        self.food_near_lion_prob = food_near_lion_prob
        self.food_near_lion_offset = food_near_lion_offset
        self.fixed_cage_state = fixed_cage_state
        self.rng = np.random.default_rng(seed)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._ctrl_scale = self.model.actuator_ctrlrange[:, 1].copy()

        self.lion_caged = True
        self.food_near_lion = False
        self._step_count = 0

    # ------------------------------------------------------------------ setup

    def _sample_layout(self):
        """Choose cage/food/lion/agent positions with minimum separations.

        The lion is placed before the food so a conflict layout can co-locate
        the food with the lion. In that case reaching the food is safe only
        when the lion is caged -- with a loose lion the food sits inside the
        kill zone, so the correct behaviour flips on the caged/loose symbol.
        """
        cage_xy = self.rng.uniform(-4.5, 4.5, size=2)

        if self.fixed_cage_state is not None:
            self.lion_caged = bool(self.fixed_cage_state)
        else:
            self.lion_caged = bool(self.rng.random() < self.lion_caged_prob)
        self.food_near_lion = bool(self.rng.random() < self.food_near_lion_prob)

        if self.lion_caged:
            lion_xy = cage_xy.copy()
        else:
            while True:
                lion_xy = self.rng.uniform(-ARENA_HALF + 1, ARENA_HALF - 1, size=2)
                if np.linalg.norm(lion_xy - cage_xy) > 3.0:
                    break

        if self.food_near_lion:
            # food on/near the lion: safe to grab iff the lion is caged. A gap
            # keeps both objects separately perceivable (SAM occlusion) while
            # the food-reach and kill zones still overlap so a loose lion is
            # dangerous; offset 0 reproduces the oracle's exact overlap.
            if self.food_near_lion_offset > 0:
                theta = self.rng.uniform(-np.pi, np.pi)
                food_xy = lion_xy + self.food_near_lion_offset * np.array(
                    [np.cos(theta), np.sin(theta)]
                )
                food_xy = np.clip(food_xy, -ARENA_HALF + 0.5, ARENA_HALF - 0.5)
            else:
                food_xy = lion_xy.copy()
        else:
            while True:
                food_xy = self.rng.uniform(-ARENA_HALF + 1, ARENA_HALF - 1, size=2)
                if (
                    np.linalg.norm(food_xy - cage_xy) > 3.0
                    and np.linalg.norm(food_xy - lion_xy) > 2.5
                ):
                    break

        while True:
            agent_xy = self.rng.uniform(-ARENA_HALF + 1, ARENA_HALF - 1, size=2)
            if (
                np.linalg.norm(agent_xy - cage_xy) > 3.0
                and np.linalg.norm(agent_xy - food_xy) > 3.0
                and np.linalg.norm(agent_xy - lion_xy) > 3.0
            ):
                break

        return agent_xy, lion_xy, food_xy, cage_xy

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        agent_xy, lion_xy, food_xy, cage_xy = self._sample_layout()

        self.model.body("lion").pos[:2] = lion_xy
        self.model.body("food").pos[:2] = food_xy
        self.model.body("cage").pos[:2] = cage_xy

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = agent_xy
        self.data.qpos[2] = self.rng.uniform(-np.pi, np.pi)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist_food = self._get_info()["dist_to_food"]
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------ core

    def _apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action * self._ctrl_scale
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        self._apply_action(action)
        self._step_count += 1

        info = self._get_info()
        reward = STEP_PENALTY
        reward += FOOD_SHAPING * (self._prev_dist_food - info["dist_to_food"])
        self._prev_dist_food = info["dist_to_food"]
        terminated = False

        # A loose lion in range kills before the food can be collected, so
        # food co-located with a loose lion is a death trap (checked first);
        # a caged lion is harmless and the food is safe to grab. When the lion
        # is caged or >= its catch radius away, this reduces to the plain
        # food-reach check, so non-conflict layouts are unaffected.
        if not self.lion_caged and info["dist_to_lion"] < LION_CATCH_RADIUS:
            reward += LION_PENALTY
            terminated = True
            info["lion_caught"] = True
        elif info["dist_to_food"] < FOOD_REACH_RADIUS:
            reward += FOOD_REWARD
            terminated = True
            info["food_reached"] = True

        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        agent_xy = self.data.qpos[0:2]
        yaw = self.data.qpos[2]
        qvel = self.data.qvel[0:3]
        lion_rel = (self.model.body("lion").pos[:2] - agent_xy) / POS_SCALE
        food_rel = (self.model.body("food").pos[:2] - agent_xy) / POS_SCALE
        cage_rel = (self.model.body("cage").pos[:2] - agent_xy) / POS_SCALE
        return np.concatenate(
            [
                agent_xy / POS_SCALE,
                [np.cos(yaw), np.sin(yaw)],
                qvel,
                lion_rel,
                food_rel,
                cage_rel,
            ]
        )

    def _get_info(self):
        agent_xy = self.data.qpos[0:2]
        return {
            "dist_to_lion": float(np.linalg.norm(agent_xy - self.model.body("lion").pos[:2])),
            "dist_to_food": float(np.linalg.norm(agent_xy - self.model.body("food").pos[:2])),
            "dist_to_cage": float(np.linalg.norm(agent_xy - self.model.body("cage").pos[:2])),
            "lion_caged": self.lion_caged,
            "food_near_lion": self.food_near_lion,
            "food_reached": False,
            "lion_caught": False,
        }

    # ------------------------------------------------------------------ vision

    def render_panorama(self, with_segmentation=False):
        """Render all 4 agent-mounted cameras. Returns list of HxWx3 uint8 arrays.

        with_segmentation=True returns a list of (frame, seg) pairs instead,
        where seg is HxWx2 int32 of (object id, object type) -- MuJoCo's own
        record of which geom painted each pixel. The segmentation pass reuses
        the same updated scene, so the two images are pixel-aligned. This is
        ground truth for evaluation only; the agent never sees it.
        """
        frames = []
        for cam in CAMERA_NAMES:
            self.renderer.update_scene(self.data, camera=cam)
            frame = self.renderer.render().copy()
            if not with_segmentation:
                frames.append(frame)
                continue
            self.renderer.enable_segmentation_rendering()
            try:
                seg = self.renderer.render().copy()
            finally:
                self.renderer.disable_segmentation_rendering()
            frames.append((frame, seg))
        return frames

    def close(self):
        self.renderer.close()


class KinematicKGWorldEnv(KGWorldEnv):
    """Physics-free phase-2 variant: 4 discrete unit moves, no momentum.

    MuJoCo is kept purely as a renderer -- qpos is written directly and
    mj_forward recomputes camera poses; mj_step is never called, so there is
    no inertia, no servo lag, and an action's displacement is exact.

    Moves are egocentric relative to the yaw drawn at reset (there is no turn
    action, so facing is fixed for the whole episode and the cameras keep a
    constant orientation).

    Observation (4,): [x/POS_SCALE, y/POS_SCALE, cos(yaw), sin(yaw)].
    Object positions are deliberately absent -- they reach the policy only
    through the KG symbol triples and their ground-plane distance estimates.

    Action: Discrete(4) -- 0 forward, 1 backward, 2 left, 3 right.
    """

    def __init__(self, step_size=1.0, **kwargs):
        super().__init__(**kwargs)
        self.step_size = step_size
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def _apply_action(self, action):
        yaw = self.data.qpos[2]
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        move = (fwd, -fwd, left, -left)[int(action)] * self.step_size
        self.data.qpos[0:2] = np.clip(self.data.qpos[0:2] + move, -ARENA_HALF, ARENA_HALF)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        yaw = self.data.qpos[2]
        return np.concatenate(
            [self.data.qpos[0:2] / POS_SCALE, [np.cos(yaw), np.sin(yaw)]]
        )

    # ------------------------------------------------- ground-plane geometry

    def ground_delta(self, cam_name, px, py):
        """World-frame (dx, dy) from the agent to the floor point seen at pixel
        (px, py) of `cam_name`, divided by POS_SCALE.

        Objects rest on the floor, so casting a ray through an object's
        bounding-box bottom-center and intersecting the ground plane recovers
        its position from vision alone (no oracle body positions). Rays at or
        above the horizon, and hits beyond MAX_GROUND_RANGE, are clamped to
        MAX_GROUND_RANGE along the ray's horizontal bearing.
        """
        cam_id = self.model.camera(cam_name).id
        cam_pos = self.data.cam_xpos[cam_id]
        rot = self.data.cam_xmat[cam_id].reshape(3, 3)  # columns: right, up, -forward

        w = h = self.render_size
        half_tan = np.tan(np.deg2rad(self.model.cam_fovy[cam_id]) / 2.0)
        u = (2.0 * (px + 0.5) / w - 1.0) * half_tan  # square image: fovx == fovy
        v = (1.0 - 2.0 * (py + 0.5) / h) * half_tan
        ray = rot @ np.array([u, v, -1.0])  # camera looks along its local -z

        agent_xy = self.data.qpos[0:2]
        horiz = np.linalg.norm(ray[:2])
        if horiz < 1e-9:
            return np.zeros(2)
        if ray[2] < -1e-6:
            t = cam_pos[2] / -ray[2]
            delta = cam_pos[:2] + t * ray[:2] - agent_xy
        else:
            delta = ray[:2] / horiz * MAX_GROUND_RANGE
        dist = np.linalg.norm(delta)
        if dist > MAX_GROUND_RANGE:
            delta = delta / dist * MAX_GROUND_RANGE
        return delta / POS_SCALE
