import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from lib.ObjectCategories import BACKGROUND_ID, BODY_TO_CATEGORY

_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "kg_world.xml")
)

CAMERA_NAMES = ["cam_front", "cam_left", "cam_back", "cam_right"]

FOOD_REACH_RADIUS = 0.8
LION_CATCH_RADIUS = 0.8
WATER_REACH_RADIUS = 0.8
POISON_REACH_RADIUS = 0.8

STEP_PENALTY = -0.01
FOOD_REWARD = 10.0
WATER_REWARD = 6.0
LION_PENALTY = -50.0
POISON_PENALTY = -20.0
# Shape progress toward the nearer positive resource.  The terminal rewards
# still make food preferable to water when both are equally reachable.
RESOURCE_SHAPING = 0.5

ARENA_HALF = 7.0  # object placement stays inside this
POS_SCALE = 8.0  # normalization for positions in the observation

# ground-plane range estimates are clamped here (arena diagonal is ~19.8)
MAX_GROUND_RANGE = 20.0


class KGWorldEnv(gym.Env):
    """Flat arena with five visually grounded object categories.

    Observation (17,): agent xy, yaw cos/sin, qvel(vx, vy, wyaw), followed by
    lion/food/cage/water/poison positions relative to the agent.  This verbose
    observation is used only by the phase-1/debug environment.
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
        self.fixed_cage_state = fixed_cage_state
        self.rng = np.random.default_rng(seed)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._ctrl_scale = self.model.actuator_ctrlrange[:, 1].copy()

        self.lion_caged = True
        self._step_count = 0

    # ------------------------------------------------------------------ setup

    def _sample_position(self, existing, low=None, high=None, min_sep=2.5):
        low = -ARENA_HALF + 1 if low is None else low
        high = ARENA_HALF - 1 if high is None else high
        for _ in range(10_000):
            candidate = self.rng.uniform(low, high, size=2)
            if all(np.linalg.norm(candidate - other) >= min_sep for other in existing):
                return candidate
        raise RuntimeError("Could not sample a non-overlapping object layout")

    def _sample_layout(self):
        """Choose five object positions with conservative minimum separations."""
        cage_xy = self.rng.uniform(-4.5, 4.5, size=2)
        placed = [cage_xy]
        food_xy = self._sample_position(placed, min_sep=3.0)
        placed.append(food_xy)
        water_xy = self._sample_position(placed, min_sep=2.5)
        placed.append(water_xy)
        poison_xy = self._sample_position(placed, min_sep=2.5)
        placed.append(poison_xy)

        if self.fixed_cage_state is not None:
            self.lion_caged = bool(self.fixed_cage_state)
        else:
            self.lion_caged = bool(self.rng.random() < self.lion_caged_prob)

        agent_xy = self._sample_position(placed, min_sep=3.0)

        if self.lion_caged:
            lion_xy = cage_xy.copy()
        else:
            lion_xy = self._sample_position(placed + [agent_xy], min_sep=3.0)

        return {
            "agent": agent_xy,
            "lion": lion_xy,
            "food": food_xy,
            "cage": cage_xy,
            "water": water_xy,
            "poison": poison_xy,
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        layout = self._sample_layout()

        for body_name in BODY_TO_CATEGORY:
            self.model.body(body_name).pos[:2] = layout[body_name]

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = layout["agent"]
        self.data.qpos[2] = self.rng.uniform(-np.pi, np.pi)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        info = self._get_info()
        self._prev_resource_dist = min(info["dist_to_food"], info["dist_to_water"])
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
        resource_dist = min(info["dist_to_food"], info["dist_to_water"])
        reward += RESOURCE_SHAPING * (self._prev_resource_dist - resource_dist)
        self._prev_resource_dist = resource_dist
        terminated = False

        if info["dist_to_food"] < FOOD_REACH_RADIUS:
            reward += FOOD_REWARD
            terminated = True
            info["food_reached"] = True
        elif info["dist_to_water"] < WATER_REACH_RADIUS:
            reward += WATER_REWARD
            terminated = True
            info["water_reached"] = True
        elif info["dist_to_poison"] < POISON_REACH_RADIUS:
            reward += POISON_PENALTY
            terminated = True
            info["poison_touched"] = True
        elif not self.lion_caged and info["dist_to_lion"] < LION_CATCH_RADIUS:
            reward += LION_PENALTY
            terminated = True
            info["lion_caught"] = True

        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        agent_xy = self.data.qpos[0:2]
        yaw = self.data.qpos[2]
        qvel = self.data.qvel[0:3]
        lion_rel = (self.model.body("lion").pos[:2] - agent_xy) / POS_SCALE
        food_rel = (self.model.body("food").pos[:2] - agent_xy) / POS_SCALE
        cage_rel = (self.model.body("cage").pos[:2] - agent_xy) / POS_SCALE
        water_rel = (self.model.body("water").pos[:2] - agent_xy) / POS_SCALE
        poison_rel = (self.model.body("poison").pos[:2] - agent_xy) / POS_SCALE
        return np.concatenate(
            [
                agent_xy / POS_SCALE,
                [np.cos(yaw), np.sin(yaw)],
                qvel,
                lion_rel,
                food_rel,
                cage_rel,
                water_rel,
                poison_rel,
            ]
        )

    def _get_info(self):
        agent_xy = self.data.qpos[0:2]
        return {
            "dist_to_lion": float(np.linalg.norm(agent_xy - self.model.body("lion").pos[:2])),
            "dist_to_food": float(np.linalg.norm(agent_xy - self.model.body("food").pos[:2])),
            "dist_to_cage": float(np.linalg.norm(agent_xy - self.model.body("cage").pos[:2])),
            "dist_to_water": float(np.linalg.norm(agent_xy - self.model.body("water").pos[:2])),
            "dist_to_poison": float(np.linalg.norm(agent_xy - self.model.body("poison").pos[:2])),
            "lion_caged": self.lion_caged,
            "food_reached": False,
            "water_reached": False,
            "poison_touched": False,
            "lion_caught": False,
        }

    # ------------------------------------------------------------------ vision

    def render_panorama(self):
        """Render all 4 agent-mounted cameras. Returns list of HxWx3 uint8 arrays."""
        frames = []
        for cam in CAMERA_NAMES:
            self.renderer.update_scene(self.data, camera=cam)
            frames.append(self.renderer.render().copy())
        return frames

    def _segmentation_to_categories(self, segmentation):
        """Convert MuJoCo's (object-id, object-type) image to category IDs.

        This is simulator-oracle information used by dataset generation and
        evaluation only.  The learned model never receives it at inference.
        """
        labels = np.full(segmentation.shape[:2], BACKGROUND_ID, dtype=np.int64)
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        # MuJoCo has used both documented channel descriptions across releases;
        # identify the type channel by the number of mjOBJ_GEOM pixels.
        first_is_type = int((segmentation[..., 0] == geom_type).sum()) > int(
            (segmentation[..., 1] == geom_type).sum()
        )
        if first_is_type:
            object_types, object_ids = segmentation[..., 0], segmentation[..., 1]
        else:
            object_ids, object_types = segmentation[..., 0], segmentation[..., 1]
        valid_geom_ids = np.unique(object_ids[object_types == geom_type])
        for geom_id in valid_geom_ids:
            geom_id = int(geom_id)
            if geom_id < 0 or geom_id >= self.model.ngeom:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            category_id = BODY_TO_CATEGORY.get(body_name)
            if category_id is not None:
                labels[(object_types == geom_type) & (object_ids == geom_id)] = category_id
        return labels

    def render_panorama_with_labels(self):
        """Return aligned RGB frames and oracle category masks for each camera."""
        rgb_frames = self.render_panorama()
        label_frames = []
        self.renderer.enable_segmentation_rendering()
        try:
            for cam in CAMERA_NAMES:
                self.renderer.update_scene(self.data, camera=cam)
                label_frames.append(self._segmentation_to_categories(self.renderer.render()))
        finally:
            self.renderer.disable_segmentation_rendering()
        return rgb_frames, label_frames

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
