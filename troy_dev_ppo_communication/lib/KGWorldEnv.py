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
WATER_REACH_RADIUS = 0.8
POISON_REACH_RADIUS = 0.8

STEP_PENALTY = -0.01
FOOD_REWARD = 10.0
LION_PENALTY = -50.0
WATER_REWARD = 6.0
POISON_PENALTY = -20.0
# potential-based shaping: reward getting closer to food, so the sparse +10
# is discoverable; policy-invariant and carries no caged/loose information.
# With resources=True the same coefficient shapes toward the nearer of
# food/water; the terminal rewards still make food the better target.
FOOD_SHAPING = 0.5
RESOURCE_SHAPING = FOOD_SHAPING

# Bodies parked here when resources=False: far below the floor, so the arena
# cameras never see them and SAM cannot propose them.
_PARKED_Z = -100.0

ARENA_HALF = 7.0  # object placement stays inside this
POS_SCALE = 8.0  # normalization for positions in the observation

# ground-plane range estimates are clamped here (arena diagonal is ~19.8)
MAX_GROUND_RANGE = 20.0


class KGWorldEnv(gym.Env):
    """Flat arena with a velocity-controlled ball agent, a lion, a cage, food,
    and a stationary receiving agent.

    Observation (13,): agent xy, yaw cos/sin, qvel(vx, vy, wyaw),
    lion/food/cage positions relative to the agent (world frame, /POS_SCALE).
    With resources=True this becomes (17,) -- water and poison are appended.
    The observation deliberately does NOT contain the caged/loose flag --
    the agent can only learn that distinction through the KG symbol triples.

    Action (3,): world-frame [vx, vy, yaw_rate] in [-1, 1], scaled to ctrlrange.

    resources=False (default) is the three-object world all prior results were
    produced on. Nothing about that path changes when the flag is off: water and
    poison are parked below the floor (invisible to the cameras and to SAM), the
    observation stays 13-dim, shaping stays food-only, and -- importantly -- no
    RNG is drawn for them, so a given seed reproduces the old layout stream
    exactly. resources=True adds water (+6) and poison (-20) for the
    soft-category work, which needs five visually distinct categories.

    The receiving agent ("receiver") is present in BOTH worlds and is inert: it
    is placed at reset, never moves, and contributes nothing to the observation,
    the reward, or termination. It exists so the transmitting agent can learn to
    identify it as its own category; it will act in a later phase. Its position
    is drawn last, so the agent/lion/food/cage layout for a given seed is
    unchanged from a pre-receiver run.
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
        resources=False,  # True adds water/poison (5-object soft-category world)
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
        self.resources = bool(resources)
        self.rng = np.random.default_rng(seed)

        obs_dim = 17 if self.resources else 13
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._ctrl_scale = self.model.actuator_ctrlrange[:, 1].copy()

        if not self.resources:
            self._park_resources()

        self.lion_caged = True
        self.food_near_lion = False
        self._step_count = 0

    def _park_resources(self):
        """Sink water/poison below the floor so they cannot be seen or reached."""
        for name in ("water", "poison"):
            self.model.body(name).pos[2] = _PARKED_Z

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

        layout = {
            "agent": agent_xy,
            "lion": lion_xy,
            "food": food_xy,
            "cage": cage_xy,
        }

        # Sampled last and only when enabled, so the draw sequence above is
        # byte-identical to the three-object env for any given seed.
        placed = [cage_xy, food_xy, lion_xy, agent_xy]
        if self.resources:
            layout["water"] = self._sample_position(placed, min_sep=2.5)
            placed.append(layout["water"])
            layout["poison"] = self._sample_position(placed, min_sep=2.5)
            placed.append(layout["poison"])

        return layout

    def _sample_position(self, existing, min_sep=2.5):
        """Uniform arena position at least `min_sep` from every placed object.

        Bounded retries: an over-constrained layout raises instead of hanging,
        which is how the caller finds out the arena is too crowded.
        """
        for _ in range(10_000):
            candidate = self.rng.uniform(-ARENA_HALF + 1, ARENA_HALF - 1, size=2)
            if all(np.linalg.norm(candidate - other) >= min_sep for other in existing):
                return candidate
        raise RuntimeError("Could not sample a non-overlapping object layout")

    def _sample_receiver_position(self, layout):
        """Place the receiving agent clear of every object already positioned.

        Called from reset AFTER the agent's yaw is drawn, not from
        _sample_layout, because the yaw draw follows _sample_layout: sampling
        the receiver any earlier would shift the yaw stream and change what the
        agent faces at reset for a given seed. Drawing it dead last leaves every
        RNG value a pre-receiver run consumed exactly as it was -- same layout,
        same facing -- so datasets and runs stay comparable across this change.

        The receiver is re-placed each episode so the transmitting agent has to
        recognise it at many ranges and bearings rather than memorising one
        spot. It never moves within an episode.
        """
        placed = [layout["cage"], layout["food"], layout["lion"], layout["agent"]]
        if self.resources:
            placed += [layout["water"], layout["poison"]]
        return self._sample_position(placed, min_sep=2.5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        layout = self._sample_layout()

        for name in ("lion", "food", "cage"):
            self.model.body(name).pos[:2] = layout[name]
        if self.resources:
            for name in ("water", "poison"):
                self.model.body(name).pos[:2] = layout[name]

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = layout["agent"]
        self.data.qpos[2] = self.rng.uniform(-np.pi, np.pi)
        self.model.body("receiver").pos[:2] = self._sample_receiver_position(layout)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_resource_dist = self._resource_dist(self._get_info())
        return self._get_obs(), self._get_info()

    def _resource_dist(self, info):
        """Distance to the nearest positive resource driving potential shaping."""
        if self.resources:
            return min(info["dist_to_food"], info["dist_to_water"])
        return info["dist_to_food"]

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
        resource_dist = self._resource_dist(info)
        reward += RESOURCE_SHAPING * (self._prev_resource_dist - resource_dist)
        self._prev_resource_dist = resource_dist
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
        # Checked after food/lion so the three-object outcome ordering is
        # untouched; unreachable at all when resources are parked below the floor.
        elif self.resources and info["dist_to_water"] < WATER_REACH_RADIUS:
            reward += WATER_REWARD
            terminated = True
            info["water_reached"] = True
        elif self.resources and info["dist_to_poison"] < POISON_REACH_RADIUS:
            reward += POISON_PENALTY
            terminated = True
            info["poison_touched"] = True

        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        agent_xy = self.data.qpos[0:2]
        yaw = self.data.qpos[2]
        qvel = self.data.qvel[0:3]
        lion_rel = (self.model.body("lion").pos[:2] - agent_xy) / POS_SCALE
        food_rel = (self.model.body("food").pos[:2] - agent_xy) / POS_SCALE
        cage_rel = (self.model.body("cage").pos[:2] - agent_xy) / POS_SCALE
        parts = [
            agent_xy / POS_SCALE,
            [np.cos(yaw), np.sin(yaw)],
            qvel,
            lion_rel,
            food_rel,
            cage_rel,
        ]
        if self.resources:
            parts.append((self.model.body("water").pos[:2] - agent_xy) / POS_SCALE)
            parts.append((self.model.body("poison").pos[:2] - agent_xy) / POS_SCALE)
        return np.concatenate(parts)

    def _get_info(self):
        agent_xy = self.data.qpos[0:2]
        info = {
            "dist_to_lion": float(np.linalg.norm(agent_xy - self.model.body("lion").pos[:2])),
            "dist_to_food": float(np.linalg.norm(agent_xy - self.model.body("food").pos[:2])),
            "dist_to_cage": float(np.linalg.norm(agent_xy - self.model.body("cage").pos[:2])),
            "lion_caged": self.lion_caged,
            "food_near_lion": self.food_near_lion,
            "food_reached": False,
            "lion_caught": False,
            # Reported for diagnostics only -- the receiving agent carries no
            # reward, no termination, and no shaping.
            "dist_to_receiver": float(
                np.linalg.norm(agent_xy - self.model.body("receiver").pos[:2])
            ),
        }
        if self.resources:
            info["dist_to_water"] = float(
                np.linalg.norm(agent_xy - self.model.body("water").pos[:2])
            )
            info["dist_to_poison"] = float(
                np.linalg.norm(agent_xy - self.model.body("poison").pos[:2])
            )
            info["water_reached"] = False
            info["poison_touched"] = False
        return info

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

    def _segmentation_to_categories(self, segmentation):
        """Convert MuJoCo's (object-id, object-type) image to category IDs.

        Simulator-oracle information for dataset generation and evaluation only;
        the learned category model never receives it at inference.
        """
        from lib.ObjectCategories import BACKGROUND_ID, BODY_TO_CATEGORY

        labels = np.full(segmentation.shape[:2], BACKGROUND_ID, dtype=np.int64)
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        # MuJoCo has shipped both channel orderings across releases; identify the
        # type channel by which one actually holds mjOBJ_GEOM values.
        first_is_type = int((segmentation[..., 0] == geom_type).sum()) > int(
            (segmentation[..., 1] == geom_type).sum()
        )
        if first_is_type:
            object_types, object_ids = segmentation[..., 0], segmentation[..., 1]
        else:
            object_ids, object_types = segmentation[..., 0], segmentation[..., 1]
        for geom_id in np.unique(object_ids[object_types == geom_type]):
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


class CommunicationKGWorldEnv(KinematicKGWorldEnv):
    """Two-agent variant: an immobile transmitter, a mobile blind receiver.

    The transmitting agent is the `agent` body -- it keeps the 4-camera rig and
    the whole perception stack, but it cannot move. Its action is not a move at
    all; the trainer asks it for a symbol and never calls this env's step with
    it. Its observation stays [x/8, y/8, cos yaw, sin yaw], constant for the
    episode since it never moves.

    The receiving agent is the `receiver` body. It takes the Discrete(4)
    egocentric moves the single-agent env gave the transmitter, relative to a
    yaw drawn once at reset (there is no turn action, so its facing is fixed for
    the episode). It is blind: it never sees an observation from this env at
    all, only the symbol stream, so its yaw is a fixed action->displacement
    mapping rather than something it perceives.

    Everything that decides an outcome now keys off the RECEIVER's position --
    food reach, lion catch, and the potential-based shaping term. The
    transmitter sits still, so if these still keyed off it nothing could ever be
    earned. Both agents are credited the same reward (the trainer does that);
    this env just returns it once.

    `receiver_anchor()` gives the transmitter an oracle read of where the
    receiver is, every step. That is a deliberate, stated shortcut: SAM at
    ~1.1 s/frame cannot run per-step, and the receiver is the one thing in the
    world that moves, so a 75-step-stale anchor for it would be uncorrelated
    with the truth. The static lion/cage/food keep coming from real perception.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.receiver_yaw = 0.0
        self.receiver_caught = False

    # ------------------------------------------------------------------ setup

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # drawn after every other RNG consumer, same discipline as the receiver
        # position itself, so a given seed reproduces the one-agent layout
        self.receiver_yaw = float(self.rng.uniform(-np.pi, np.pi))
        self.receiver_caught = False
        self._prev_resource_dist = self._resource_dist(self._get_info())
        return obs, self._get_info()

    # ------------------------------------------------------------------ core

    @property
    def receiver_xy(self):
        return self.model.body("receiver").pos[:2].copy()

    def receiver_anchor(self):
        """Receiver world position in /POS_SCALE units, for AnchorMemory.

        Written straight into the transmitter's anchor slot each step, so
        deltas_from_anchors rotates it into the transmitter's frame exactly as
        it does for a SAM-derived anchor -- the receiver's offset is simply
        refreshed every step instead of every perceive_every steps.
        """
        return self.receiver_xy / POS_SCALE

    def _apply_action(self, action):
        """Move the RECEIVER one unit step, egocentric to its own reset yaw."""
        yaw = self.receiver_yaw
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        move = (fwd, -fwd, left, -left)[int(action)] * self.step_size
        self.model.body("receiver").pos[:2] = np.clip(
            self.receiver_xy + move, -ARENA_HALF, ARENA_HALF
        )
        mujoco.mj_forward(self.model, self.data)

    def _get_info(self):
        """Outcome distances measured from the RECEIVER, not the transmitter."""
        recv_xy = self.receiver_xy
        agent_xy = self.data.qpos[0:2]
        info = {
            "dist_to_lion": float(np.linalg.norm(recv_xy - self.model.body("lion").pos[:2])),
            "dist_to_food": float(np.linalg.norm(recv_xy - self.model.body("food").pos[:2])),
            "dist_to_cage": float(np.linalg.norm(recv_xy - self.model.body("cage").pos[:2])),
            "lion_caged": self.lion_caged,
            "food_near_lion": self.food_near_lion,
            "food_reached": False,
            "lion_caught": False,
            # transmitter -> receiver, the one thing that varies in its input
            "dist_to_receiver": float(np.linalg.norm(agent_xy - recv_xy)),
        }
        if self.resources:
            info["dist_to_water"] = float(
                np.linalg.norm(recv_xy - self.model.body("water").pos[:2])
            )
            info["dist_to_poison"] = float(
                np.linalg.norm(recv_xy - self.model.body("poison").pos[:2])
            )
            info["water_reached"] = False
            info["poison_touched"] = False
        return info
