"""G1 self-play boxing environment.

Two G1 humanoids in an arena, each driven by a frozen LocoBase29 balance
policy + learned arm residuals (the fight layer). The fight policy controls
14 arm joints per agent; the base policy keeps the robot upright.

Wraps the two-agent arena as a single-agent gymnasium.Env for SB3 PPO
training. The opponent is a frozen policy (or random).

Architecture:
  - Arena: g1_arena.build_arena() (two G1s, fist collision, walls)
  - Balance: LocoBase29 per agent (frozen ONNX, runs at 50 Hz)
  - Fight policy: PPO over 14-dim arm residuals, per agent
  - Damage: fist-to-torso contact force, HP-based
  - Termination: knockdown (pelvis below threshold) or timeout

Based on G1PunchEnv (single-robot bag env) and SelfPlayEnv (old boxing env).
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from g1_arena import build_arena, SKILL_JOINTS, N_SKILL, N_QPOS, N_QVEL, DT, FRAME_SKIP, RESIDUAL_SCALE
from loco_base29 import LocoBase29, HOME

# Per-agent qpos/qvel slice offsets in the combined model
# Robot 1: qpos[0:36], qvel[0:35]
# Robot 2: qpos[36:72], qvel[35:70]
QPOS_OFFSET = [0, N_QPOS]
QVEL_OFFSET = [0, N_QVEL]
ACT_OFFSET = [0, 29]  # 29 actuators per robot

# Torso geom name for damage detection (the main torso body)
TORSO_BODY = "torso_link"

# Observation: own 14 skill joints pos/vel (28) + own torso orientation/omega (7)
#            + own HP (1) + opp HP (1)
#            + opponent relative position (3) + opponent torso orientation (4)
#            + opponent arm joint positions (14) — to track their guard
# Total: 28 + 7 + 2 + 3 + 4 + 14 = 58
OBS_DIM = 58

# HP
MAX_HP = 100.0
DAMAGE_PER_HIT = 15.0      # per unit contact force
DAMAGE_COOLDOWN = 0.3      # seconds between scoring hits (anti-spam)
DAMAGE_CAP_PER_HIT = 25.0  # max HP lost in one hit


class G1SelfPlayEnv(gym.Env):
    """Single-agent view of the G1 boxing arena for PPO training.

    The challenger (agent 1) is trained; the opponent (agent 2) is frozen.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None, max_steps=2000, randomize=True):
        super().__init__()
        self.model = build_arena()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        # Actuator ctrlrange for clipping
        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Body/geom IDs for each agent
        self._setup_ids()

        # Domain randomization base values
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        # Balance policies (one per robot)
        self.loco = [LocoBase29(), LocoBase29()]

        # Opponent policy (SB3 model or None for random)
        self.opponent = opponent_model

        # Action/obs spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_SKILL,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)

        # State
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}

    def _setup_ids(self):
        """Cache body/geom IDs for both agents."""
        self.pelvis_id = []
        self.torso_id = []
        self.fist_geoms = []  # [agent][left, right] geom ids

        for i, pfx in enumerate(["r1_", "r2_"]):
            self.pelvis_id.append(self.model.body(f"{pfx}pelvis").id)
            self.torso_id.append(self.model.body(f"{pfx}{TORSO_BODY}").id)
            fists = []
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"{pfx}{side}_fist_col")
                fists.append(gid)
            self.fist_geoms.append(fists)

        # Torso subtree body IDs — hardcoded list of body names that are
        # valid punch targets (torso + head + arms). Static list avoids the
        # O(nbody * depth) parent walk that was hanging __init__.
        TORSO_SUBTREE_NAMES = [
            TORSO_BODY,  # torso_link
            "left_shoulder_pitch_link", "left_shoulder_roll_link",
            "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link",
            "right_shoulder_yaw_link", "right_elbow_link",
            "right_wrist_roll_link", "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]
        self.torso_bodies = []
        for i, pfx in enumerate(["r1_", "r2_"]):
            body_ids = set()
            for name in TORSO_SUBTREE_NAMES:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{pfx}{name}")
                if bid >= 0:
                    body_ids.add(bid)
            self.torso_bodies.append(body_ids)

    def _get_obs(self, agent=0):
        """Observation for the given agent."""
        qp_off = QPOS_OFFSET[agent]
        qv_off = QVEL_OFFSET[agent]
        opp = 1 - agent

        # Own skill joints (14 pos + 14 vel)
        q = self.data.qpos[qp_off + 7 + 15: qp_off + 7 + 29]
        qd = self.data.qvel[qv_off + 6 + 15: qv_off + 6 + 29]

        # Own torso orientation (quat 4 + angular vel 3)
        quat = self.data.qpos[qp_off + 3: qp_off + 7]
        omega = self.data.qvel[qv_off + 3: qv_off + 6]

        # HP
        hp_self = np.array([self.hp[agent]])
        hp_opp = np.array([self.hp[opp]])

        # Opponent relative position (3) + orientation (4)
        my_pelvis = self.data.xpos[self.pelvis_id[agent]]
        opp_pelvis = self.data.xpos[self.pelvis_id[opp]]
        rel_pos = opp_pelvis - my_pelvis
        opp_quat = self.data.qpos[QPOS_OFFSET[opp] + 3: QPOS_OFFSET[opp] + 7]

        # Opponent arm joint positions (14) — to read their guard
        opp_arms = self.data.qpos[QPOS_OFFSET[opp] + 7 + 15: QPOS_OFFSET[opp] + 7 + 29]

        return np.concatenate([
            q, qd, quat, omega, hp_self, hp_opp,
            rel_pos, opp_quat, opp_arms
        ]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set both robots standing at HOME pose
        for agent in range(2):
            off = QPOS_OFFSET[agent]
            self.data.qpos[off + 2] = 0.75          # pelvis height
            self.data.qpos[off + 7: off + 36] = HOME
            self.data.qpos[off + 7: off + 36] += self.np_random.uniform(-0.02, 0.02, 29)
            # Face each other: robot 2 placed on opposite side, no rotation
            # (the balance policy can't handle a 180° Z flip; the fight
            # policy learns to approach from any direction).
            if agent == 1:
                pass  # no rotation — both face forward

        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(
                0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(
                0.9, 1.1)

        mujoco.mj_forward(self.model, self.data)
        for loco in self.loco:
            loco.reset()

        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}

        return self._get_obs(0), {}

    def _pelvis_z(self, agent):
        return float(self.data.xpos[self.pelvis_id[agent]][2])

    def _opp_action(self, agent):
        """Get action for the opponent (agent 1)."""
        if self.opponent is None:
            return np.random.uniform(-1, 1, N_SKILL)
        obs = self._get_obs(agent)
        a, _ = self.opponent.predict(obs, deterministic=True)
        return np.clip(a, -1, 1)

    def step(self, action):
        """Step the arena. action = challenger (agent 0) arm residuals."""
        # Get opponent action
        opp_action = self._opp_action(1)
        actions = [np.clip(action, -1, 1), opp_action]

        # Low-pass filter both agents' residuals
        for agent in range(2):
            raw = actions[agent] * RESIDUAL_SCALE
            self._residuals[agent] += 0.25 * (raw - self._residuals[agent])

        # Physics step loop (same as G1PunchEnv)
        for _ in range(self.frame_skip):
            for agent in range(2):
                qp = QPOS_OFFSET[agent]
                qv = QVEL_OFFSET[agent]
                self.loco[agent].update(self.data.qpos[qp:qp+N_QPOS],
                                        self.data.qvel[qv:qv+N_QVEL])
                target = self.loco[agent].target.copy()
                target[SKILL_JOINTS] += self._residuals[agent]

                # Compute PD torque for this agent's 29 actuators
                tau = self.loco[agent].pd_torque(
                    self.data.qpos[qp:qp+N_QPOS],
                    self.data.qvel[qv:qv+N_QVEL],
                    target_override=target)
                act_off = ACT_OFFSET[agent]
                self.data.ctrl[act_off:act_off+29] = np.clip(
                    tau, self.lo[act_off:act_off+29], self.hi[act_off:act_off+29])
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1

        # Damage detection: check fist-to-torso contacts
        self._update_damage()

        # Reward for challenger (agent 0)
        reward = self._compute_reward(0)

        # Termination
        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps

        # Win/loss shaping at episode end
        if terminated or truncated:
            if self.hp[1] <= 0 or (z1 < 0.4 and z0 > 0.4):
                reward += 25.0   # win
            elif self.hp[0] <= 0 or (z0 < 0.4 and z1 > 0.4):
                reward -= 25.0   # loss

        info = {
            "hp_0": self.hp[0], "hp_1": self.hp[1],
            "pelvis_z_0": z0, "pelvis_z_1": z1,
        }
        return self._get_obs(0), reward, terminated, truncated, info

    def _update_damage(self):
        """Detect fist-to-opponent-torso contacts and apply HP damage."""
        for con in range(self.data.ncon):
            contact = self.data.contact[con]
            g1, g2 = contact.geom1, contact.geom2
            # Get contact force via mj_contactForce (6-dim wrench)
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, con, f)
            force = np.linalg.norm(f[:3])

            if force < 5.0:
                continue  # ignore light brush contacts

            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            # Check if one geom is a fist and the other is in the
            # opponent's torso subtree (body ID match)
            for attacker in range(2):
                defender = 1 - attacker
                for fist in self.fist_geoms[attacker]:
                    fist_body = self.model.geom_bodyid[fist]
                    if g1 == fist and b2 in self.torso_bodies[defender]:
                        self._register_hit(attacker, defender, force)
                    elif g2 == fist and b1 in self.torso_bodies[defender]:
                        self._register_hit(attacker, defender, force)

    def _register_hit(self, attacker, defender, force):
        """Record a hit with cooldown and HP damage."""
        t = self.step_count * DT * FRAME_SKIP
        if t - self._last_hit_time[attacker] > DAMAGE_COOLDOWN:
            dmg = min(DAMAGE_CAP_PER_HIT,
                      force * DAMAGE_PER_HIT / 100.0)
            self.hp[defender] = max(0, self.hp[defender] - dmg)
            self._last_hit_time[attacker] = t
            self._contact_states[(attacker, defender)] = {
                'force': force, 'damage': dmg}

    def _compute_reward(self, agent):
        """Reward for the given agent."""
        opp = 1 - agent
        reward = 0.0

        # Damage dealt (positive)
        if (agent, opp) in self._contact_states:
            cs = self._contact_states[(agent, opp)]
            reward += 5.0 * cs['damage']
            # Big hit bonus
            if cs['force'] > 50:
                reward += 3.0

        # Damage taken (negative)
        if (opp, agent) in self._contact_states:
            cs = self._contact_states[(opp, agent)]
            reward -= 2.0 * cs['damage']

        # Stability: small alive bonus, big tilt penalty
        z = self._pelvis_z(agent)
        quat = self.data.qpos[QPOS_OFFSET[agent] + 3: QPOS_OFFSET[agent] + 7]
        tilt = 1.0 - float(quat[0]) ** 2
        reward += 0.05 if z > 0.6 else 0.0
        reward -= 3.0 * tilt
        reward -= 2.0 * max(0.0, 0.72 - z)

        # Approach: reward closing distance to opponent
        my_pos = self.data.xpos[self.pelvis_id[agent]]
        opp_pos = self.data.xpos[self.pelvis_id[opp]]
        dist = np.linalg.norm(opp_pos - my_pos)
        reward += 0.5 * max(0.0, 1.0 - dist)  # closer = better

        # Energy penalty
        reward -= 0.01 * float(np.sum(np.square(self._residuals[agent])))

        # Clear contact states after reward computation
        self._contact_states = {}

        return reward


def make_g1_selfplay_env(opponent_path=None, **kw):
    from stable_baselines3 import PPO
    opp = PPO.load(opponent_path) if opponent_path else None
    return G1SelfPlayEnv(opponent_model=opp, **kw)
