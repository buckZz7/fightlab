"""G1 full-body boxing environment with mocap priors.

No frozen balance base. The policy controls all 29 joints, guided by
mocap imitation reward (DeepMimic-style) to stay standing and move like
a boxer, plus task reward to hit the opponent.

Architecture:
  - Two G1s in a boxing arena (from g1_arena)
  - Each policy outputs 29 joint position targets in [-1, 1]
  - Scaled to per-joint ranges, PD torque to physics
  - Mocap imitation reward keeps motion human-like
  - Anti-shove hit reward drives fighting behavior

Training stages:
  Stage 1 (this file): train a single-agent mocap tracker that can
    stand, walk, and punch from mocap reference alone.
  Stage 2: use tracker as opponent for full-body self-play.
"""
import joblib
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from g1_arena import build_arena, N_QPOS, N_QVEL, DT, FRAME_SKIP
from g1_mocap_punch_env import ReferenceMotion, IMIT_MAP, IMIT_JOINTS_29
from loco_base29 import HOME

# 29-DoF joint ranges (from G1 spec for position control)
# Each joint has a [min, max] range in radians
JOINT_RANGE_MIN = np.array([
    -0.6, -0.4, -0.4, -1.6, -0.4, -0.3,  # left leg
    -0.6, -0.4, -0.4, -1.6, -0.4, -0.3,  # right leg
    -0.4, -0.3, -0.2,                      # waist
    -2.0, -0.4, -2.0, -2.0, -0.4, -0.4, -0.4,  # left arm
    -2.0, -0.4, -2.0, -2.0, -0.4, -0.4, -0.4,  # right arm
])
JOINT_RANGE_MAX = np.array([
    0.6, 0.4, 0.4, 0.0, 0.3, 0.3,  # left leg
    0.6, 0.4, 0.4, 0.0, 0.3, 0.3,  # right leg
    0.4, 0.3, 0.2,                  # waist
    2.0, 0.4, 2.0, 0.0, 0.4, 0.4, 0.4,  # left arm
    2.0, 0.4, 2.0, 0.0, 0.4, 0.4, 0.4,  # right arm
])

# Leg joints: 0-11, Waist: 12-14, Arms: 15-28
ALL_JOINTS = list(range(29))
N_JOINTS = 29
ACT_DIM = N_JOINTS

# Mocap imitation: which joints we track from reference (same as g1_mocap_punch)
# PD gains (same as LocoBase29)
KP = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 28.5, 28.5,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
KD = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 1.8, 1.8,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1])

# Damage constants
MAX_HP = 100.0
DAMAGE_PER_HIT = 15.0
DAMAGE_COOLDOWN = 0.3
DAMAGE_CAP_PER_HIT = 25.0
STEPS_PER_MOCAP_FRAME = (1.0 / 30) / (DT * FRAME_SKIP)

# Per-agent slice offsets
QPOS_OFFSET = [0, N_QPOS]
QVEL_OFFSET = [0, N_QVEL]
ACT_OFFSET = [0, 29]


class G1FullBodyEnv(gym.Env):
    """Single-agent view of G1 boxing arena with full-body control.

    No frozen balance base. The policy controls all 29 joints.
    Mocap imitation reward shapes standing + walking + punching.

    Stage 1: train against mocap reference (bag removed).
    Stage 2: train against a frozen opponent policy (self-play).
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_mocap_path=None, max_steps=500, randomize=False):
        super().__init__()
        self.model = build_arena()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Cache IDs
        self.pelvis_id = [self.model.body("r1_pelvis").id,
                          self.model.body("r2_pelvis").id]
        self.torso_id = [self.model.body("r1_torso_link").id,
                          self.model.body("r2_torso_link").id]
        self.fist_geoms = []
        for pfx in ["r1_", "r2_"]:
            fists = []
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"{pfx}{side}_fist_col")
                fists.append(gid)
            self.fist_geoms.append(fists)
        # Torso subtree bodies (valid punch targets)
        TORSO_SUBTREE = ["torso_link", "head_link",
                         "left_shoulder_pitch_link", "left_shoulder_roll_link",
                         "left_shoulder_yaw_link", "left_elbow_link",
                         "left_wrist_roll_link", "left_wrist_pitch_link",
                         "left_wrist_yaw_link",
                         "right_shoulder_pitch_link", "right_shoulder_roll_link",
                         "right_shoulder_yaw_link", "right_elbow_link",
                         "right_wrist_roll_link", "right_wrist_pitch_link",
                         "right_wrist_yaw_link"]
        self.torso_bodies = []
        for pfx in ["r1_", "r2_"]:
            ids = set()
            for name in TORSO_SUBTREE:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{name}")
                if bid >= 0:
                    ids.add(bid)
            self.torso_bodies.append(ids)

        # Mocap reference (for imitation reward)
        self.ref = None
        if opponent_mocap_path:
            self.ref = ReferenceMotion(opponent_mocap_path)

        # Opponent (SB3 policy or None)
        self.opponent = None

        # Domain randomization
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        # Obs: 29 joint pos + 29 joint vel + 4 quat + 3 omega + HP(2)
        #       + opponent rel pos(3) + opp quat(4) + opp joint pos(29)
        #       + mocap phase(1) + imit_err(11) = 115
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(115,), dtype=np.float64)

        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._mocap_frame = 0
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}

    def _get_obs(self, agent=0):
        qp = QPOS_OFFSET[agent]
        qv = QVEL_OFFSET[agent]
        opp = 1 - agent

        qpos = self.data.qpos[qp + 7: qp + 36]
        qvel = self.data.qvel[qv + 6: qv + 35]
        quat = self.data.qpos[qp + 3: qp + 7]
        omega = self.data.qvel[qv + 3: qv + 6]
        hp = np.array(self.hp)

        my_pos = self.data.xpos[self.pelvis_id[agent]]
        opp_pos = self.data.xpos[self.pelvis_id[opp]]
        rel_pos = opp_pos - my_pos
        opp_quat = self.data.qpos[QPOS_OFFSET[opp] + 3: QPOS_OFFSET[opp] + 7]
        opp_qpos = self.data.qpos[QPOS_OFFSET[opp] + 7: QPOS_OFFSET[opp] + 36]

        # Mocap phase + imitation error
        frame = self._mocap_frame
        phase = np.array([frame / self.ref.T]) if self.ref else np.zeros(1)
        if self.ref:
            ref = self.ref.target_29dof(frame)
            imit_err = (ref[IMIT_JOINTS_29] - qpos[IMIT_JOINTS_29])
        else:
            imit_err = np.zeros(len(IMIT_JOINTS_29))

        return np.concatenate([
            qpos, qvel, quat, omega, hp, rel_pos, opp_quat, opp_qpos,
            phase, imit_err
        ]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Both robots at HOME
        for agent in range(2):
            off = QPOS_OFFSET[agent]
            self.data.qpos[off + 2] = 0.75
            self.data.qpos[off + 7: off + 36] = HOME
            if self.ref:
                ref0 = self.ref.target_29dof(0)
                for j in IMIT_JOINTS_29:
                    self.data.qpos[off + 7 + j] = ref0[j]

        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(0.9, 1.1)

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._mocap_frame = 0
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}
        return self._get_obs(0), {}

    def _pelvis_z(self, agent):
        return float(self.data.xpos[self.pelvis_id[agent]][2])

    def step(self, action):
        arm = np.clip(action[:ACT_DIM], -1, 1)
        opp_action = np.zeros(ACT_DIM)

        # Map action [-1,1] to joint ranges
        target = HOME + arm * (JOINT_RANGE_MAX - JOINT_RANGE_MIN) / 2.0

        # Physics
        for _ in range(self.frame_skip):
            # Agent 0 (challenger)
            tau0 = KP * (target - self.data.qpos[7:36]) - KD * self.data.qvel[6:35]
            self.data.ctrl[:29] = np.clip(tau0, self.lo[:29], self.hi[:29])
            # Agent 1 (opponent) — static at HOME
            tau1 = KP * (HOME - self.data.qpos[36+7:36+36]) - KD * self.data.qvel[35+6:35+35]
            self.data.ctrl[29:58] = np.clip(tau1, self.lo[29:58], self.hi[29:58])
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1
        if self.ref:
            self._mocap_frame = int(self.step_count / STEPS_PER_MOCAP_FRAME)

        # Damage
        self._update_damage()
        reward = self._compute_reward()

        z0, z1 = self._pelvis_z(0), self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            if self.hp[1] <= 0 or (z1 < 0.4 and z0 > 0.4):
                reward += 25.0
            elif self.hp[0] <= 0 or (z0 < 0.4 and z1 > 0.4):
                reward -= 25.0

        info = {"hp_0": self.hp[0], "hp_1": self.hp[1],
                "pelvis_z_0": z0, "pelvis_z_1": z1}
        return self._get_obs(0), reward, terminated, truncated, info

    def _update_damage(self):
        """Check fist-to-torso contacts with anti-shove gating (same as g1_selfplay_env)."""
        for con in range(self.data.ncon):
            contact = self.data.contact[con]
            g1, g2 = contact.geom1, contact.geom2
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, con, f)
            force = np.linalg.norm(f[:3])
            if force < 5.0:
                continue
            b1, b2 = self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]
            for attacker in range(2):
                defender = 1 - attacker
                for fist in self.fist_geoms[attacker]:
                    hit = (g1 == fist and b2 in self.torso_bodies[defender]) or \
                          (g2 == fist and b1 in self.torso_bodies[defender])
                    if hit:
                        fist_body = self.model.geom_bodyid[fist]
                        fist_vel = self.data.cvel[fist_body][3:6].copy()
                        torso_vel = self.data.cvel[self.torso_id[attacker]][3:6].copy()
                        rel_vel = fist_vel - torso_vel
                        attack_dir = self.data.xpos[self.torso_id[defender]] - self.data.xpos[self.torso_id[attacker]]
                        an = np.linalg.norm(attack_dir)
                        if an > 1e-6:
                            attack_dir /= an
                        punch_speed = float(np.dot(rel_vel, attack_dir))
                        if punch_speed > 0.5:
                            t = self.step_count * DT * FRAME_SKIP
                            if t - self._last_hit_time[attacker] > DAMAGE_COOLDOWN:
                                dmg = min(DAMAGE_CAP_PER_HIT, force * DAMAGE_PER_HIT / 100.0)
                                self.hp[defender] = max(0, self.hp[defender] - dmg)
                                self._last_hit_time[attacker] = t
                                self._contact_states[(attacker, defender)] = dmg
                        else:
                            self._contact_states[(attacker, defender)] = 0.0

    def _compute_reward(self):
        reward = 0.0
        z = self._pelvis_z(0)

        # Stability
        quat = self.data.qpos[3:7]
        tilt = 1.0 - float(quat[0])**2
        reward += 0.05 if z > 0.6 else 0.0
        reward -= 3.0 * tilt
        reward -= 2.0 * max(0.0, 0.72 - z)

        # Mocap imitation
        if self.ref:
            frame = self._mocap_frame
            ref = self.ref.target_29dof(frame)
            cur = self.data.qpos[7:36]
            imit_err = np.square(ref[IMIT_JOINTS_29] - cur[IMIT_JOINTS_29]).mean()
            reward += 2.0 * np.exp(-4.0 * imit_err)

        # Hit opponent
        if (0, 1) in self._contact_states:
            dmg = self._contact_states[(0, 1)]
            reward += 5.0 * dmg
        if (1, 0) in self._contact_states:
            dmg = self._contact_states[(1, 0)]
            reward -= 3.0 * dmg

        # Energy
        reward -= 0.005 * float(np.sum(np.square(
            self.data.ctrl[:29] / (self.hi[:29] - self.lo[:29]))))

        self._contact_states = {}
        return reward
