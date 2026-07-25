"""G1 full-body mocap tracker (no frozen base, no bag).

Trains a 29-DoF policy to track a mocap reference. This is the warmup
stage for full-body boxing — the policy learns to stand, walk, and punch
from mocap alone.

Once trained, this policy is used as the opponent and starting policy
for self-play boxing (g1_fullbody_env).
"""
import joblib
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from g1_punch_env import build_model as build_single_model
from g1_mocap_punch_env import ReferenceMotion, IMIT_MAP, IMIT_JOINTS_29
from loco_base29 import HOME

DT = 0.002
FRAME_SKIP = 10
STEPS_PER_MOCAP_FRAME = (1.0 / 30) / (DT * FRAME_SKIP)

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

N_JOINTS = 29
ACT_DIM = N_JOINTS


class G1MocapTracker(gym.Env):
    """Full-body mocap tracking for G1 (no frozen base, no bag).

    The policy outputs 29 joint position targets. PD torque converts to
    physics. Reward is imitation of the mocap reference + stability.
    """

    metadata = {"render_modes": []}

    def __init__(self, mocap_path, max_steps=500, randomize=False):
        super().__init__()
        self.model = build_single_model(with_bag=False)  # no bag
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.ref = ReferenceMotion(mocap_path)
        self.pelvis_id = self.model.body("pelvis").id

        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        # Obs: 29 pos + 29 vel + 4 quat + 3 omega + 1 mocap_phase + 11 imit_err = 77
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(77,), dtype=np.float64)

        self.step_count = 0

    def _get_obs(self):
        qpos = self.data.qpos[7:36]
        qvel = self.data.qvel[6:35]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        frame = int(self.step_count / STEPS_PER_MOCAP_FRAME)
        phase = np.array([frame / self.ref.T])
        ref = self.ref.target_29dof(frame)
        imit_err = ref[IMIT_JOINTS_29] - qpos[IMIT_JOINTS_29]
        return np.concatenate([qpos, qvel, quat, omega, phase, imit_err]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75
        self.data.qpos[7:36] = HOME
        ref0 = self.ref.target_29dof(0)
        for j in IMIT_JOINTS_29:
            self.data.qpos[7 + j] = ref0[j]
        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(0.9, 1.1)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def _pelvis_z(self):
        return float(self.data.xpos[self.pelvis_id][2])

    def step(self, action):
        arm = np.clip(action, -1, 1)
        frame = int(self.step_count / STEPS_PER_MOCAP_FRAME)
        ref = self.ref.target_29dof(frame)

        for _ in range(self.frame_skip):
            # Target = reference + policy residual
            target = ref.copy()
            target[IMIT_JOINTS_29] += arm[IMIT_JOINTS_29] * 0.3  # small residual
            # Non-imitated joints: policy has full control
            for j in range(29):
                if j not in IMIT_JOINTS_29:
                    target[j] = HOME[j] + arm[j] * 0.5

            tau = KP * (target - self.data.qpos[7:36]) - KD * self.data.qvel[6:35]
            self.data.ctrl[:] = np.clip(tau, self.lo, self.hi)
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1

        # Reward
        z = self._pelvis_z()
        cur = self.data.qpos[7:36]
        imit_err = np.square(ref[IMIT_JOINTS_29] - cur[IMIT_JOINTS_29]).mean()

        r_imit = 2.0 * np.exp(-4.0 * imit_err)
        r_stab = 0.05 if z > 0.6 else 0.0
        tilt = 1.0 - float(self.data.qpos[4])**2
        r_tilt = -3.0 * tilt
        r_alive = -2.0 * max(0.0, 0.72 - z)
        r_energy = -0.005 * float(np.sum(np.square(arm)))

        reward = r_imit + r_stab + r_tilt + r_alive + r_energy
        terminated = z < 0.4
        truncated = self.step_count >= self.max_steps
        info = {"pelvis_z": z, "imit_err": float(imit_err), "frame": frame}
        return self._get_obs(), reward, terminated, truncated, info
