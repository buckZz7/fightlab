"""G1 mocap-imitation punch env (DeepMimic/AMP-style).

Instead of reward-shaping a punch from scratch (failed: pure RL finds
stand-still local optima), give the policy a reference punch trajectory
retargeted from human mocap (openhe/g1-retargeted-motions, ASAP format),
and reward: imitation of the reference + bag contact + staying upright.

Reference: Hooks_punch.pkl / Horse-stance_punch.pkl — 23-DoF G1 joint
angles at 30fps, punch peak ~frame 74.

23-DoF -> our 29-DoF mapping (only joints present in 23dof are imitated;
wrist pitch/yaw and extra waist joints are left to the policy):
  23dof legs 0-11  -> 29dof legs 0-11 (not imitated: base policy owns legs)
  23dof waist_yaw 12 -> 29dof waist_yaw 12
  23dof L arm 13-17 (sh_p, sh_r, sh_y, el, wr_roll) -> 29dof 15,16,17,18,20
  23dof R arm 18-22 -> 29dof 22,23,24,25,27
"""
import joblib
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from loco_base29 import LocoBase29, HOME
from g1_punch_env import build_model

DT = 0.002
FRAME_SKIP = 10            # 50 Hz control
MOCAP_FPS = 30
# env control runs 50Hz, mocap at 30fps -> advance mocap frame every ~1.67 steps
STEPS_PER_MOCAP_FRAME = (1.0 / MOCAP_FPS) / (DT * FRAME_SKIP)

SKILL_JOINTS = list(range(15, 29))     # arms only (base owns legs + waist)
N_SKILL = 14

# 29dof joint index -> 23dof dof index, for imitated joints
# (waist_yaw + 5 arm joints per side that exist in 23dof)
IMIT_MAP = {
    12: 12,          # waist_yaw -> waist_yaw
    15: 13, 16: 14, 17: 15, 18: 16, 20: 17,   # left arm
    22: 18, 23: 19, 24: 20, 25: 21, 27: 22,   # right arm
}
IMIT_JOINTS_29 = sorted(IMIT_MAP.keys())            # 11 imitated joints
IMIT_JOINTS_23 = [IMIT_MAP[j] for j in IMIT_JOINTS_29]

RESIDUAL_SCALE = np.array(
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2] +
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2])


class ReferenceMotion:
    def __init__(self, pkl_path, loop=False):
        d = joblib.load(pkl_path)
        m = d[list(d.keys())[0]]
        self.dof = m["dof"]                    # (T, 23)
        self.root = m["root_trans_offset"]     # (T, 3)
        self.fps = int(m["fps"])
        self.T = len(self.dof)
        self.loop = loop

    def target_29dof(self, frame):
        """Reference joint targets for imitated 29dof joints at `frame`."""
        f = min(frame, self.T - 1)
        ref = np.zeros(29)
        for j29, j23 in IMIT_MAP.items():
            ref[j29] = self.dof[f, j23]
        return ref


class G1MocapPunchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mocap_path, max_steps=300, randomize=False,
                 imit_weight=2.0, bag_weight=20.0, start_frame=0):
        super().__init__()
        self.render_mode = None
        self.model = build_model(with_bag=True)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize
        self.imit_weight = imit_weight
        self.bag_weight = bag_weight
        self.start_frame = start_frame

        self.ref = ReferenceMotion(mocap_path)
        self.pelvis_id = self.model.body("pelvis").id
        self.lo = self.model.actuator_ctrlrange[:, 0]
        self.hi = self.model.actuator_ctrlrange[:, 1]
        self.bag_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bag_swing")
        self.bag_dof = int(self.model.jnt_dofadr[self.bag_jnt])
        self.bag_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "heavy_bag")
        self.r_fist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_fist_col")
        self.l_fist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_fist_col")

        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()
        self.loco = LocoBase29()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_SKILL,), dtype=np.float64)
        obs0 = self._blank_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs0.shape, dtype=np.float64)
        self.step_count = 0

    def _blank_obs(self):
        # 14 arm pos + 14 arm vel + 3 waist pos + 4 quat + 3 omega
        # + 3 bag pos + 1 bag vel + 3 rel_r + 3 rel_l + 1 mocap_phase + 11 imit_error
        return np.zeros(14+14+3+4+3+3+1+3+3+1+11)

    def _mocap_frame(self):
        return self.start_frame + int(self.step_count / STEPS_PER_MOCAP_FRAME)

    def _get_obs(self):
        q = self.data.qpos[7 + 15:7 + 29]
        qd = self.data.qvel[6 + 15:6 + 29]
        waist_q = self.data.qpos[7 + 12:7 + 15]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        bag_pos = self.data.xpos[self.bag_body]
        bag_vel = np.array([self.data.qvel[self.bag_dof]])
        rel_r = bag_pos - self.data.geom_xpos[self.r_fist]
        rel_l = bag_pos - self.data.geom_xpos[self.l_fist]
        frame = self._mocap_frame()
        phase = np.array([frame / self.ref.T])
        ref = self.ref.target_29dof(frame)
        cur = self.data.qpos[7:36]
        imit_err = (ref[IMIT_JOINTS_29] - cur[IMIT_JOINTS_29])
        return np.concatenate([q, qd, waist_q, quat, omega, bag_pos, bag_vel,
                               rel_r, rel_l, phase, imit_err]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75
        self.data.qpos[7:36] = HOME
        # init imitated joints to the first reference frame
        ref0 = self.ref.target_29dof(self.start_frame)
        for j in IMIT_JOINTS_29:
            self.data.qpos[7 + j] = ref0[j]
        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(0.9, 1.1)
        mujoco.mj_forward(self.model, self.data)
        self.loco.reset()
        self.step_count = 0
        self._residual = np.zeros(N_SKILL)
        self._root_x0 = float(self.data.xpos[self.pelvis_id][0])
        return self._get_obs(), {}

    def _pelvis_z(self):
        return float(self.data.xpos[self.pelvis_id][2])

    def step(self, action):
        raw = np.clip(action, -1, 1) * RESIDUAL_SCALE
        self._residual += 0.25 * (raw - self._residual)
        residual = self._residual
        bag_vel_prev = abs(float(self.data.qvel[self.bag_dof]))

        # reference targets at the current mocap frame + policy residual on arms
        frame = self._mocap_frame()
        ref = self.ref.target_29dof(frame)

        for _ in range(self.frame_skip):
            self.loco.update(self.data.qpos, self.data.qvel)
            target = self.loco.target.copy()
            # imitated joints track the reference; policy adds residual on arms
            for j in IMIT_JOINTS_29:
                target[j] = ref[j]
            target[SKILL_JOINTS] += residual
            tau = self.loco.pd_torque(self.data.qpos, self.data.qvel, target_override=target)
            self.data.ctrl[:] = np.clip(tau, self.lo, self.hi)
            mujoco.mj_step(self.model, self.data, 1)
        self.step_count += 1

        # --- reward ---
        bag_vel = abs(float(self.data.qvel[self.bag_dof]))
        hit = max(0.0, bag_vel - bag_vel_prev)
        r_bag = self.bag_weight * hit + 2.0 * max(0.0, bag_vel - 0.2)

        # imitation: how close are imitated joints to the reference
        cur = self.data.qpos[7:36]
        imit_err = np.square(ref[IMIT_JOINTS_29] - cur[IMIT_JOINTS_29]).mean()
        r_imit = self.imit_weight * np.exp(-4.0 * imit_err)

        # root-motion penalty: don't walk into the punch. Penalize pelvis
        # x-translation from spawn so the policy learns to punch planted.
        root_x = float(self.data.xpos[self.pelvis_id][0])
        r_root = -3.0 * abs(root_x - self._root_x0)

        z = self._pelvis_z()
        quat = self.data.qpos[3:7]
        tilt = 1.0 - float(quat[0]) ** 2
        r_stab = -3.0 * tilt - 2.0 * max(0.0, 0.72 - z)

        r_energy = -0.01 * float(np.sum(np.square(residual)))
        reward = r_bag + r_imit + r_root + r_stab + r_energy

        terminated = z < 0.4
        truncated = self.step_count >= self.max_steps
        info = {"bag_vel": bag_vel, "pelvis_z": z, "tilt": tilt,
                "imit_err": float(imit_err), "frame": frame}
        return self._get_obs(), reward, terminated, truncated, info


def make_g1_mocap_punch_env(mocap_path="mocap/kungfu_retargeted/Hooks_punch.pkl", **kw):
    return G1MocapPunchEnv(mocap_path=mocap_path, **kw)
