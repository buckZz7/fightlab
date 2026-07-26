"""Stand-still balance training env (Track B prerequisite).

A G1 in MuJoCo needs ACTIVE balance control to stand >~6s.
This env trains a from-scratch SB3 policy that holds the native
standing pose via PD residuals. Reward = stay at stand height +
stay upright + small-action penalty. r2 is a frozen StandPD stander
(reference opponent, not fighting yet).

Architecture:
  - action (29): PD target residual on r1's 29 joints
    (target = native_pose[7:36] + action * SCALE_BAL)
  - obs: r1 root quat(4) + root angvel(3) + joint_pos rel(29)
        + joint_vel(29) = 65-dim
  - reward: height Gaussian + upright + action cost - fall
Once this stands 30s, we layer punches (arm residuals +
damage reward) on top = full Track B fighter.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym

from g1_arena import build_arena
from loco_base29 import StandPD, KP, KD

DT = 0.01
FRAME_SKIP = 4
N_ACT = 29
N_OBS = 4 + 3 + 29 + 29
SCALE_BAL = 0.10          # rad residual scale
STAND_Z = 0.793
FALL_Z = 0.40
NATIVE_ROOT_X = [-0.6, 0.3]


def _quat_up(q):
    """Uprightness proxy: world-up projected onto body z-axis."""
    w, x, y, z = q
    # body z-axis in world = 3rd column of rot matrix
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    return float(zz)            # 1.0 = perfectly upright


class G1BalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=1500, randomize=True):
        super().__init__()
        self.model = build_arena(ring="ropes", half=2.4)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # native pose (after reset, before placing) = the XML's stand pose
        mujoco.mj_resetData(self.model, self.data)
        self.native = [self.data.qpos[i * 36: i * 36 + 36].copy()
                       for i in range(2)]

        self.loco = StandPD()           # r2 frozen stander
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(N_ACT,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float64)
        self.step_count = 0

    def _place(self):
        mujoco.mj_resetData(self.model, self.data)
        for ai, x in enumerate(NATIVE_ROOT_X):
            off = ai * 7
            self.data.qpos[off: off + 3] = [x, 0, STAND_Z]
            self.data.qpos[off + 3: off + 32] = self.native[ai][7:36]  # native joints (29)
        if self.randomize:
            self._randomize()
        mujoco.mj_forward(self.model, self.data)

    def _randomize(self):
        bm = self.model.body_mass.copy()
        self.model.body_mass[:] = bm * np.random.uniform(0.9, 1.1, self.model.nbody)
        fr = self.model.geom_friction[:, 0].copy()
        self.model.geom_friction[:, 0] = fr * np.random.uniform(0.85, 1.15, self.model.ngeom)

    def _get_obs(self):
        off = 0                       # r1
        qp = self.data.qpos[off: off + 36]
        qv = self.data.qvel[off: off + 35]
        quat = qp[3:7]
        angvel = qv[3:6]
        jrel = qp[7:36] - self.native[0][7:36]
        jvel = qv[6:35]
        return np.concatenate([quat, angvel, jrel, jvel]).astype(np.float64)

    def reset(self, seed=None, options=None):
        self._place()
        self.step_count = 0
        self.loco.reset()
        return self._get_obs(), {}

    def step(self, action):
        act = np.clip(action, -1, 1) * SCALE_BAL
        target = self.native[0][7:36] + act     # r1 PD target

        for _ in range(self.frame_skip):
            # r1: PD to learned target
            tau1 = KP * (target - self.data.qpos[7:36]) - KD * self.data.qvel[6:35]
            self.data.ctrl[:29] = np.clip(tau1, self.lo[:29], self.hi[:29])
            # r2: StandPD frozen
            self.loco.update(self.data.qpos, self.data.qvel, off=7)
            t2 = self.loco.target
            tau2 = KP * (t2 - self.data.qpos[14:43]) - KD * self.data.qvel[13:42]
            self.data.ctrl[29:58] = np.clip(tau2, self.lo[29:], self.hi[29:])
            try:
                mujoco.mj_step(self.model, self.data, 1)
            except mujoco.FatalError:
                # Degenerate contact (rank-deficient Hessian). Treat as
                # a fall -> terminate the episode rather than kill training.
                self.data.qpos[2] = 0.0
                self.data.qpos[9] = 0.0

        self.step_count += 1
        z = float(self.data.qpos[2])
        quat = self.data.qpos[3:7]
        up = _quat_up(quat)

        reward = 0.0
        reward += np.exp(-((z - STAND_Z) ** 2) / 0.05)   # height
        reward += 0.5 * max(0.0, up)                          # upright
        reward -= 0.005 * float(np.dot(act, act))                   # action cost
        reward -= 0.02 * max(0.0, 0.4 - z)                     # near-fall penalty

        terminated = z < FALL_Z
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 10.0
        return self._get_obs(), float(reward), terminated, truncated, {"pelvis_z": z, "up": up}

    def render(self, height=480, width=640):
        rend = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]
        rend.update_scene(self.data, camera=cam)
        return rend.render()


if __name__ == "__main__":
    import time
    e = G1BalanceEnv(max_steps=300, randomize=False)
    o, _ = e.reset()
    print("obs", o.shape, "act", e.action_space.shape)
    t0 = time.time()
    for i in range(300):
        o, r, term, trunc, info = e.step(e.action_space.sample())
        if term or trunc:
            print(f"ended step {i+1} z={info['pelvis_z']:.3f}")
            break
    print(f"ran; z={float(e.data.qpos[2]):.3f} in {time.time()-t0:.1f}s")
