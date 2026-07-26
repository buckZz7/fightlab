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
from loco_base29 import StandPD, KP, KD, HOME

DT = 0.002   # MUST match g1_arena.DT (RK4 stable timestep). Overriding to
             # 0.01 made the humanoid sag + fall (RK4 too coarse at 10ms).
FRAME_SKIP = 1   # control every physics step (500Hz, matches the stable
                 # preflight PD loop). 300 steps = 0.6s; use more for longer.
N_ACT = 29
N_OBS = 4 + 3 + 29 + 29
SCALE_BAL = 0.40          # rad residual scale (policy needs real authority
                             # to actively balance; 0.10 was too weak)
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

        # Base pose = HOME (the proven-stable stand pose StandPD holds
        # for 300 steps -- see preflight.py). The balance policy learns
        # residuals around HOME. Using native here caused the policy to
        # train on an unstable base and never converge (stood 18/1500).
        self.base = HOME.copy()

        self.loco = StandPD()           # r2 frozen stander
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(N_ACT,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float64)
        self.step_count = 0

    def _place(self):
        mujoco.mj_resetData(self.model, self.data)
        for ai, x in enumerate(NATIVE_ROOT_X):
            off = ai * 36            # each robot = 7 (root) + 29 (joints) = 36 qpos
            self.data.qpos[off: off + 3] = [x, 0, STAND_Z]
            # robot joints at qpos[off+7 : off+36]
            self.data.qpos[off + 7: off + 36] = self.base[:29]  # HOME joints (29)
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
        jrel = qp[7:36] - self.base     # base is 29-dim HOME joints
        jvel = qv[6:35]
        return np.concatenate([quat, angvel, jrel, jvel]).astype(np.float64)

    def reset(self, seed=None, options=None):
        self._place()
        self.step_count = 0
        self.loco.reset()
        return self._get_obs(), {}

    def step(self, action):
        act = np.clip(action, -1, 1) * SCALE_BAL
        target = self.base + act     # r1 PD target (around HOME, 29-dim)

        for _ in range(self.frame_skip):
            # r1 (offset 0): PD to learned target
            tau1 = KP * (target - self.data.qpos[7:36]) - KD * self.data.qvel[6:35]
            self.data.ctrl[:29] = np.clip(tau1, self.lo[:29], self.hi[:29])
            # r2 (offset 36): StandPD frozen
            self.loco.update(self.data.qpos, self.data.qvel, off=36)
            t2 = self.loco.target
            tau2 = KP * (t2 - self.data.qpos[43:72]) - KD * self.data.qvel[41:70]
            self.data.ctrl[29:58] = np.clip(tau2, self.lo[29:], self.hi[29:])
            try:
                mujoco.mj_step(self.model, self.data, 1)
            except mujoco.FatalError:
                # Degenerate contact (rank-deficient Hessian). Do NOT force
                # a fall -- that poisons training + the eval. Just skip this
                # micro-step; the tol=1e-4 fix in build_arena should make
                # this rare. Breaking here keeps the episode alive.
                break

        self.step_count += 1
        z = float(self.data.qpos[2])
        quat = self.data.qpos[3:7]
        up = _quat_up(quat)

        reward = 0.0
        reward += np.exp(-((z - STAND_Z) ** 2) / 0.02)   # height (tighter)
        reward += 0.5 * max(0.0, up)                          # upright
        # discourage drift/sag: penalize torso linear+angular velocity
        lin = np.linalg.norm(self.data.qvel[3:6])
        reward -= 0.02 * lin
        reward -= 0.005 * float(np.dot(act, act))                   # action cost
        reward -= 0.02 * max(0.0, 0.4 - z)                     # near-fall penalty
        # FOOT CONTACT (plant feet on floor -- without it PD sinks)
        fc = self._foot_contact_count()
        reward += 0.1 * min(fc, 2)   # up to +0.2 for both feet planted
        # ACTION SMOOTHNESS (HoST: L2 on delta-action prevents oscillation/sag)
        if hasattr(self, "_prev_act"):
            reward -= 0.01 * float(np.sum((act - self._prev_act) ** 2))
        self._prev_act = act.copy()

        terminated = z < FALL_Z
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 10.0
        return self._get_obs(), float(reward), terminated, truncated, {"pelvis_z": z, "up": up}

    def _foot_contact_count(self):
        """Count r1 foot geoms touching any non-r1 body (i.e. the floor)."""
        fb = {"r1_left_ankle_roll_link", "r1_right_ankle_roll_link",
              "r1_left_ankle_pitch_link", "r1_right_ankle_pitch_link"}
        n = 0
        for c in range(self.data.ncon):
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[self.data.contact[c].geom1])
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[self.data.contact[c].geom2])
            if (b1 in fb and b2 not in fb) or (b2 in fb and b1 not in fb):
                n += 1
        return n

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
