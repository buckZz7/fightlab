"""G1 punch environment: frozen whole-body balance base + learned arm/waist
striking against a heavy bag.

Composition of validated pieces:
  - LocoBase29 (frozen mjlab ONNX policy) keeps the G1 upright
  - Fist collision spheres (stock G1 wrists have contype=0)
  - Heavy bag at measured jab range (0.30m, z=1.0)
  - Reward = bag swing velocity (punch power proxy) - stability penalty

Action space (17): waist yaw/roll/pitch + both arms (14 joints), as position-
target residuals added on top of the base policy's targets, scaled per the
arm-sweep tolerance envelope (base shrugged off 30% range at 2Hz).

Obs (own 17 joints pos/vel + torso orientation/omega + bag state) = 46.
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from loco_base29 import LocoBase29, HOME

XML = "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
DT = 0.002
FRAME_SKIP = 10            # 50 Hz

WAIST = slice(12, 15)
L_ARM = slice(15, 22)
R_ARM = slice(22, 29)
# Skill = arms only (14). The base policy KEEPS the waist: it uses waist
# rotation for balance, and overriding it breaks the balance loop (found
# empirically — waist residuals fall in ~20 steps, arm sweeps pass).
SKILL_JOINTS = list(range(15, 29))
N_SKILL = 14

# per-joint residual scale (validated: base shrugged off 30% range at 2Hz
# on arms while retaining waist control)
RESIDUAL_SCALE = np.array(
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2] +   # left arm
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2])    # right arm


def build_model(with_bag=True):
    spec = mujoco.MjSpec.from_file(XML)
    # fist collision spheres (stock wrists are contype=0)
    for side in ("left", "right"):
        w = next(b for b in spec.bodies if b.name == f"{side}_wrist_yaw_link")
        w.add_geom(name=f"{side}_fist_col", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.06], pos=[0.05, 0, 0], mass=0.3,
                   rgba=[1, 0, 0, 0.5], contype=1, conaffinity=1)
    if with_bag:
        world = spec.worldbody
        stand = world.add_body(name="bag_stand", pos=[0.30, -0.12, 0.0])
        stand.add_geom(name="stand_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.03, 0.5, 0], pos=[0, 0, 0.5],
                       rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
        bag = stand.add_body(name="heavy_bag", pos=[0, 0, 1.0])
        bag.add_joint(name="bag_swing", type=mujoco.mjtJoint.mjJNT_SLIDE,
                      axis=[1, 0, 0], range=[-0.6, 0.6], damping=8.0,
                      stiffness=40.0)
        bag.add_geom(name="bag", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.12], mass=20.0, rgba=[0.8, 0.2, 0.2, 1])
    return spec.compile()


class G1PunchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=1000, randomize=True):
        super().__init__()
        self.render_mode = None
        self.model = build_model(with_bag=True)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        self.pelvis_id = self.model.body("pelvis").id
        self.lo = self.model.actuator_ctrlrange[:, 0]
        self.hi = self.model.actuator_ctrlrange[:, 1]
        self.bag_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "bag_swing")
        self.bag_dof = int(self.model.jnt_dofadr[self.bag_jnt])
        self.bag_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "heavy_bag")
        self.r_fist = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_fist_col")
        self.l_fist = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_fist_col")

        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        self.loco = LocoBase29()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_SKILL,), dtype=np.float64)
        obs0 = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs0.shape, dtype=np.float64)
        self.step_count = 0

    def _get_obs(self):
        q = self.data.qpos[7 + 15:7 + 29]           # 14 skill joints pos
        qd = self.data.qvel[6 + 15:6 + 29]          # 14 vel
        waist_q = self.data.qpos[7 + 12:7 + 15]     # 3 waist (base-controlled, observe)
        torso_quat = self.data.qpos[3:7]            # 4
        omega = self.data.qvel[3:6]                 # 3
        bag_pos = self.data.xpos[self.bag_body]     # 3
        bag_vel = np.array([self.data.qvel[self.bag_dof]])  # 1
        fist_r = self.data.geom_xpos[self.r_fist]   # 3
        fist_l = self.data.geom_xpos[self.l_fist]   # 3
        rel_r = bag_pos - fist_r                    # 3
        rel_l = bag_pos - fist_l                    # 3
        return np.concatenate([q, qd, waist_q, torso_quat, omega,
                               bag_pos, bag_vel, rel_r, rel_l]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75
        self.data.qpos[7:36] = HOME
        self.data.qpos[7:36] += self.np_random.uniform(-0.02, 0.02, 29)
        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(
                0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(
                0.9, 1.1)
        mujoco.mj_forward(self.model, self.data)
        self.loco.reset()
        self.step_count = 0
        self._residual = np.zeros(N_SKILL)
        return self._get_obs(), {}

    def _pelvis_z(self):
        return float(self.data.xpos[self.pelvis_id][2])

    def step(self, action):
        # low-pass filter the action: position-control PD amplifies 50 Hz
        # jitter into instability; smooth targets are how the sweep test passed.
        raw = np.clip(action, -1, 1) * RESIDUAL_SCALE
        self._residual += 0.25 * (raw - self._residual)
        residual = self._residual
        bag_vel_prev = abs(float(self.data.qvel[self.bag_dof]))

        self.loco.update(self.data.qpos, self.data.qvel)
        target = self.loco.target.copy()
        target[SKILL_JOINTS] += residual
        tau = self.loco.pd_torque(self.data.qpos, self.data.qvel,
                                  target_override=target)
        self.data.ctrl[:] = np.clip(tau, self.lo, self.hi)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data, 1)
        self.step_count += 1

        # --- reward ---
        bag_vel = abs(float(self.data.qvel[self.bag_dof]))
        # punch power: reward bag acceleration spikes (hits), not steady swing
        hit = max(0.0, bag_vel - bag_vel_prev)
        r_power = 10.0 * hit + 0.5 * max(0.0, bag_vel - 0.3)
        # stability shaping: stay near HOME pose on legs, upright torso.
        # (falling is heavily penalized by termination; shape against drift)
        z = self._pelvis_z()
        quat = self.data.qpos[3:7]
        tilt = 1.0 - float(quat[0]) ** 2     # ~0 when upright
        r_stab = -2.0 * tilt - 1.0 * max(0.0, 0.72 - z)
        # energy penalty on residuals
        r_energy = -0.01 * float(np.sum(np.square(residual)))
        reward = r_power + r_stab + r_energy

        terminated = z < 0.4
        truncated = self.step_count >= self.max_steps
        info = {
            "bag_vel": bag_vel,
            "pelvis_z": z,
            "tilt": tilt,
            "hit": hit,
        }
        return self._get_obs(), reward, terminated, truncated, info


def make_g1_punch_env(**kw):
    return G1PunchEnv(**kw)
