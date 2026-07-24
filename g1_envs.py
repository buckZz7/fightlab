"""G1-based envs for FightLab: real Unitree G1 morphology (29 DoF).

This is the embodiment that matters for sim2real — replaces the generic
gymnasium Humanoid. Three curriculum levels:

  G1BalanceEnv  — stand upright while shoved at random (torso impulses)
  G1PunchEnv    — balance + strike a heavy bag at jab range
  G1BoxingEnv   — two G1s, HP damage from fist contact force, self-play

Actions: normalized position targets in [-1, 1], mapped to joint ctrl ranges
(G1 uses torque-limited position actuators; unitree deploy stacks do the same).

Domain randomization per episode: torso mass, floor friction, actuator gear.
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

G1_SCENE = "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

DT = 0.002
FRAME_SKIP = 10          # 50 Hz control

# Unitree G1 deployed gains (unitree_rl_mjlab deploy/robots/g1/config/policy/
# velocity/v0/params/deploy.yaml) — the exact PD config that stands and walks
# on the real robot. Joint order matches the 29dof XML actuator order.
G1_KP = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
                  40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
                  40.2, 28.5, 28.5,
                  14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
                  14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
G1_KD = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
                  2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
                  2.6, 1.8, 1.8,
                  0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
                  0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1])
G1_HOME = np.array([-0.1, 0, 0, 0.3, -0.2, 0,
                    -0.1, 0, 0, 0.3, -0.2, 0,
                    0, 0, 0,
                    0.35, 0.18, 0, 0.87, 0, 0, 0,
                    0.35, -0.18, 0, 0.87, 0, 0, 0])
G1_ACTION_SCALE = np.array([0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                            0.55, 0.44, 0.44,
                            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
                            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07])


class G1Base(gym.Env):
    """Shared G1 sim plumbing: load scene, step physics, base obs."""

    metadata = {"render_modes": []}

    def __init__(self, xml_path=G1_SCENE, frame_skip=FRAME_SKIP,
                 healthy_z_range=(0.45, 1.2), max_steps=2000):
        super().__init__()
        # Stock XML (raw torque motors). Control = PD computed in code at the
        # physics rate, exactly like unitree's deploy stack:
        #   tau = kp * (q_target - q) - kd * qd, applied as motor torques.
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = frame_skip
        self.healthy_z_range = healthy_z_range
        self.max_steps = max_steps
        self.step_count = 0

        self.pelvis_id = self.model.body("pelvis").id
        self.nu = self.model.nu
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Deployed G1 config (unitree_rl_mjlab deploy.yaml): PD gains, HOME
        # pose, per-joint action scale. Actuator order in the 29dof XML matches
        # the yaml joint order (legs, waist, arms).
        assert self.nu == 29, f"expected 29 actuators, got {self.nu}"
        self.kp = G1_KP.copy()
        self.kd = G1_KD.copy()
        self.action_scale = G1_ACTION_SCALE.copy()
        self.home = G1_HOME.copy()
        self.q_target = self.home.copy()
        self.default_qpos = self.data.qpos.ravel().copy()
        self.default_qpos[2] = 0.8
        self.default_qpos[7:7 + self.nu] = self.home

        # domain-rand bases
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()
        self._base_gear = self.model.actuator_gear[:, 0].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_dim(),), dtype=np.float64)

    def _obs_dim(self):
        # root vel(3) + root angvel(3) + gravity orient(3) + joint pos(nq-7) + joint vel(nv-6)
        nj = self.model.nq - 7
        return 3 + 3 + 3 + nj + self.model.nv - 6

    def _get_obs(self):
        nj = self.model.nq - 7
        pelvis_quat = self.data.qpos[3:7]
        # gravity direction in body frame (orientation proxy)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, pelvis_quat)
        grav = rot.reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.concatenate([
            self.data.qvel[0:3],                    # root lin vel
            self.data.qvel[3:6],                    # root ang vel
            grav,                                   # orientation
            self.data.qpos[7:7 + nj],               # joint pos
            self.data.qvel[6:6 + self.model.nv - 6],  # joint vel
        ]).astype(np.float64)

    def _apply(self, action):
        # Policy output [-1,1] -> joint position target around HOME, using
        # unitree's per-joint action scale. PD torque applied per physics
        # substep in step().
        a = np.clip(action, -1, 1)
        self.q_target = self.home + a * self.action_scale

    def _domain_randomize(self, rng):
        self.model.body_mass[:] = self._base_mass * rng.uniform(0.9, 1.1, self._base_mass.shape)
        self.model.geom_friction[:] = self._base_friction * rng.uniform(0.85, 1.15)
        self.model.actuator_gear[:, 0] = self._base_gear * rng.uniform(0.9, 1.1, self._base_gear.shape)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos
        # small noise on joint angles
        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, self.model.nq - 7)
        self.step_count = 0
        self.q_target = self.home.copy()
        self._domain_randomize(self.np_random)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _pelvis_z(self):
        return float(self.data.xpos[self.pelvis_id][2])

    def _terminated(self):
        z = self._pelvis_z()
        return z < self.healthy_z_range[0] or z > self.healthy_z_range[1]

    def step(self, action):
        self._apply(action)
        # PD control at the physics rate (unitree deploy style):
        # tau = kp*(q_target - q) - kd*qd, through the XML's torque motors.
        q = self.data.qpos[7:7 + self.nu]
        qd = self.data.qvel[6:6 + self.nu]
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = np.clip(
                self.kp * (self.q_target - q) - self.kd * qd,
                self.ctrl_lo, self.ctrl_hi)
            mujoco.mj_step(self.model, self.data, 1)
            q = self.data.qpos[7:7 + self.nu]
            qd = self.data.qvel[6:6 + self.nu]
        self.step_count += 1
        obs = self._get_obs()
        reward, info = self._reward_info()
        terminated = self._terminated()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def _reward_info(self):
        return 0.0, {}


class G1BalanceEnv(G1Base):
    """Stand upright while shoved: random horizontal torso impulses."""

    def __init__(self, push_interval=250, push_jitter=150,
                 push_force_range=(100.0, 400.0), push_duration=10,
                 **kw):
        super().__init__(**kw)
        self._push_interval = push_interval
        self._push_jitter = push_jitter
        self._push_force_range = push_force_range
        self._push_duration = push_duration
        self._steps_to_next = 0
        self._push_steps_left = 0
        self._push_vec = np.zeros(3)
        self.pushes_survived = 0
        self.pushes_total = 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._steps_to_next = self._push_interval
        self._push_steps_left = 0
        self.pushes_survived = 0
        self.pushes_total = 0
        return obs, info

    def step(self, action):
        self._steps_to_next -= 1
        if self._steps_to_next <= 0 and self._push_steps_left == 0:
            mag = self.np_random.uniform(*self._push_force_range)
            ang = self.np_random.uniform(0, 2 * np.pi)
            self._push_vec = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self._push_steps_left = self._push_duration
            self.pushes_total += 1
            self._steps_to_next = self._push_interval + int(
                self.np_random.integers(-self._push_jitter, self._push_jitter))

        if self._push_steps_left > 0:
            self.data.xfrc_applied[self.pelvis_id, :3] = self._push_vec
            self._push_steps_left -= 1
            if self._push_steps_left == 0:
                self._push_vec[:] = 0
                self.pushes_survived += 1
        else:
            self.data.xfrc_applied[self.pelvis_id, :] = 0

        obs, reward, terminated, truncated, info = super().step(action)
        if terminated and self._push_steps_left > 0:
            self.pushes_survived -= 1
        info["pushes_survived"] = self.pushes_survived
        info["pushes_total"] = self.pushes_total
        info["pelvis_z"] = self._pelvis_z()
        return obs, reward, terminated, truncated, info

    def _reward_info(self):
        z = self._pelvis_z()
        target_z = 0.79  # nominal G1 standing pelvis height
        upright = max(0.0, 1.0 - abs(z - target_z))
        alive = 1.0 if not self._terminated() else 0.0
        ctrl_cost = 0.001 * float(np.sum(np.square(self.data.ctrl)))
        return 1.5 * alive + upright - ctrl_cost, {}


def make_g1_balance_env(**kw):
    return G1BalanceEnv(**kw)
