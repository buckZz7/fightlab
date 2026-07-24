"""Punch env: humanoid with a strike target.

Milestone 2 of the fight league: stay upright AND hit things.
A padded target (a heavy sphere on a stand) sits at jab range in front of
the humanoid. Reward = standing + upright + damage dealt to the target,
minus damage taken from pushes and control cost.

Domain randomization (the sim-to-real insurance) is baked in at reset:
torso mass, floor friction, and motor strength (gear scale) all jitter
per episode so policies can't overfit to one perfect physics.
"""
import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
import mujoco

from envs import BalanceEnv


class PunchEnv(BalanceEnv):
    """BalanceEnv + a punchable target + domain randomization."""

    def __init__(
        self,
        target_distance=1.0,        # meters in front of spawn
        target_radius=0.12,         # heavy bag head
        target_mass=20.0,           # kg, heavy bag-ish
        dr_mass_jitter=0.10,        # +/-10% torso mass
        dr_friction_jitter=0.15,    # +/-15% floor friction
        dr_gear_jitter=0.10,        # +/-10% actuator gear
        **kwargs,
    ):
        # Stripped obs layout differs from BalanceEnv's (body-count-dependent
        # terms removed), so balance policies can't be transferred into this
        # env directly — but observations are now stable across body additions.
        # qpos/qvel for the humanoid are unaffected by the extra bag DOF.
        kwargs.setdefault("include_cinert_in_observation", False)
        kwargs.setdefault("include_cvel_in_observation", False)
        kwargs.setdefault("include_qfrc_actuator_in_observation", False)
        kwargs.setdefault("include_cfrc_ext_in_observation", False)
        super().__init__(**kwargs)
        self._target_distance = target_distance
        self._target_radius = target_radius
        self._target_mass = target_mass
        self._dr_mass_jitter = dr_mass_jitter
        self._dr_friction_jitter = dr_friction_jitter
        self._dr_gear_jitter = dr_gear_jitter
        self._target_body = None
        self._target_prev_vel = np.zeros(3)
        self.target_hp = 100.0
        self._build_target()

    def _build_target(self):
        """Add a heavy sphere on a fixed stand ahead of the humanoid.

        self.model is a compiled MjModel; edit via MjSpec and recompile.
        The obs/action space is unchanged (humanoid joints only), so
        policies trained on BalanceEnv transfer directly.
        """
        # Load the gymnasium humanoid XML and extend it with the target.
        # BalanceEnv's model was compiled from this same file, so obs/action
        # spaces match and balance policies transfer directly.
        import os
        import gymnasium.envs.mujoco as _gym_mj
        xml_path = os.path.join(os.path.dirname(_gym_mj.__file__),
                                "assets", "humanoid.xml")
        spec = mujoco.MjSpec.from_file(xml_path)
        world = spec.worldbody
        stand = world.add_body(name="target_stand", pos=[self._target_distance, 0, 0.0])
        stand.add_geom(name="stand_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.03, 0.75, 0], pos=[0, 0, 0.75],
                       rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
        bag = stand.add_body(name="target_bag", pos=[0, 0, 1.45])
        bag.add_joint(name="bag_swing", type=mujoco.mjtJoint.mjJNT_SLIDE,
                      axis=[1, 0, 0], range=[-0.6, 0.6], damping=8.0,
                      stiffness=40.0)
        bag.add_geom(name="bag", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[self._target_radius], mass=self._target_mass,
                     rgba=[0.8, 0.2, 0.2, 1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # Recompiled model has an extra DOF (bag slide joint); refresh
        # cached initial state, action space, and observation space so the
        # shapes gymnasium advertises match what _get_obs actually returns.
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._set_action_space()
        # observation_space is just an attribute on MujocoEnv; rebuild it
        # from a real observation vector.
        import gymnasium.spaces as _spaces
        obs_vec = self._get_obs()
        self.observation_space = _spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_vec.shape, dtype=np.float64)

    # --- domain randomization ---
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        # torso mass jitter
        torso_id = self.model.body("torso").id
        base_mass = 8.907  # nominal humanoid torso mass
        self.model.body_mass[torso_id] = base_mass * (1 + rng.uniform(
            -self._dr_mass_jitter, self._dr_mass_jitter))
        # floor friction jitter
        self.model.geom_friction[0, 0] = 1.0 * (1 + rng.uniform(
            -self._dr_friction_jitter, self._dr_friction_jitter))
        # actuator gear jitter (per-actuator)
        gear = self.model.actuator_gear[:, 0].copy()
        jitter = 1 + rng.uniform(-self._dr_gear_jitter, self._dr_gear_jitter,
                                 size=gear.shape)
        self.model.actuator_gear[:, 0] = gear * jitter
        # reset target state
        self.target_hp = 100.0
        self._target_prev_vel = np.zeros(3)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Damage proxy: bag swing speed above 1 m/s means a solid fist hit
        # landed. Scales with impact speed.
        damage = 0.0
        bag_jnt = self.model.joint("bag_swing").id
        bag_vel = float(self.data.qvel[self.model.jnt_dofadr[bag_jnt]])
        bag_speed = abs(bag_vel)
        if bag_speed > 1.0:
            damage = (bag_speed - 1.0) * 5.0
            self.target_hp -= damage
            reward += damage * 0.5
        info["target_hp"] = self.target_hp
        info["damage_dealt"] = damage
        return obs, reward, terminated, truncated, info


def make_punch_env(render_mode=None, **kwargs):
    return PunchEnv(render_mode=render_mode, **kwargs)
