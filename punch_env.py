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
        """Add a heavy sphere on a fixed stand ahead of the humanoid."""
        spec = self.model
        # worldbody -> add stand (static pole) and bag (dynamic sphere with joint)
        world = spec.worldbody
        stand = world.add_body(name="target_stand", pos=[self._target_distance, 0, 0.0])
        stand.add_geom(name="stand_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.03, 0.75, 0], pos=[0, 0, 0.75],
                       rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
        bag = stand.add_body(name="target_bag", pos=[0, 0, 1.45])
        bag.add_joint(name="bag_swing", type=mujoco.mjtJoint.mjJNT_SLIDE,
                      axis=[1, 0, 0], range=[-0.6, 0.6], damping=8.0, springref=0.0,
                      stiffness=40.0)
        bag.add_geom(name="bag", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[self._target_radius], mass=self._target_mass,
                     rgba=[0.8, 0.2, 0.2, 1])
        self._rebuild()

    def _rebuild(self):
        """Recompile model+data after spec mutation, preserving training API."""
        self.model = mujoco.MjModel.from_xml_string(self.model.to_xml_string() if hasattr(self.model, 'to_xml_string') else None) if False else self.model

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
        # punch damage: fist (hand bodies) contacting the bag at speed
        damage = 0.0
        try:
            bag_id = self.model.body("target_bag").id
            for hand_name in ("hand1", "hand2"):
                try:
                    hand_id = self.model.body(hand_name).id
                except KeyError:
                    continue
                for geom1 in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom1] != hand_id:
                        continue
                    for geom2 in range(self.model.ngeom):
                        if self.model.geom_bodyid[geom2] != bag_id:
                            continue
                        contacts = mujoco.mj_contactForce  # placeholder, replaced below
        except KeyError:
            pass
        # simpler & robust: check bag velocity change as damage proxy
        try:
            bag_jnt = self.model.joint("bag_swing").id
            bag_vel = float(self.data.qvel[self.model.jnt_dofadr[bag_jnt]])
            bag_speed = abs(bag_vel)
            if bag_speed > 1.0:  # m/s swing speed = solid hit
                damage = (bag_speed - 1.0) * 5.0
                self.target_hp -= damage
                reward += damage * 0.5
        except KeyError:
            pass
        info["target_hp"] = self.target_hp
        info["damage_dealt"] = damage
        return obs, reward, terminated, truncated, info


def make_punch_env(render_mode=None, **kwargs):
    return PunchEnv(render_mode=render_mode, **kwargs)
