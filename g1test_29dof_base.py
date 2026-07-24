"""Verify LocomotionBase on the 29-DoF G1 scene: frozen legs stand,
upper body held by a simple PD at HOME. This is the wiring the fight
layer will use (legs from motion.pt, arms from another controller)."""
import mujoco
import numpy as np
import sys
sys.path.insert(0, "/opt/data/fightlab-repo-new")

from loco_base import LocomotionBase, LEG_HOME
from g1_envs import G1_HOME, G1_SCENE

m = mujoco.MjModel.from_xml_path(G1_SCENE)
d = mujoco.MjData(m)
m.opt.timestep = 0.002

d.qpos[2] = 0.75
d.qpos[7:36] = G1_HOME
mujoco.mj_forward(m, d)

loco = LocomotionBase()
pelvis = m.body("pelvis").id
lo = m.actuator_ctrlrange[:, 0]
hi = m.actuator_ctrlrange[:, 1]

# arm/waist hold gains (joints 12..28), moderate
ARM_KP, ARM_KD = 40.0, 2.0

z_min, z_end = 1.0, 0.0
i = 0
for i in range(15000):  # 30s
    leg_tau = loco.pd_torque(d.qpos, d.qvel)
    loco.update(d.qpos, d.qvel)
    # hold upper body at HOME
    arm_q = d.qpos[7 + 12:7 + 29]
    arm_qd = d.qvel[6 + 12:6 + 29]
    arm_tau = ARM_KP * (G1_HOME[12:29] - arm_q) - ARM_KD * arm_qd
    tau = np.concatenate([leg_tau, arm_tau])
    d.ctrl[:] = np.clip(tau, lo, hi)
    mujoco.mj_step(m, d)
    z = float(d.xpos[pelvis][2])
    z_min = min(z_min, z)
    z_end = z
    if z < 0.4:
        break

print(f"29dof + frozen legs (LocomotionBase) + held arms: "
      f"{i+1} steps ({(i+1)*0.002:.1f}s) z min={z_min:.2f} end={z_end:.2f}")
