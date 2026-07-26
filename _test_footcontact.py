import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np, sys, mujoco
sys.path.insert(0, ".")
from g1_arena import build_arena
from loco_base29 import StandPD, KP, KD, HOME

m = build_arena(ring="ropes", half=2.4)
d = mujoco.MjData(m)
lo = m.actuator_ctrlrange[:, 0].copy()
hi = m.actuator_ctrlrange[:, 1].copy()
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [-0.6, 0, 0.793]
d.qpos[7:36] = HOME
d.qpos[36:39] = [0.3, 0, 0.793]
d.qpos[43:72] = HOME
mujoco.mj_forward(m, d)

# count foot-floor contacts at first step
mujoco.mj_step(m, d, 1)
foot_bodies = {"r1_left_ankle_roll_link", "r1_right_ankle_roll_link",
               "r2_left_ankle_roll_link", "r2_right_ankle_roll_link"}
nfoot = 0
for c in range(d.ncon):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[c].geom1])
    bn2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[c].geom2])
    if bn in foot_bodies or bn2 in foot_bodies:
        nfoot += 1
print("foot-floor contacts at step1:", nfoot)

mujoco.mj_resetData(m, d)
d.qpos[0:3] = [-0.6, 0, 0.793]
d.qpos[7:36] = HOME
d.qpos[36:39] = [0.3, 0, 0.793]
d.qpos[43:72] = HOME
mujoco.mj_forward(m, d)

minz = 9
fell = None
for i in range(1500):  # 3s @ DT=0.002
    tau = KP * (HOME - d.qpos[7:36]) - KD * d.qvel[6:35]
    d.ctrl[:29] = np.clip(tau, lo[:29], hi[:29])
    tau2 = KP * (HOME - d.qpos[43:72]) - KD * d.qvel[41:70]
    d.ctrl[29:58] = np.clip(tau2, lo[29:], hi[29:])
    mujoco.mj_step(m, d, 1)
    z = float(d.qpos[2])
    minz = min(minz, z)
    if z < 0.4:
        fell = i + 1
        break
print("PD-to-HOME stand (feet-collision ON):", "PASS" if fell is None else f"FAIL@{fell}", "minz", round(minz, 3))
