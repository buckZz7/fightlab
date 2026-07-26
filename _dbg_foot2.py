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

foot_bodies = {"r1_left_ankle_roll_link", "r1_right_ankle_roll_link",
               "r2_left_ankle_roll_link", "r2_right_ankle_roll_link",
               "r1_left_ankle_pitch_link", "r1_right_ankle_pitch_link",
               "r2_left_ankle_pitch_link", "r2_right_ankle_pitch_link"}

def foot_contacts():
    n = 0
    for c in range(d.ncon):
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[c].geom1])
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[c].geom2])
        if b1 in foot_bodies or b2 in foot_bodies:
            n += 1
    return n

for i in range(30):
    tau = KP * (HOME - d.qpos[7:36]) - KD * d.qvel[6:35]
    d.ctrl[:29] = np.clip(tau, lo[:29], hi[:29])
    tau2 = KP * (HOME - d.qpos[43:72]) - KD * d.qvel[41:70]
    d.ctrl[29:58] = np.clip(tau2, lo[29:], hi[29:])
    mujoco.mj_step(m, d, 1)
    if i in (0, 5, 10, 29):
        # lowest foot z
        lz = 9
        for gi in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]) or ""
            if bn in foot_bodies:
                lz = min(lz, float(d.geom_xpos[gi][2]))
        print(f"step {i}: ncon={d.ncon} foot_contacts={foot_contacts()} lowest_foot_z={lz:.3f} pelvis_z={float(d.qpos[2]):.3f}")
