import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np
import mujoco
import sys
sys.path.insert(0, ".")
from g1_arena import build_arena
from loco_base29 import StandPD, KP, KD, HOME

m = build_arena(ring="ropes", half=2.4)
d = mujoco.MjData(m)
lo = m.actuator_ctrlrange[:, 0].copy()
hi = m.actuator_ctrlrange[:, 1].copy()
# place r1 at native-ish
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [-0.6, 0, 0.793]
d.qpos[7:36] = HOME
mujoco.mj_forward(m, d)

spd = StandPD()
minz = 9.0
first_drop = None
for i in range(1500):
    tau = KP * (HOME - d.qpos[7:36]) - KD * d.qvel[6:35]
    d.ctrl[:29] = np.clip(tau, lo[:29], hi[:29])
    # r2 stand too
    tau2 = KP * (HOME - d.qpos[14:43]) - KD * d.qvel[13:42]
    d.ctrl[29:58] = np.clip(tau2, lo[29:], hi[29:])
    mujoco.mj_step(m, d, 1)
    z = float(d.qpos[2])
    minz = min(minz, z)
    if z < 0.5 and first_drop is None:
        first_drop = i + 1
        break
print(f"PD-to-HOME stiff gains: fell@{first_drop} minz={minz:.3f} end_step={i+1}")
print("STAND:", "PASS (holds)" if first_drop is None else "FAIL")
