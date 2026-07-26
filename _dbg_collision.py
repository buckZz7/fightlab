import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np, sys, mujoco
sys.path.insert(0, ".")
from g1_arena import build_arena

m = build_arena(ring="ropes", half=2.4)
# Find all geom body names containing ankle/foot and their contype
print("=== ankle/foot geoms after build_arena ===")
for i in range(m.ngeom):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
    if "ankle" in bn or "foot" in bn:
        print(f"  geom{i} body={bn} type={m.geom_type[i]} contype={m.geom_contype[i]} conaff={m.geom_conaffinity[i]}")
print("=== lowest geom (min z extent) of r1 ===")
# place r1 at HOME and find geoms with smallest world z
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [-0.6, 0, 0.793]
d.qpos[7:36] = 0.0  # neutral-ish; just to compute positions
mujoco.mj_forward(m, d)
# find r1 geoms with body containing 'r1' and smallest xpos z
lo_z = 9
lo_name = ""
for i in range(m.ngeom):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
    if bn.startswith("r1"):
        z = float(d.geom_xpos[i][2])
        if z < lo_z:
            lo_z = z
            lo_name = bn
print(f"  lowest r1 geom: {lo_name} z={lo_z:.3f}")
