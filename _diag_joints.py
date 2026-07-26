import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np
import mujoco
import sys
sys.path.insert(0, ".")
from g1_arena import build_arena
from loco_base29 import HOME, KP, KD

m = build_arena(ring="ropes", half=2.4)
print("nq pos joints (excl free):", m.njnt - 1, "(should be 29)")
names = []
for i in range(m.njnt):
    jid = i
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    names.append(nm)
print("joint order in model:")
for i, nm in enumerate(names):
    print(f"  [{i:2d}] {nm}")
print("\nHOME len:", len(HOME))
print("HOME[:5]:", HOME[:5])
print("KP len:", len(KP))
# Check: does joint i in model correspond to HOME[i]? Print HOME aligned
print("\nmodel joint -> HOME (rad):")
for i, nm in enumerate(names[1:], start=0):  # skip free joint idx 0
    if i < len(HOME):
        print(f"  [{i:2d}] {nm:30s} HOME={HOME[i]:.3f}")
