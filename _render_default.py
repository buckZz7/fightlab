"""Render a frame using build_default_2bot (G1 default scene + 2 robots).
Uses the same OSMesa setup as single_frame.py (which works).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")  # MUST be before any mujoco import
import numpy as np
import mujoco
from street_arena import build_default_2bot

model = build_default_2bot()
data = mujoco.MjData(model)
for ai, x in enumerate([-0.6, 0.3]):
    off = ai * 36
    data.qpos[off:off + 3] = [x, 0, 0.76]
    data.qpos[off + 3:off + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera()
rend = mujoco.Renderer(model, height=540, width=960)
rend.update_scene(data, camera=cam)
img = rend.render()
import PIL.Image
PIL.Image.fromarray(img).save("/tmp/pure_default.png")
print("saved", img.shape)
