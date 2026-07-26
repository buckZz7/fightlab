"""Render using the built-in broadcast tracking camera."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from street_arena import build_default_2bot

model = build_default_2bot()
data = mujoco.MjData(model)
for ai, x in enumerate([-0.6, 0.3]):
    off = ai * 36
    data.qpos[off:off + 3] = [x, 0, 0.793]
    data.qpos[off + 3:off + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

# Use the named broadcast camera
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
print(f"broadcast camera id={cam_id}, ncam={model.ncam}")

rend = mujoco.Renderer(model, height=540, width=960)
rend.update_scene(data, camera=cam_id)
img = rend.render()
import PIL.Image
PIL.Image.fromarray(img).save("/tmp/broadcast_track.png")
print("saved", img.shape)
