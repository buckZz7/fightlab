import sys
sys.path.insert(0, "/workspace/repo")
print("1: imports starting")
import mujoco
print("2: mujoco ok")
import numpy as np
print("3: numpy ok")
from g1_mocap_punch_env import G1MocapPunchEnv
print("4: env import ok")
env = G1MocapPunchEnv("mocap/kungfu_retargeted/Horse-stance_punch.pkl", randomize=False)
print("5: env built")
obs, _ = env.reset(seed=7)
print("6: reset ok")
r = mujoco.Renderer(env.model, 480, 640)
print("7: renderer ok")
r.update_scene(env.data)
img = r.render()
print("8: RENDER OK", img.shape)
