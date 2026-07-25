"""Render a fight video of the gen1 policy vs mocap opponent."""
import sys; sys.path.insert(0, ".")
from stable_baselines3 import PPO
from g1_selfplay_env import G1SelfPlayEnv
import numpy as np
import mujoco
import imageio

model = PPO.load("models/boxing_gen1.zip")
env = G1SelfPlayEnv(opponent_mocap=True, max_steps=1000, randomize=False)
obs, _ = env.reset(seed=42)

renderer = mujoco.Renderer(env.model, height=360, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 3.0
cam.elevation = -10

frames = []
hp0, hp1 = 100, 100
for i in range(500):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, info = env.step(a)
    hp0, hp1 = info["hp_0"], info["hp_1"]
    if i % 2 == 0:
        cam.lookat[:] = [0, 0, 0.8]
        cam.azimuth = 90 + 15 * np.sin(i * 0.02 * 0.3)
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())
    if term or trunc:
        break

imageio.mimsave("/workspace/repo/fight_render.mp4", frames, fps=30)
print("RENDERED", len(frames), "frames, hp:", hp0, "/", hp1)
