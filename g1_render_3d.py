"""Real 3D G1 renderer via MuJoCo's offscreen renderer (EGL hardware GL).

Runs on a GPU pod. Renders the actual G1 mesh geometry with materials,
floor, shadows — not stick figures. Camera: ringside side view with a
slow orbit for drama.

Usage (on a GL-capable machine):
  MUJOCO_GL=egl python g1_render_3d.py --model models/g1_mocap_punch.zip --out punch_3d.mp4
"""
import argparse

import imageio
import mujoco
import numpy as np


def render_rollout(model_path, out, seconds=12, seed=7,
                   width=1280, height=720, fps=30):
    from stable_baselines3 import PPO
    from g1_mocap_punch_env import G1MocapPunchEnv

    env = G1MocapPunchEnv("mocap/kungfu_retargeted/Horse-stance_punch.pkl",
                          randomize=False)
    obs, _ = env.reset(seed=seed)
    model = PPO.load(model_path)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.6
    cam.elevation = -12

    frames = []
    steps = int(seconds / 0.02)
    every = max(1, int((1.0 / fps) / 0.02))
    pelvis = env.model.body("pelvis").id
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, t, tr, info = env.step(a)
        if i % every == 0:
            # slow orbit around the action
            cam.azimuth = 90 + 12 * np.sin(i * 0.005)
            p = env.data.xpos[pelvis]
            cam.lookat[:] = [p[0], p[1], 1.0]
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
        if t or tr:
            obs, _ = env.reset(seed=seed + 1)
    imageio.mimsave(out, frames, fps=fps)
    print({"video": out, "frames": len(frames)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="punch_3d.mp4")
    ap.add_argument("--seconds", type=float, default=12)
    args = ap.parse_args()
    render_rollout(args.model, args.out, args.seconds)
