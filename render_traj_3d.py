"""Render a dumped trajectory npz to 3D mp4 via MuJoCo EGL (no torch in proc).

Usage:
  MUJOCO_GL=osmesa python render_traj_3d.py --traj /workspace/traj.npz --out /workspace/punch_3d.mp4
"""
import argparse

import imageio
import mujoco
import numpy as np

from g1_mocap_punch_env import build_model


def render(traj_path, out, width=1280, height=720, fps=30):
    z = np.load(traj_path, allow_pickle=True)
    qpos_frames = z["qpos"]
    infos = list(z["infos"])

    model = build_model(with_bag=True)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.6
    cam.elevation = -12
    pelvis = model.body("pelvis").id

    frames = []
    for i, qp in enumerate(qpos_frames):
        data.qpos[:] = qp
        data.qvel[:] = z["qvel"][i]
        mujoco.mj_forward(model, data)
        cam.azimuth = 90 + 12 * np.sin(i * 0.02)
        p = data.xpos[pelvis]
        cam.lookat[:] = [p[0], p[1], 1.0]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        # simple HUD: bag vel text via matplotlib-free overlay skip (keep clean 3D)
        frames.append(img)
    imageio.mimsave(out, frames, fps=fps)
    print({"video": out, "frames": len(frames)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="/workspace/punch_3d.mp4")
    args = ap.parse_args()
    render(args.traj, args.out)
