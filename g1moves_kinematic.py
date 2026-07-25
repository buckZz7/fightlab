"""Kinematic playback of a G1 Moves NPZ motion in our MuJoCo scene.

Sets qpos directly from joint_pos (no dynamics, no policy). Proves the motion
maps to our model and renders a clean kick/punch video. No IMU/obs needed.
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio.v2 as imageio

XML = os.environ.get("G1_SCENE_XML", "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--out", default="/tmp/kin.mp4")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()

    n = np.load(args.npz)
    jp = n["joint_pos"].astype(np.float32)
    fps = float(np.asarray(n["fps"]).item())
    N = jp.shape[0]
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / fps
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    min_z = 1.0
    for _ in range(args.loops):
        for t in range(N):
            q = np.zeros(model.nq)
            q[2] = 0.78
            q[7:36] = jp[t]
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            min_z = min(min_z, data.qpos[2])
            renderer.update_scene(data)
            frames.append(renderer.render())
    imageio.mimsave(args.out, frames, fps=int(args.fps))
    print(f"saved {args.out} | frames={len(frames)} | min_pelvis_z={min_z:.3f} (kinematic, no dynamics)")

if __name__ == "__main__":
    main()
