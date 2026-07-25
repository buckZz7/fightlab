"""Two-process 3D render pipeline (avoids torch+EGL segfault in one process).

Step 1 (this file): rollout a trained policy, dump per-frame trajectory to npz.
Step 2 (render_traj_3d.py): load npz, render with MuJoCo EGL, write mp4.

Usage:
  python dump_traj.py --model /workspace/g1_mocap_punch.zip --out /workspace/traj.npz --seconds 12
"""
import argparse

import numpy as np


def dump(model_path, out, seconds=12, seed=7):
    from stable_baselines3 import PPO
    from g1_mocap_punch_env import G1MocapPunchEnv

    env = G1MocapPunchEnv("mocap/kungfu_retargeted/Horse-stance_punch.pkl",
                          randomize=False)
    obs, _ = env.reset(seed=seed)
    model = PPO.load(model_path)
    fps = 30
    every = max(1, int((1.0 / fps) / 0.02))
    steps = int(seconds / 0.02)

    qpos_frames, qvel_frames, infos = [], [], []
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, t, tr, info = env.step(a)
        if i % every == 0:
            qpos_frames.append(env.data.qpos.copy())
            qvel_frames.append(env.data.qvel.copy())
            infos.append({"bag_vel": info["bag_vel"], "pelvis_z": info["pelvis_z"], "t": i * 0.02})
        if t or tr:
            obs, _ = env.reset(seed=seed + 1)
    np.savez_compressed(out,
                        qpos=np.array(qpos_frames),
                        qvel=np.array(qvel_frames),
                        infos=np.array(infos, dtype=object))
    print({"traj": out, "frames": len(qpos_frames)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/workspace/traj.npz")
    ap.add_argument("--seconds", type=float, default=12)
    args = ap.parse_args()
    dump(args.model, args.out, args.seconds)
