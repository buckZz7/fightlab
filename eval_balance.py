"""Evaluate a trained balance policy: push survival stats + replay video.

Usage:
  python eval_balance.py --model models/balance_ppo.zip --episodes 20
  python eval_balance.py --model models/balance_ppo.zip --video replay.mp4
"""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from envs import make_env


def evaluate(model_path, episodes=20, hard=False):
    model = PPO.load(model_path)
    kw = {}
    if hard:
        kw = dict(push_interval=80, push_jitter=40, push_force_range=(250.0, 600.0))
    env = make_env(**kw)
    ep_lens, survived, totals = [], 0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc
        ep_lens.append(steps)
        survived += info["pushes_survived"]
        totals += info["pushes_total"]
    return {
        "episodes": episodes,
        "mean_ep_len": float(np.mean(ep_lens)),
        "max_ep_len": int(max(ep_lens)),
        "pushes_survived": int(survived),
        "pushes_total": int(totals),
        "survival_rate": round(survived / max(totals, 1), 3),
        "hard_mode": hard,
    }


def render_replay(model_path, video_path="replay.mp4", seconds=30, seed=7):
    """Software-rendered stick-figure replay (no GL needed).

    Draws body segment positions projected to the x-z plane per frame.
    Good enough to verify behavior; real 3D mp4 rendering happens in CI
    where Mesa/EGL is available.
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = PPO.load(model_path)
    env = make_env()  # no render_mode: pure physics
    obs, _ = env.reset(seed=seed)
    frames = []
    fps = 30
    steps_per_frame = max(1, int((1.0 / fps) / 0.015))
    total_steps = int(seconds / 0.015)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
    fell_at = None
    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if i % steps_per_frame == 0:
            ax.clear()
            # body segment positions from the kinematic tree
            xs = env.data.xpos[:, 0]
            zs = env.data.xpos[:, 2]
            ax.scatter(xs, zs, c="tab:blue", s=25)
            torso = env.data.body("torso").xpos
            ax.scatter([torso[0]], [torso[2]], c="tab:red", s=80)
            ax.axhline(0, color="k", lw=2)
            ax.set_xlim(torso[0] - 3, torso[0] + 3)
            ax.set_ylim(-0.2, 2.6)
            ax.set_title(f"t={i*0.015:.1f}s  pushes survived={info['pushes_survived']}")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
        if (term or trunc) and fell_at is None:
            fell_at = i
            # keep recording ~1s of the fall
        if fell_at is not None and i > fell_at + int(1.0 / 0.015):
            break
    plt.close(fig)
    imageio.mimsave(video_path, frames, fps=fps)
    return {"video": video_path, "frames": len(frames)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    stats = evaluate(args.model, args.episodes, hard=args.hard)
    print(json.dumps(stats, indent=2))
    if args.video:
        out = render_replay(args.model, args.video)
        print(json.dumps(out))
