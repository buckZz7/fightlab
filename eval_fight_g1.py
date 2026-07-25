"""G1 fight evaluation: pit two policies in the boxing arena.

  python eval_fight_g1.py --red models/boxing_gen1.zip --blue models/boxing_gen2.zip --matches 20
  python eval_fight_g1.py --red models/boxing_gen1.zip --blue random --video fight.mp4

Renders via MuJoCo offscreen (OSMesa for CPU, EGL for GPU).
"""
import argparse
import json
import os

import numpy as np
import mujoco

from g1_selfplay_env import G1SelfPlayEnv, N_SKILL

RESIDUAL_SCALE = np.array(
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2] +
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2])


def load(path):
    if not path or path == "random":
        return None
    from stable_baselines3 import PPO
    return PPO.load(path)


def policy_action(model, obs):
    if model is None:
        return np.random.uniform(-1, 1, N_SKILL)
    a, _ = model.predict(obs, deterministic=True)
    return np.clip(a, -1, 1)


def fight(red, blue, max_steps=2000, seed=None):
    """One bout. red = agent 0 (challenger), blue = agent 1 (opponent).

    We run the env with red as the trained agent and blue as the opponent.
    The env always trains agent 0, so we swap roles for blue's perspective.
    """
    # Run red as challenger, blue as opponent
    env = G1SelfPlayEnv(opponent_model=blue, max_steps=max_steps, randomize=True)
    obs, _ = env.reset(seed=seed)
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_steps):
        action = policy_action(red, obs)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break

    z0, z1 = info["pelvis_z_0"], info["pelvis_z_1"]
    hp0, hp1 = info["hp_0"], info["hp_1"]

    ko0 = z0 < 0.4
    ko1 = z1 < 0.4

    if ko1 or hp1 <= 0:
        winner = "red"
    elif ko0 or hp0 <= 0:
        winner = "blue"
    elif hp0 > hp1:
        winner = "red"
    elif hp1 > hp0:
        winner = "blue"
    else:
        winner = "draw"

    return {
        "winner": winner,
        "hp_red": round(hp0, 1),
        "hp_blue": round(hp1, 1),
        "z_red": round(z0, 3),
        "z_blue": round(z1, 3),
        "steps": i + 1,
        "ko": bool(ko0 or ko1),
    }


def series(red_path, blue_path, matches=20):
    red, blue = load(red_path), load(blue_path)
    results = []
    for i in range(matches):
        r = fight(red, blue, seed=1000 + i)
        results.append(r)
        w = r["winner"]
        print(f"  Match {i+1}: {w:4s} (R:{r['hp_red']:.0f} B:{r['hp_blue']:.0f} "
              f"{'KO' if r['ko'] else 'DEC'} {r['steps']}s)")

    wins = {"red": 0, "blue": 0, "draw": 0}
    kos = 0
    for r in results:
        wins[r["winner"]] += 1
        kos += r["ko"]

    summary = {
        "matches": matches,
        "red": red_path,
        "blue": blue_path,
        "red_wins": wins["red"],
        "blue_wins": wins["blue"],
        "draws": wins["draw"],
        "ko_rate": round(kos / matches, 2),
        "avg_steps": round(float(np.mean([r["steps"] for r in results])), 1),
    }
    return summary


def render_fight(red_path, blue_path, video_path="fight.mp4", seconds=40, seed=7):
    """MuJoCo offscreen render of a single bout."""
    import imageio

    red, blue = load(red_path), load(blue_path)
    env = G1SelfPlayEnv(opponent_model=blue, max_steps=int(seconds / 0.02),
                        randomize=False)
    obs, _ = env.reset(seed=seed)

    # Try OSMesa, fall back to EGL
    renderer = mujoco.Renderer(env.model, height=360, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.0
    cam.elevation = -10

    frames = []
    fps = 30
    every = max(1, int((1.0 / fps) / 0.02))
    pelvis = env.model.body("r1_pelvis").id

    for i in range(int(seconds / 0.02)):
        action = policy_action(red, obs)
        obs, r, term, trunc, info = env.step(action)
        if i % every == 0:
            p = env.data.xpos[pelvis]
            cam.lookat[:] = [0, 0, 0.8]
            cam.azimuth = 90 + 15 * np.sin(i * 0.02 * 0.3)
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
        if term or trunc:
            break

    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved {len(frames)} frames to {video_path}")
    return {"video": video_path, "frames": len(frames),
            "final_hp": f"{info['hp_0']:.0f} / {info['hp_1']:.0f}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="G1 fight evaluation")
    ap.add_argument("--red", default="random", help="Red (challenger) model path or 'random'")
    ap.add_argument("--blue", default="random", help="Blue (opponent) model path or 'random'")
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--video", default=None, help="Render a single bout to video")
    ap.add_argument("--seconds", type=int, default=40, help="Video duration in seconds")
    args = ap.parse_args()

    if args.video:
        r = render_fight(args.red, args.blue, args.video, args.seconds)
        print(json.dumps(r, indent=2))
    else:
        summary = series(args.red, args.blue, args.matches)
        print(json.dumps(summary, indent=2))
