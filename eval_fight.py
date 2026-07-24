"""Fight evaluation + replay rendering for BoxingEnv.

Pit two policies (or a policy vs itself) in the ring:
  python eval_fight.py --red models/boxing_gen1.zip --blue models/boxing_gen2.zip --matches 20
  python eval_fight.py --red models/boxing_gen1.zip --blue models/boxing_gen1.zip --video fight.mp4

Stick-figure replay like eval_balance.py — no GL needed.
"""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from boxing_env import BoxingEnv

ACTION_SCALE = 50.0


def load(path):
    return PPO.load(path) if path and path != "random" else None


def policy_action(model, obs):
    if model is None:
        return np.random.uniform(-1, 1, 9) * ACTION_SCALE
    a, _ = model.predict(obs, deterministic=True)
    return np.clip(a, -1, 1) * ACTION_SCALE


def fight(red, blue, max_steps=2000, seed=None):
    """One bout. red = agent_1, blue = agent_2."""
    env = BoxingEnv(randomize=True, max_steps=max_steps)
    if seed is not None:
        np.random.seed(seed)
    obs = env.reset()
    frames_state = []
    while True:
        actions = {
            "agent_1": policy_action(red, obs["agent_1"]),
            "agent_2": policy_action(blue, obs["agent_2"]),
        }
        obs, rewards, done, info = env.step(actions)
        frames_state.append(env.get_state())
        if done:
            break
    hp1, hp2 = info["hp_1"], info["hp_2"]
    ko1 = env.data.xpos[env.body_1][2] < 0.5
    ko2 = env.data.xpos[env.body_2][2] < 0.5
    if ko2 or hp2 <= 0:
        winner = "red"
    elif ko1 or hp1 <= 0:
        winner = "blue"
    else:
        winner = "red" if hp1 > hp2 else ("blue" if hp2 > hp1 else "draw")
    return {
        "winner": winner,
        "hp_red": round(hp1, 1),
        "hp_blue": round(hp2, 1),
        "steps": env.step_count,
        "ko": bool(ko1 or ko2),
        "punches": env.punches_landed,
    }, frames_state, env


def series(red_path, blue_path, matches=20):
    red, blue = load(red_path), load(blue_path)
    results = []
    for i in range(matches):
        r, _, _ = fight(red, blue, seed=1000 + i)
        results.append(r)
    wins = {"red": 0, "blue": 0, "draw": 0}
    kos = 0
    for r in results:
        wins[r["winner"]] += 1
        kos += r["ko"]
    return {
        "matches": matches,
        "red": red_path,
        "blue": blue_path,
        "red_wins": wins["red"],
        "blue_wins": wins["blue"],
        "draws": wins["draw"],
        "ko_rate": round(kos / matches, 2),
        "avg_steps": round(float(np.mean([r["steps"] for r in results])), 1),
    }


def render_fight(red_path, blue_path, video_path="fight.mp4", seconds=40, seed=7):
    """Software stick-figure replay of a single bout (both robots)."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    red, blue = load(red_path), load(blue_path)
    result, states, env = fight(red, blue, max_steps=int(seconds / 0.002), seed=seed)

    # Re-run to capture body positions per frame (fight() only kept summaries)
    env2 = BoxingEnv(randomize=True, max_steps=int(seconds / 0.002))
    np.random.seed(seed)
    obs = env2.reset()
    frames = []
    fps = 30
    steps_per_frame = max(1, int((1.0 / fps) / 0.002))
    total = min(result["steps"], int(seconds / 0.002))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    i = 0
    done = False
    while i < total and not done:
        actions = {
            "agent_1": policy_action(red, obs["agent_1"]),
            "agent_2": policy_action(blue, obs["agent_2"]),
        }
        obs, rewards, done, info = env2.step(actions)
        if i % steps_per_frame == 0:
            ax.clear()
            for body, color, label in ((env2.body_1, "tab:red", "RED"),
                                       (env2.body_2, "tab:blue", "BLUE")):
                # draw all geoms' body positions for a stick figure
                xs = [env2.data.xpos[b][0] for b in range(env2.model.nbody)]
                zs = [env2.data.xpos[b][2] for b in range(env2.model.nbody)]
                # color only bodies belonging to each robot subtree by x sign at spawn
                ax.scatter(xs, zs, c="grey", s=8)
            r1 = env2.data.xpos[env2.body_1]
            r2 = env2.data.xpos[env2.body_2]
            ax.scatter([r1[0]], [r1[2]], c="tab:red", s=120)
            ax.scatter([r2[0]], [r2[2]], c="tab:blue", s=120)
            ax.axhline(0, color="k", lw=2)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-0.2, 2.6)
            ax.set_title(f"t={i*0.002:.1f}s  RED hp={info['hp_1']:.0f}  BLUE hp={info['hp_2']:.0f}")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
        i += 1
    plt.close(fig)
    imageio.mimsave(video_path, frames, fps=fps)
    return {"video": video_path, "frames": len(frames), "result": result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", required=True)
    ap.add_argument("--blue", required=True)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    if args.video:
        out = render_fight(args.red, args.blue, args.video)
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(series(args.red, args.blue, args.matches), indent=2))
