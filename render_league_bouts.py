"""Render league bouts to MP4s for the king-of-the-hill page.

Reads league_standings.json, renders the top matchups (king's bouts +
a couple of others) to docs/bouts/*.mp4 via bout_fighter-style render.

Usage:
  python3 render_league_bouts.py --standings docs/league_test.json \
      --pd --steps 600 --out-dir docs/bouts
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge
from league import _load_entrant


def render_bout(spec_a, spec_b, balance, out, steps, king_of=None):
    """king_of: name of the king fighter among {spec_a, spec_b} -> its
    robot gets RED gloves (the other is the BLUE challenger)."""
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=steps, randomize=False)
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    # king role: red gloves to whichever entrant is the king.
    # r1 = red-side (drives via env.step), r2 = blue-side (opponent).
    if king_of == spec_a:
        env._color_gloves_by_king(0)   # r1 is king -> red
    elif king_of == spec_b:
        env._color_gloves_by_king(1)   # r2 is king -> red
    judge = BoxingJudge(env, round_seconds=30.0, rounds=3)
    red = _load_entrant(spec_a, env, for_blue=False)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # TIGHT broadcast close-up: fighters fill the frame (dist ~2.2),
    # ring ropes at the edges as context. Side-on (az=90) so both bots
    # are in profile facing each other. el=10 slight above.
    cam.azimuth = 90.0; cam.elevation = 10.0; cam.distance = 2.2
    cam.lookat[:] = [-0.15, 0, 0.95]
    rend = mujoco.Renderer(env.model, height=540, width=960)

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < steps:
        a1 = red.predict(obs, deterministic=True)[0]
        obs, rew, term, trunc, info = judge.step(a1)
        rend.update_scene(env.data, camera=cam)
        frames.append(rend.render())
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1
    if frames:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        import imageio.v2 as imageio
        # mimsave for a frame SEQUENCE (imsave = single image).
        # pad short bouts so the container has >= 2 frames.
        seq = [f[..., ::-1] for f in frames]
        while len(seq) < 2:
            seq = seq + seq
        imageio.mimsave(out, seq, fps=30)
    card = judge.card()
    return len(frames), card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_standings.json")
    ap.add_argument("--pd", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out-dir", default="docs/bouts")
    ap.add_argument("--max-bouts", type=int, default=4)
    a = ap.parse_args()
    balance = None if a.pd else a.balance

    d = json.load(open(a.standings))
    results = d["results"]
    king = d.get("king")
    # pick the top matchups: king's bouts + spread
    rendered = []
    seen = set()
    for r in results:
        key = (r["red"], r["blue"])
        if key in seen:
            continue
        seen.add(key)
        if len(rendered) >= a.max_bouts:
            break
        out = os.path.join(a.out_dir,
                            f"{r['red'].replace(':', '_')}_vs_"
                            f"{r['blue'].replace(':', '_')}.mp4")
        n, card = render_bout(r["red"], r["blue"], balance, out, a.steps,
                               king_of=king)
        r["mp4"] = os.path.relpath(out, os.path.dirname(a.standings))
        r["n_frames"] = n
        r["card"] = card
        rendered.append(r)
        print(f"[rendered] {out} ({n} frames) winner={card['winner']}")

    # write back the updated results (with mp4 paths)
    d["results"] = results
    with open(a.standings, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[done] {len(rendered)} bouts rendered -> {a.out_dir}")


if __name__ == "__main__":
    main()
