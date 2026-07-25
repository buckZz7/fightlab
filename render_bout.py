"""Render a King-of-the-Hill bout video with boxing-rules overlay.

Runs a full bout under BoxingJudge and renders it with round/score/HUD.
Usage:
  python render_bout.py RED_PATH BLUE_PATH --out bout.mp4 \
      --rounds 3 --round-seconds 30 --max-steps 2000
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mujoco
import imageio.v2 as imageio
from boxing_rules import run_bout, BoxingJudge, ROUNDS, ROUND_SECONDS
from g1_selfplay_env import G1SelfPlayEnv, make_g1_selfplay_env


def render_bout(red_path, blue_path, out, rounds=ROUNDS, round_seconds=ROUND_SECONDS,
                max_steps=2000, height=480, width=640, fps=30):
    env = make_g1_selfplay_env(opponent_path2=blue_path, randomize=False, max_steps=max_steps)
    judge = BoxingJudge(env, round_seconds=round_seconds, rounds=rounds)
    from stable_baselines3 import PPO
    red = PPO.load(red_path, env=env)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.2
    cam.elevation = -8
    cam.lookat[:] = [0, 0, 0.8]

    obs, _ = env.reset()
    frames = []
    hud = {"round": 1, "r_pts": 0.0, "b_pts": 0.0, "r_hp": 100, "b_hp": 100,
           "r_fouls": 0.0, "b_fouls": 0.0, "method": "", "winner": None}

    step = 0
    done = False
    while not done:
        a0 = red.predict(obs, deterministic=True)[0]
        obs, rew, term, trunc, info = judge.step(a0)
        done = term or trunc
        step += 1
        # update HUD
        hud["round"] = judge.round + 1
        hud["r_pts"] = judge.scores[0]
        hud["b_pts"] = judge.scores[1]
        hud["r_hp"] = env.hp[0]
        hud["b_hp"] = env.hp[1]
        hud["r_fouls"] = judge.foul_points[0]
        hud["b_fouls"] = judge.foul_points[1]
        if judge.ko:
            hud["method"] = "KO"
        elif judge.dq is not None:
            hud["method"] = "DQ"
        elif "decision" in info:
            hud["method"] = "DEC"

        if step % 2 == 0:
            cam.azimuth = 90 + 25 * np.sin(step * 0.004)
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            # HUD overlay (simple: draw via numpy)
            frame = draw_hud(frame, hud, red_path, blue_path)
            frames.append(frame)
        if done:
            break

    imageio.mimsave(out, frames, fps=fps)
    card = judge.card()
    winner = "RED" if card["winner"] == 0 else "BLUE"
    print(f"BOUT RENDERED: {len(frames)} frames -> {out}")
    print(f"Winner: {winner} by {card['method']} | "
          f"pts {card['total_points'][0]:.0f}-{card['total_points'][1]:.0f} | "
          f"HP {card['final_hp'][0]:.0f}-{card['final_hp'][1]:.0f} | "
          f"fouls {card['foul_points']}")
    return card


def draw_hud(frame, hud, red_path, blue_path):
    """Lightweight HUD: top bar with round, scores, HP bars (numpy only)."""
    try:
        import cv2
    except Exception:
        return frame  # no opencv -> raw frame
    h, w = frame.shape[:2]
    # top bar
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    # round
    cv2.putText(frame, f"R{hud['round']}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    # red (left) score + hp
    cv2.putText(frame, f"RED {hud['r_pts']:.0f}", (120, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 160, 255), 2)
    cv2.rectangle(frame, (260, 12), (260 + int(hud['r_hp'] * 1.4), 28),
                  (80, 160, 255), -1)
    cv2.putText(frame, f"{hud['r_hp']:.0f}", (420, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # blue (right) score + hp
    cv2.putText(frame, f"BLU {hud['b_pts']:.0f}", (w - 320, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 80), 2)
    cv2.rectangle(frame, (w - 260, 12), (w - 260 + int(hud['b_hp'] * 1.4), 28),
                  (255, 120, 80), -1)
    cv2.putText(frame, f"{hud['b_hp']:.0f}", (w - 100, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # fouls
    cv2.putText(frame, f"fouls R{hud['r_fouls']:.0f} B{hud['b_fouls']:.0f}",
                (w // 2 - 70, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 200, 80), 1)
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("red")
    ap.add_argument("blue")
    ap.add_argument("--out", default="bout.mp4")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--round-seconds", type=float, default=ROUND_SECONDS)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    render_bout(args.red, args.blue, args.out, rounds=args.rounds,
                round_seconds=args.round_seconds, max_steps=args.max_steps,
                fps=args.fps)
