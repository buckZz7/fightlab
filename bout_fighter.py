"""2-bot Track B bout: render a fight between two fighter policies.

Red (r1) vs Blue (r2). Reuses G1FighterEnv (all damage/facing/
contact logic) with p1 as the trained fighter and p2 as opponent.
Outputs an MP4 + prints HP.

Usage:
  python3 bout_fighter.py --p1 models/fighter_v1 \\
      --balance models/balance_v1 [--p2 models/fighter_v1] \\
      --out docs/fighter_bout.mp4 --steps 1500
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv
from g1_arena import build_arena


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", default="")
    ap.add_argument("--balance", required=True)
    ap.add_argument("--out", default="docs/fighter_bout.mp4")
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()

    p1 = PPO.load(a.p1)
    p2 = PPO.load(a.p2) if a.p2 else None
    env = G1FighterEnv(balance_path=a.balance, opponent_path=a.p2 or None,
                       max_steps=a.steps, randomize=False)

    rend = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]

    o, _ = env.reset()
    frames = []
    for step in range(a.steps):
        # r1 acts via p1; r2 acts via p2 (or env's frozen sandbag)
        a1, _ = p1.predict(o, deterministic=True)
        if p2 is not None:
            a2, _ = p2.predict(env._get_obs(1), deterministic=True)
            # env.step takes r1's action; r2 is handled internally via opponent
            act = a1
        else:
            act = a1
        o, r, term, trunc, info = env.step(act)
        if step % 2 == 0:
            rend.update_scene(env.data, camera=cam)
            frames.append(rend.render())
        if term or trunc:
            break

    print(f"bout ended step {step}: hp0={env.hp[0]:.0f} hp1={env.hp[1]:.0f}")
    try:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        imageio.mimsave(a.out, frames, fps=30)
        print(f"[saved] {a.out} ({len(frames)} frames)")
    except Exception as e:
        print("[warn] mp4 write failed:", e)
        for i, f in enumerate(frames[:3]):
            import PIL.Image
            PIL.Image.fromarray(f).save(f"{a.out}.{i}.png")
        print("wrote PNG frames instead")


if __name__ == "__main__":
    main()
