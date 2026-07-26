"""2-bot Track B bout: render a fight between two fighter policies.

Red (r1) vs Blue (r2). Reuses G1FighterEnv (all damage/facing/
contact logic) with p1 as the trained fighter (r1) and p2 as opponent
(r2). The env loads the frozen balance policy itself via
`balance_path` (the substrate), and r2 is driven by `opponent_path`
(r2's own fighter policy) OR a frozen StandPD sandbag.

Outputs an MP4 + prints HP.

Usage:
  # real run:
  python3 bout_fighter.py --p1 models/fighter_v1 \
      --balance models/balance_v1 [--p2 models/fighter_v1] \
      --out docs/fighter_bout.mp4 --steps 1500
  # smoke test (no fighter model yet): omit --p1; uses a random
  # fighter actor (the env's balance substrate still stands both bots).
  python3 bout_fighter.py --balance /tmp/bal_test \
      --out /tmp/smoke_bout.mp4 --steps 400
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv


class RandomFighter:
    """Placeholder fighter: random 17-dim actions. Both bots still
    stand via the frozen balance substrate."""
    def __init__(self, env):
        self.env = env
    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default=None,
                        help="fighter policy for r1 (None = random)")
    ap.add_argument("--p2", default=None,
                        help="fighter policy for r2 (None = frozen sandbag)")
    ap.add_argument("--balance", required=True,
                        help="balance (substrate) policy path")
    ap.add_argument("--out", default="docs/fighter_bout.mp4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--max_round_seconds", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    env = G1FighterEnv(balance_path=a.balance, opponent_path=a.p2,
                       max_steps=a.steps, randomize=False)
    from boxing_rules import BoxingJudge
    judge = BoxingJudge(env, round_seconds=a.max_round_seconds,
                         rounds=a.rounds)

    p1 = PPO.load(a.p1) if a.p1 else RandomFighter(env)

    rend = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < a.steps:
        a1, _ = p1.predict(obs, deterministic=True)
        # env.step takes r1's action; r2 is driven internally
        # (by opponent_path fighter policy, or the frozen sandbag).
        obs, rew, term, trunc, info = judge.step(a1)
        rend.update_scene(env.data, camera=cam)
        frames.append(rend.render())
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1

    if frames:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(a.out, fourcc, 30, (640, 480))
            for f in frames:
                vw.write(np.ascontiguousarray(f[..., ::-1]))  # RGB->BGR
            vw.release()
        except Exception:
            import imageio.v2 as imageio
            imageio.imsave(a.out, [f[..., ::-1] for f in frames], fps=30)
        print(f"[saved] {a.out} ({len(frames)} frames)")

    card = judge.card()
    print(f"[bout] CARD: winner={card['winner']} method={card['method']} "
          f"hp={card['final_hp']} rounds={card['round_scores']}")


if __name__ == "__main__":
    main()
