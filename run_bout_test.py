"""End-to-end bout pipeline TEST (runs WITHOUT a trained model).

Exercises the full 1v1 eval path so it's proven before the real
balance_v1/fighter_v1 exist:
    G1FighterEnv (random policy) -> BoxingJudge (3 rounds) -> render MP4

Usage:
  python3 run_bout_test.py --out bout_test.mp4 --steps 1500
When a real policy exists, swap the random actor for PPO.load(...) and
pass opponent_path=fighter_v1 to G1FighterEnv.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge


class RandomActor:
    """Placeholder policy: random actions. Stands via frozen balance."""
    def __init__(self, env):
        self.env = env
    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", default=None,
                    help="balance policy path (None = PD-to-HOME placeholder)")
    ap.add_argument("--out", default="bout_test.mp4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--max_round_seconds", type=float, default=3.0)  # short for test
    a = ap.parse_args()

    env = G1FighterEnv(balance_path=a.balance, opponent_path=None,
                       max_steps=a.steps, randomize=False)
    judge = BoxingJudge(env, round_seconds=a.max_round_seconds, rounds=3)

    # Red = random actor (placeholder). Blue = frozen StandPD sandbag.
    red = RandomActor(env)

    rend = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < a.steps:
        a0 = red.predict(obs)[0]
        obs, rew, term, trunc, info = judge.step(a0)
        # render
        rend.update_scene(env.data, camera=cam)
        frames.append(rend.render())
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1

    # save MP4
    if frames:
        import cv2  # opencv may not be on pod; fall back to imageio
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(a.out, fourcc, 30, (640, 480))
            for f in frames:
                vw.write(np.ascontiguousarray(f[..., ::-1]))  # RGB->BGR
            vw.release()
        except Exception:
            import imageio.v2 as imageio
            imageio.mimsave(a.out, [f[..., ::-1] for f in frames], fps=30)
        print(f"[bout_test] wrote {len(frames)} frames -> {a.out}")

    card = judge.card()
    print("[bout_test] CARD:", {k: card[k] for k in
          ["round_scores", "total_points", "winner", "method", "final_hp"]})
    print("[bout_test] OK -- pipeline runs end-to-end")


if __name__ == "__main__":
    main()
