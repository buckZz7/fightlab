"""Street fight: 2 natural G1s in default MuJoCo space, bare-handed.

From-scratch minimal render: default navy checkerboard, two G1
humanoids facing each other, NO ring, NO gloves, NO body paint.
King vs challenger = a small headband color accent. Physics fist
sphere stays invisible (damage only).

Usage (on pod):
  python3 street_fight.py --steps 400 --out /tmp/street.mp4
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge
from league import _load_entrant


def render(spec_a, spec_b, balance, out, steps, cam=None):
    """Two fighters facing off in default space (open ring, no ring)."""
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=steps, randomize=False, ring="open")
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    judge = BoxingJudge(env, round_seconds=30.0, rounds=3)
    red = _load_entrant(spec_a, env, for_blue=False)

    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam = cam or {}
    c.azimuth = cam.get("az", 90.0)
    c.elevation = cam.get("el", 10.0)
    c.distance = cam.get("dist", 3.0)
    c.lookat[:] = cam.get("lookat", [-0.15, 0, 0.7])
    rend = mujoco.Renderer(env.model, height=540, width=960)

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < steps:
        a1 = red.predict(obs, deterministic=True)[0]
        obs, rew, term, trunc, info = judge.step(a1)
        rend.update_scene(env.data, camera=c)
        frames.append(rend.render())
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1

    if frames:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        import imageio.v2 as imageio
        seq = [f[..., ::-1] for f in frames]
        while len(seq) < 2:
            seq = seq + seq
        imageio.mimsave(out, seq, fps=30)
    return len(frames), judge.card()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="scripted:jabbler")
    ap.add_argument("--b", default="scripted:defender")
    ap.add_argument("--balance", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="/tmp/street.mp4")
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--el", type=float, default=10.0)
    ap.add_argument("--dist", type=float, default=3.0)
    ap.add_argument("--lookat", default="-0.15 0 0.7")
    a = ap.parse_args()
    cam = {"az": a.az, "el": a.el, "dist": a.dist,
           "lookat": [float(x) for x in a.lookat.split()]}
    n, card = render(a.a, a.b, a.balance, a.out, a.steps, cam)
    print(f"[street] {a.out} ({n} frames) winner={card['winner']} "
          f"method={card['method']} hp={card['final_hp']}")


if __name__ == "__main__":
    main()
