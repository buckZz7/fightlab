"""2-bot Track B bout: render a fight between two fighter policies.

Red (r1) vs Blue (r2). Reuses G1FighterEnv (all damage/facing/
contact logic) with p1 as the trained fighter (r1) and p2 as opponent
(r2). The env loads the frozen balance policy itself via
`balance_path` (the substrate); r2 is driven by `opponent_path`
(r2's own fighter policy) OR a frozen StandPD sandbag.

Outputs an MP4 + prints HP + a scored BoutCard.

Camera is TUNABLE from the CLI so the ropes never sit between
the lens and the bots (the default is an elevated 3/4 ring view):
  --cam_az  (deg, 0 = +X / along ring axis; -135 = diagonal)
  --cam_el  (deg, +up)
  --cam_dist (m)
  --cam_lookat "x y z" (defaults to ring center, chest height)

Usage:
  # real run (trained fighters):
  python3 bout_fighter.py --p1 models/fighter_v1 \
       --balance models/balance_v1 [--p2 models/fighter_v1] \
       --out docs/fighter_bout.mp4 --steps 1500
  # DEMO (no trained fighter yet): scripted shadowboxers so the
  # bots actually punch + footwork. Uses the balance substrate to
  # keep them standing. --cam_* to tune the shot.
  python3 bout_fighter.py --balance /tmp/bal_test \
       --demo --out /tmp/demo_bout.mp4 --steps 900
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge


class ShadowBoxer:
    """Scripted boxer: drives arm joints (14) with jab/cross/guard
    trajectories + footwork (3) to close distance. Both bots stand
    via the frozen balance substrate; this actor only modulates the
    arm residuals + walk cmd so it LOOKS like a fight.

    action (17) = [arm residual 14 | walk cmd 3]
      arm residual 14 -> HOME[15:29] (shoulders/elbows/wrists)
      walk cmd 3        -> (vx, vy, wz) * [0.5, 0.3, 1.0]
    """
    def __init__(self, env, style="red", profile="balanced"):
        self.env = env
        self.style = style
        self.profile = profile
        # which side leads: red leads with RIGHT (cross), blue with LEFT (jab)
        self.lead = 0 if style == "blue" else 1  # 0=joint-block L, 1=R
        self.t = 0.0
        self.phase = 0.0 if style == "red" else math.pi  # desync the two
        # profile tuning: punch cadence (rate), aggressiveness (walk), guard
        if profile == "jabbler":      # fast jabs, closes distance hard
            self.cadence = 3.2; self.walk_fwd = 0.6; self.punch_amp = 0.9
        elif profile == "defender":   # guard-heavy, low aggression, counter
            self.cadence = 1.6; self.walk_fwd = 0.15; self.punch_amp = 0.6
        else:                          # balanced
            self.cadence = 2.4; self.walk_fwd = 0.4; self.punch_amp = 0.8

    def predict(self, obs, deterministic=True):
        self.t += 1
        dt = self.env.model.opt.timestep * self.env.frame_skip
        # desync the two bots by PI so it reads as an EXCHANGE
        # (red punches while blue guards, then swap), not mirrored sync.
        self.phase += dt * self.cadence
        p = self.phase if self.style == "red" else self.phase + math.pi

        arm = np.zeros(14)
        arm = np.zeros(14)
        # Arm joints are qpos 22:36 (NOT 15:29). Layout
        # (arm action idx -> joint):
        #   0 L_sh_p  1 L_sh_r  2 L_sh_y  3 L_elb
        #   4 L_wr_r  5 L_wr_p  6 L_wr_y
        #   7 R_sh_p  8 R_sh_r  9 R_sh_y 10 R_elb
        #  11 R_wr_r 12 R_wr_p 13 R_wr_y
        # MuJoCo G1: shoulder_pitch NEGATIVE drives the arm
        # FORWARD+toward opponent (FK-verified: sh=-1.3 -> wrist
        # 0.33m ahead of pelvis, z raised to 0.98). ELBOW
        # POSITIVE = bent; straight punch ~0.1, guard ~1.3.
        # GUARD: shoulder moderately forward/up, elbow bent (hand
        # comes up near chest, reads as a clear guard).
        arm[0] = -0.7;  arm[3] = 1.3      # L guard (fwd + bent)
        arm[7] = -0.7;  arm[10] = 1.3     # R guard

        # PUNCH: lead arm extends forward (shoulder -1.0 max, elbow
        # straightens to ~0.1). Cap at -1.0 so the arm drives FORWARD
        # at the opponent, not up toward the ceiling at peak.
        atk = max(0.0, math.sin(p)) ** 2
        amp = 0.3 * self.punch_amp
        if self.lead == 1:  # red throws RIGHT cross
            arm[7] = -0.7 - amp * atk         # shoulder drives forward (cap -1.0)
            arm[10] = 1.3 - 1.2 * atk         # elbow straightens
        else:               # blue throws LEFT jab
            arm[0] = -0.7 - amp * atk
            arm[3] = 1.3 - 1.2 * atk
        # COUNTER from rear arm on off-beat.
        rear = max(0.0, math.sin(p + math.pi)) ** 2
        if self.lead == 1:
            arm[0] = -0.7 - amp * rear
            arm[3] = 1.3 - 1.1 * rear
        else:
            arm[7] = -0.7 - amp * rear
            arm[10] = 1.3 - 1.1 * rear

        # Footwork: shuffle forward toward center, weave a little.
        # walk_fwd scales aggression (jabbler advances hard, defender
        # stays planted and counters).
        walk = np.array([self.walk_fwd + 0.2 * math.sin(p * 0.5),   # vx
                         0.15 * math.sin(p * 0.9),                     # vy
                         0.4 * math.sin(p * 0.4)])                    # wz
        act = np.concatenate([np.clip(arm, -1, 1), walk]).astype(np.float64)
        return act, None


def _make_camera(args):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = args.cam_az
    cam.elevation = args.cam_el
    cam.distance = args.cam_dist
    cam.lookat[:] = [float(x) for x in args.cam_lookat.split()]
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default=None,
                    help="fighter policy for r1 (None = shadowboxer demo)")
    ap.add_argument("--p2", default=None,
                    help="fighter policy for r2 (None = shadowboxer demo)")
    ap.add_argument("--balance", required=True,
                    help="balance (substrate) policy path")
    ap.add_argument("--out", default="docs/fighter_bout.mp4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--max_round_seconds", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--demo", action="store_true",
                    help="use scripted ShadowBoxers for both bots")
    ap.add_argument("--no-terminate", action="store_true",
                    help="demo: don't stop on fall/KO (full-length clip)")
    # --- tunable camera (tight broadcast close-up default) ---
    ap.add_argument("--cam_az", type=float, default=90.0,
                    help="azimuth deg (90 = side-on, bots in profile)")
    ap.add_argument("--cam_el", type=float, default=10.0,
                    help="elevation deg (+up)")
    ap.add_argument("--cam_dist", type=float, default=2.2,
                    help="camera distance (m) -- 2.2 = fighters fill frame")
    ap.add_argument("--cam_lookat", default="-0.15 0 0.95",
                    help="lookat 'x y z' (ring center, chest height)")
    a = ap.parse_args()

    env = G1FighterEnv(balance_path=a.balance, opponent_path=a.p2,
                       max_steps=a.steps, randomize=False, demo=a.demo)
    judge = BoxingJudge(env, round_seconds=a.max_round_seconds,
                        rounds=a.rounds)

    if a.demo or not a.p1:
        p1 = ShadowBoxer(env, style="red")
    else:
        p1 = PPO.load(a.p1)
    # r2: demo -> shadowboxer(blue); else opponent_path (env drives it)
    if a.demo or (a.p2 is None and not a.p1):
        # when demo, drive r2 via opponent_path hook using a ShadowBoxer
        env.opponent = ShadowBoxer(env, style="blue")

    cam = _make_camera(a)
    rend = mujoco.Renderer(env.model, height=540, width=960)

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < a.steps:
        a1, _ = p1.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)
        rend.update_scene(env.data, camera=cam)
        frames.append(rend.render())
        done = (not a.no_terminate) and (
            term or trunc or judge.ko or (judge.winner is not None))
        t += 1

    if frames:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        # imageio (mp4 via ffmpeg) finalizes containers reliably;
        # cv2 mp4v occasionally drops the moov atom on early stop.
        try:
            import imageio.v2 as imageio
            imageio.imsave(a.out, [f[..., ::-1] for f in frames], fps=30)
        except Exception as e:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(a.out, fourcc, 30, (960, 540))
            for f in frames:
                vw.write(np.ascontiguousarray(f[..., ::-1]))
            vw.release()
        print(f"[saved] {a.out} ({len(frames)} frames)")

    card = judge.card()
    print(f"[bout] CARD: winner={card['winner']} method={card['method']} "
          f"hp={card['final_hp']} rounds={card['round_scores']}")


if __name__ == "__main__":
    main()
