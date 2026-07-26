"""Eval a Track B fighter policy.

Checks the two things that matter (RoboStriker metrics):
  1. STANDS -- pelvis stays > 0.4 for 30s (1500 steps) vs a
     frozen StandPD opponent (sandbag warmup stage).
  2. LANDS HITS -- eta_hit (hit landing rate) + ER (engagement
     rate) over N episodes, using the anti-shove damage rule.

Usage:
  python3 eval_fighter.py --policy models/fighter_v1 \\
      --balance models/balance_v1 --episodes 10
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from g1_fighter_env import G1FighterEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--balance", required=True)
    ap.add_argument("--opponent", default="")   # leave "" = sandbag (StandPD)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=1500)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    pol = PPO.load(a.policy)
    env = G1FighterEnv(balance_path=a.balance,
                       opponent_path=a.opponent or None,
                       max_steps=a.max_steps, randomize=False)

    stands = 0
    strikes = 0          # successful strikes (dmg dealt, RoboStriker: F>10N)
    attempts = 0         # offensive attempts (punches thrown, arm residual large)
    engagements = 0
    dmg_dealt_total = 0.0
    for ep in range(a.episodes):
        o, _ = env.reset()
        fell = False
        ep_strikes = 0
        ep_attempts = 0
        engaged = False
        for _ in range(a.max_steps):
            act, _ = pol.predict(o, deterministic=True)
            arm = act[:14]
            if np.linalg.norm(arm) > 0.3:      # bot is throwing a punch
                ep_attempts += 1
            o, r, term, trunc, info = env.step(act)
            # count a strike when opponent HP dropped by >=0.5 this step
            if (100.0 - env.hp[1]) > ep_strikes + 0.5:
                ep_strikes += 1
                engaged = True
            if term:
                fell = True
                break
            if trunc:
                break
        if not fell:
            stands += 1
        if engaged:
            engagements += 1
        strikes += ep_strikes
        attempts += max(ep_attempts, ep_strikes)
        dmg_dealt_total += (100.0 - env.hp[1])
        print(f"ep{ep}: stood={'Y' if not fell else 'N'} "
              f"oppHP={env.hp[1]:.0f} strikes={ep_strikes} attempts={ep_attempts}")

    eta_hit = strikes / max(attempts, 1)
    er = engagements / a.episodes
    print(f"\n=== EVAL ({a.episodes} eps) ===")
    print(f"stand_rate : {stands}/{a.episodes} = {stands/a.episodes:.2f}")
    print(f"ER (engage): {er:.2f}")
    print(f"avg dmg    : {dmg_dealt_total/a.episodes:.1f}")
    print(f"eta_hit    : {eta_hit:.2f}  (strikes/attempts = {strikes}/{attempts})")
    print("PASS stand" if stands >= a.episodes * 0.8 else "WEAK stand")


if __name__ == "__main__":
    main()
