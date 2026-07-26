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
    hits = 0
    engagements = 0
    dmg_dealt_total = 0.0
    for ep in range(a.episodes):
        o, _ = env.reset()
        fell = False
        ep_hits = 0
        engaged = False
        for _ in range(a.max_steps):
            act, _ = pol.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(act)
            dmg = 100.0 - info["hp_1"]           # opp hp lost this ep
            if dmg > 0.5:
                engaged = True
                ep_hits += 1
            if term:
                fell = True
                break
            if trunc:
                break
        if not fell:
            stands += 1
        if engaged:
            engagements += 1
        dmg_dealt_total += (100.0 - env.hp[1])
        print(f"ep{ep}: stood={'Y' if not fell else 'N'} "
              f"oppHP={env.hp[1]:.0f} dmg_dealt={100-env.hp[1]:.1f}")

    print(f"\n=== EVAL ({a.episodes} eps) ===")
    print(f"stand_rate : {stands}/{a.episodes} = {stands/a.episodes:.2f}")
    print(f"ER (engage): {engagements}/{a.episodes} = {engagements/a.episodes:.2f}")
    print(f"avg dmg    : {dmg_dealt_total/a.episodes:.1f}")
    print(f"eta_hit    : {hits/a.episodes:.2f}  (landing rate, hits/eps)")
    print("PASS stand" if stands >= a.episodes * 0.8 else "WEAK stand")


if __name__ == "__main__":
    main()
