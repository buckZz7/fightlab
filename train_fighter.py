"""Train the Track B FIGHTER: balance substrate + punches.

Loads models/balance_v1.zip (the stand-still policy) as a FROZEN
substrate, then trains a 17-dim policy (arm residual 14 + walk
cmd 3) with:
  - damage reward (anti-shove)  -> learn to HIT
  - motion-match bonus (G1 Moves clips) -> learn CLEAN punches
  - facing + approach + balance penalty

Usage:
  python3 train_fighter.py --balance models/balance_v1 \\
      --steps 2_000_000 --out models/fighter_v1
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from g1_fighter_env import G1FighterEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", default="models/balance_v1")
    ap.add_argument("--opponent", default="")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--out", default="models/fighter_v1")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--randomize", action="store_true", default=True)
    ap.add_argument("--tb", default="")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    opp = a.opponent if a.opponent else None
    env = G1FighterEnv(
        balance_path=a.balance, opponent_path=opp,
        max_steps=a.max_steps, randomize=a.randomize)

    try:
        check_env(env, warn=True)
        print("[ok] env check passed")
    except Exception as e:
        print("[warn] env check:", e)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=512,
        n_epochs=12,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=a.tb or None,
        device="auto",
    )
    print(f"[train] balance={a.balance} steps={a.steps:,} out={a.out}")
    model.learn(total_timesteps=a.steps, tb_log_name="fighter")
    model.save(a.out)
    print(f"[saved] {a.out}.zip")


if __name__ == "__main__":
    main()
