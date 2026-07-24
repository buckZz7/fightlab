"""Train the punch policy: PPO on PunchEnv (bag striking + domain randomization).

Milestone 2 of the fight league: stay upright, absorb pushes, hit the bag.
Trains from scratch (PunchEnv's stripped obs layout differs from BalanceEnv).

Usage:
  python train_punch.py --timesteps 3000000 --out models/punch_ppo
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from punch_env import make_punch_env

N_ENVS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--out", default="models/punch_ppo")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = SubprocVecEnv([make_punch_env for _ in range(args.envs)])
    env = VecMonitor(env)
    eval_env = make_punch_env()

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1),
        save_path=args.out + "_ckpt",
        name_prefix="punch",
    )
    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=args.out + "_best",
        eval_freq=max(50_000 // args.envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        gae_lambda=0.9,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.001,
        learning_rate=3e-4,
        clip_range=0.2,
    )
    model.learn(total_timesteps=args.timesteps, callback=[ckpt, evaluator])
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
