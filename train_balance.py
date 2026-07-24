"""Train the balance baseline: PPO on BalanceEnv.

Usage: python train_balance.py --timesteps 1000000 --out models/balance_ppo
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs import make_env

N_ENVS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="models/balance_ppo")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def make():
        return make_env()

    env = SubprocVecEnv([make for _ in range(args.envs)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=8192,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu",
    )

    ckpt = CheckpointCallback(save_freq=200_000 // args.envs, save_path=args.out + "_ckpt",
                              name_prefix="balance")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
