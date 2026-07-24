"""Train the G1 punch policy: PPO on G1PunchEnv (frozen balance base +
arm residuals vs heavy bag).

Usage:
  python train_g1_punch.py --timesteps 2000000 --out models/g1_punch_ppo
"""
import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from g1_punch_env import make_g1_punch_env
from threaded_vecenv import make_threaded_vec_env

# Torch defaults to all cores for its op thread pools and starves the sim
# threads (170x collapse observed: 2138 -> 12 steps/s). Cap torch to 2 so
# the 8 env threads own the CPU.
torch.set_num_threads(2)

N_ENVS = 8   # G1 sim + ONNX policy per env is heavier than the toy humanoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="models/g1_punch_ppo")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from an existing checkpoint .zip")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = make_threaded_vec_env(make_g1_punch_env, args.envs)
    env = VecMonitor(env)
    eval_env = make_g1_punch_env(randomize=False)

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1),
        save_path=args.out + "_ckpt",
        name_prefix="g1_punch",
    )

    if args.init_from:
        model = PPO.load(args.init_from, env=env)
        print(f"warm-starting from {args.init_from}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=256,             # shorter rollouts: faster update cycles on CPU
            batch_size=256,
            gae_lambda=0.9,
            gamma=0.99,
            n_epochs=4,              # fewer gradient passes per rollout
            ent_coef=0.005,          # keep exploration up early (sparse hit reward)
            learning_rate=3e-4,
            clip_range=0.2,
            policy_kwargs=dict(
                log_std_init=-1.5,   # small initial exploration -> residuals start gentle
                net_arch=[256, 256],
            ),
        )
    model.learn(total_timesteps=args.timesteps, callback=[ckpt])
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
