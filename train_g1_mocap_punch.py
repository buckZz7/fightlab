"""Train the G1 mocap-imitation punch policy (DeepMimic-style).

PPO on G1MocapPunchEnv: track the reference punch, adapt for bag damage,
stay upright. Designed to run on a RunPod GPU pod.

Usage:
  python train_g1_mocap_punch.py --timesteps 800000 --out models/g1_mocap_punch --envs 16
"""
import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from g1_mocap_punch_env import make_g1_mocap_punch_env
from threaded_vecenv import make_threaded_vec_env

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=800_000)
    ap.add_argument("--out", default="models/g1_mocap_punch")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--mocap", default="mocap/kungfu_retargeted/Horse-stance_punch.pkl")
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def make():
        return make_g1_mocap_punch_env(mocap_path=args.mocap)

    env = make_threaded_vec_env(make, args.envs)
    env = VecMonitor(env)

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1),
        save_path=args.out + "_ckpt",
        name_prefix="g1_mocap_punch",
    )

    if args.init_from:
        model = PPO.load(args.init_from, env=env)
        print(f"warm-starting from {args.init_from}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            n_steps=256, batch_size=256,
            gae_lambda=0.9, gamma=0.99, n_epochs=4,
            ent_coef=0.005, learning_rate=3e-4, clip_range=0.2,
            policy_kwargs=dict(log_std_init=-1.5, net_arch=[256, 256]),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    model.learn(total_timesteps=args.timesteps, callback=[ckpt])
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
