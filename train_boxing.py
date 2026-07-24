"""Train the boxing challenger: PPO in SelfPlayEnv against a frozen opponent.

Usage:
  # Stage 2: vs random/frozen bag policy
  python train_boxing.py --timesteps 3000000 --out models/boxing_gen1

  # Stage 3: vs frozen previous king
  python train_boxing.py --opponent models/boxing_gen1.zip --out models/boxing_gen2
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from selfplay_env import make_selfplay_env

N_ENVS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--out", default="models/boxing_gen1")
    ap.add_argument("--opponent", default=None,
                    help="path to frozen opponent .zip; omit for random")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def make():
        return make_selfplay_env(opponent_path=args.opponent)

    env = SubprocVecEnv([make for _ in range(args.envs)])
    env = VecMonitor(env)

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1),
        save_path=args.out + "_ckpt",
        name_prefix="boxing",
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
    model.learn(total_timesteps=args.timesteps, callback=[ckpt])
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
