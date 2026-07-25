"""Train G1 self-play boxing: PPO on G1SelfPlayEnv.

Gen 1: train against random opponent (no model loaded).
Gen 2+: train against frozen previous king.

Usage:
  python train_g1_selfplay.py --timesteps 1000000 --out models/boxing_gen1
  python train_g1_selfplay.py --opponent models/boxing_gen1.zip --timesteps 1000000 --out models/boxing_gen2
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from g1_selfplay_env import G1SelfPlayEnv, N_SKILL, OBS_DIM


def make_env(opponent_path=None, opponent_mocap=False, rank=0, seed=0, max_steps=2000):
    def _init():
        from stable_baselines3 import PPO
        opp = PPO.load(opponent_path) if opponent_path else None
        env = G1SelfPlayEnv(opponent_model=opp, opponent_mocap=opponent_mocap,
                            max_steps=max_steps, randomize=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(timesteps, out, opponent_path=None, opponent_mocap=False,
          n_envs=4, max_steps=2000):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Single env if n_envs=1 (DummyVecEnv), else SubprocVecEnv
    if n_envs == 1:
        env = DummyVecEnv([make_env(opponent_path, opponent_mocap, 0, 42, max_steps)])
    else:
        env = SubprocVecEnv([make_env(opponent_path, opponent_mocap, i, 42, max_steps)
                             for i in range(n_envs)], start_method="spawn")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    ckpt = CheckpointCallback(
        save_freq=max(timesteps // 10, 10000),
        save_path=f"{out}_ckpt",
        name_prefix="boxing",
    )

    opp_name = "mocap" if opponent_mocap else (os.path.basename(opponent_path) if opponent_path else "random")
    print(f"Training {timesteps} steps with {n_envs} envs vs {opp_name}")
    model.learn(total_timesteps=timesteps, callback=ckpt)
    model.save(out)
    print(f"Saved {out}")
    env.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="models/boxing_gen1")
    ap.add_argument("--opponent", default=None, help="Frozen opponent model path")
    ap.add_argument("--mocap", action="store_true", help="Use mocap replay opponent (warmup)")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    train(args.timesteps, args.out, args.opponent, args.mocap, args.envs, args.max_steps)
