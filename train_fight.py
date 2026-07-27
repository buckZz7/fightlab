"""Train a combat policy on the walker-based FightEnv.

Phase 1: Sandbag warmup (passive opponent)
Phase 2: Self-play (fight copies of itself)

Uses RoboStriker's approach: warmup first, then competitive training.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from fight_env import FightEnv


def make_env(opponent, max_steps, seed):
    def _init():
        env = FightEnv(max_steps=max_steps, opponent=opponent)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--out", default="models/fighter_walker")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--opponent", default="sandbag",
                    help="sandbag, scripted:jabbler, scripted:defender, or model path")
    ap.add_argument("--max-steps", type=int, default=3000, help="bout length during training")
    args = ap.parse_args()

    print(f"[train] opponent={args.opponent}, steps={args.steps}, envs={args.envs}")

    env = SubprocVecEnv([make_env(args.opponent, args.max_steps, i) for i in range(args.envs)])

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                device="cpu",  # MLP policy runs faster on CPU
                verbose=1)

    print(f"[train] training {args.steps} steps...")
    model.learn(total_timesteps=args.steps)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"[train] saved {args.out}")


if __name__ == "__main__":
    main()
