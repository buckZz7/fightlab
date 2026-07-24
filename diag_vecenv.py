"""Diagnostic: find max working worker count for SubprocVecEnv."""
import sys
from stable_baselines3.common.vec_env import SubprocVecEnv
from g1_punch_env import make_g1_punch_env
import numpy as np

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    env = SubprocVecEnv([make_g1_punch_env for _ in range(n)], start_method="spawn")
    obs = env.reset()
    for i in range(20):
        obs, r, d, info = env.step(np.stack([env.action_space.sample() * 0.3 for _ in range(n)]))
    print(f"{n}-env spawn OK, 20 vector-steps done")
    env.close()
