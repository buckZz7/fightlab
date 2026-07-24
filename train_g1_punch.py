"""Train the G1 punch policy: PPO on G1PunchEnv (frozen balance base +
arm residuals vs heavy bag).

Usage:
  python train_g1_punch.py --timesteps 2000000 --out models/g1_punch_ppo
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from g1_punch_env import make_g1_punch_env
from threaded_vecenv import make_threaded_vec_env

N_ENVS = 8   # G1 sim + ONNX policy per env is heavier than the toy humanoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--out", default="models/g1_punch_ppo")
    ap.add_argument("--envs", type=int, default=N_ENVS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = make_threaded_vec_env(make_g1_punch_env, args.envs)
    env = VecMonitor(env)
    eval_env = make_g1_punch_env(randomize=False)

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1),
        save_path=args.out + "_ckpt",
        name_prefix="g1_punch",
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
        n_steps=1024,
        batch_size=512,
        gae_lambda=0.9,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.005,          # keep exploration up early (sparse hit reward)
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(
            log_std_init=-1.5,   # small initial exploration -> residuals start gentle
            net_arch=[256, 256],
        ),
    )
    model.learn(total_timesteps=args.timesteps, callback=[ckpt, evaluator])
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
