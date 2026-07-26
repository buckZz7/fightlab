"""Train a stand-still balance policy for G1 (Track B prerequisite).

Usage:
  python3 train_balance.py --steps 1_000_000 --out models/balance_v1
Launches on GPU if available, else CPU. Logs to stdout;
SB3 TensorBoard optional.
"""
import os, sys, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from g1_balance_env import G1BalanceEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", default="models/balance_v1")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--randomize", action="store_true", default=True)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--tb", default="")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    env = G1BalanceEnv(max_steps=a.max_steps, randomize=a.randomize)

    # env sanity
    try:
        check_env(env, warn=True)
        print("[ok] env check passed")
    except Exception as e:
        print("[warn] env check:", e)

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=a.tb or None,
        device="auto",
    )
    print(f"[train] {a.steps:,} steps, {a.n_envs} envs, out={a.out}")
    model.learn(total_timesteps=a.steps, tb_log_name="balance")
    model.save(a.out)
    print(f"[saved] {a.out}.zip")

    # quick stand test
    test = G1BalanceEnv(max_steps=1500, randomize=False)
    o, _ = test.reset()
    survived = 0
    for i in range(1500):
        a2, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, info = test.step(a2)
        if term:
            break
        survived = i + 1
    print(f"[eval] stood {survived}/1500 steps ({(survived/1500*100):.0f}s) z={float(test.data.qpos[2]):.3f}")


if __name__ == "__main__":
    main()
