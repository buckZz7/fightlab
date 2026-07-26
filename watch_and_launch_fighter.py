"""Watcher: when balance_v1 training finishes, verify it STANDS
(>750 steps), then auto-launch the fighter training (train_fighter.py).

Polls for models/balance_v1.zip. On appearance:
  1. Stand test (750 steps) -- if it fails, alert and STOP (broken substrate).
  2. If pass, launch train_fighter.py (2M steps) in background.
Run on pod: setsid python3 watch_and_launch_fighter.py > watch_fighter.log 2>&1 &
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO
from g1_balance_env import G1BalanceEnv

BALANCE = "models/balance_v1"
STAND_GOAL = 750


def balance_stands():
    model = PPO.load(BALANCE)
    e = G1BalanceEnv(max_steps=STAND_GOAL, randomize=False)
    o, _ = e.reset()
    survived = 0
    for i in range(STAND_GOAL):
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, info = e.step(a)
        if term:
            break
        survived = i + 1
    print(f"[watch] balance stood {survived}/{STAND_GOAL} steps (z={float(e.data.qpos[2]):.3f})")
    return survived >= STAND_GOAL


def main():
    print("[watch] polling for", BALANCE + ".zip")
    while not os.path.exists(BALANCE + ".zip"):
        time.sleep(30)
    print("[watch] balance model appeared; verifying stand...")
    if not balance_stands():
        print("[watch] FAILED stand test -- NOT launching fighter. Fix balance.")
        return
    print("[watch] PASS -- launching fighter training")
    env = dict(os.environ)
    subprocess.Popen(
        ["python3", "train_fighter.py", "--balance", BALANCE,
         "--steps", "2_000_000", "--out", "models/fighter_v1"],
        env=env, stdout=open("fighter_train.log", "w"),
        stderr=subprocess.STDOUT)
    print("[watch] fighter training launched (fighter_train.log)")


if __name__ == "__main__":
    main()
