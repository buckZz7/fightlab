import numpy as np
from g1_balance_env import G1BalanceEnv
from stable_baselines3 import PPO

e = G1BalanceEnv(max_steps=1500, randomize=False)
pol = PPO.load("models/balance_v1")
minz = 9.0
fell = None
o, _ = e.reset()
for i in range(1500):
    a, _ = pol.predict(o, deterministic=True)
    o, r, term, trunc, info = e.step(a)
    minz = min(minz, info["pelvis_z"])
    if term:
        fell = i + 1
        break
label = "FULL 1500" if fell is None else str(fell)
secs = (fell or 1500) * 0.02
print(f"balance_v1: stood {label} steps (~{secs:.1f}s); min pelvis z={minz:.3f}")
print("STAND TEST:", "PASS" if (fell is None or fell > 1400) else "FAIL")
