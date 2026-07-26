import os
os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np
import mujoco
import sys
sys.path.insert(0, ".")
from g1_arena import build_arena
from g1_balance_env import G1BalanceEnv
from g1_fighter_env import G1FighterEnv
from stable_baselines3 import PPO

bal = PPO.load("models/balance_v1")

# 1) balance env: how long does it stand?
e = G1BalanceEnv(max_steps=300, randomize=False)
o, _ = e.reset()
zs = []
for i in range(300):
    a, _ = bal.predict(o, deterministic=True)
    o, r, term, trunc, info = e.step(a)
    zs.append(float(e.data.qpos[2]))
    if term:
        print(f"[balance_env] fell at {i+1}, z={zs[-1]:.3f}")
        break
else:
    print(f"[balance_env] stood 300, minz={min(zs):.3f}")
print("[balance_env] native[:5] used as base; HOME[:5]=", __import__("loco_base29").HOME[:5])

# 2) fighter env: trajectory of r1 pelvis
fe = G1FighterEnv(balance_path="models/balance_v1", opponent_path=None,
                  max_steps=300, randomize=False)
fo, _ = fe.reset()
fz = []
for i in range(300):
    fa = np.zeros(fe.action_space.shape[0])
    fo, r, term, trunc, info = fe.step(fa)
    fz.append(info["pelvis_z_0"])
    if term:
        print(f"[fighter_env] r1 fell at {i+1}, z={fz[-1]:.3f}")
        break
else:
    print(f"[fighter_env] r1 stood 300, minz={min(fz):.3f}")

# compare native vs HOME
print("fighter native[:5]=", fe.native[:5])
print("balance native[:5]=", e.native[0][7:12] if hasattr(e,'native') else 'n/a')
