import numpy as np
from g1_balance_env import G1BalanceEnv

for scale in [0.10, 0.40]:
    e = G1BalanceEnv(max_steps=1500, randomize=False)
    # monkeypatch scale
    import g1_balance_env as E
    E.SCALE_BAL = scale
    minz = 9.0
    first_drop = None
    o, _ = e.reset()
    for i in range(1500):
        o, r, term, trunc, info = e.step(np.zeros(29))
        minz = min(minz, info["pelvis_z"])
        if info["pelvis_z"] < 0.5 and first_drop is None:
            first_drop = i + 1
        if term:
            break
    print(f"SCALE_BAL={scale}: zeros target=native; fell@{first_drop} minz={minz:.3f} end={i+1}")
