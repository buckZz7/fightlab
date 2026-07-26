"""Fighter-env pre-flight: prove G1FighterEnv can STAND + STEP with the
trained balance substrate BEFORE launching the 2M-step fighter run.

This is the "find fixes earlier" gate for the fighter: the balance
model is trained, but the fighter env applies it with a residual scale
+ DR wrapper that we just debugged. A 2M-step run that silently falls
in 0.5s = another wasted hour. This catches it in ~20s.

Run on pod once models/balance_v1.zip exists:
  python3 preflight_fighter.py
Exits non-zero if the fighter env can't stand on the frozen substrate.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

BALANCE = "models/balance_v1.zip"
STAND_STEPS = 200      # ~4s -- fighter env w/ frozen substrate should hold
FALL_Z = 0.40


def check_fighter_stands():
    from g1_fighter_env import G1FighterEnv

    balance = BALANCE if os.path.exists(BALANCE) else None
    mode = "frozen-balance-model" if balance else "PD-substrate (balance=None)"
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=STAND_STEPS, randomize=False)
    o, _ = env.reset()
    assert o.shape == (85,), f"fighter obs shape {o.shape} != 85"
    minz = 9.0
    fell = None
    for i in range(STAND_STEPS):
        # stand still: zero action (no arm residual, no walk)
        a = np.zeros(env.action_space.shape[0])
        o, r, term, trunc, info = env.step(a)
        z = info["pelvis_z_0"]
        minz = min(minz, z)
        if term or z < FALL_Z:
            fell = i + 1
            break
    ok = fell is None
    print(f"[fighter-preflight] {mode} stand: "
          f"{'PASS' if ok else 'FAIL'} "
          f"(held {STAND_STEPS if ok else fell} steps, minz={minz:.3f})")
    if not ok:
        print("  -> fighter env falls on the balance substrate. "
              "Check SCALE_BAL match / DR / model quality before training.")
    return ok


if __name__ == "__main__":
    ok = check_fighter_stands()
    if not ok:
        print("[fighter-preflight] FAILED -- do NOT launch train_fighter.py")
        sys.exit(1)
    print("[fighter-preflight] PASS -- safe to train fighter")
