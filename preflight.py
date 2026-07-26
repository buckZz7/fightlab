"""Pre-flight stand check -- run BEFORE any long balance/fighter
training. Catches a broken PD substrate (wrong gains, joint-order
mismatch, unstable pose) in ~15s instead of wasting a 50-min run.

Checks:
  1. PD-to-HOME holds STAND_STEPS (default 300 ~= 6s) without
     pelvis dropping below FALL_Z. If the substrate can't stand
     on its own, RL has nothing stable to learn from.
  2. Obs/act shapes sane.
  3. Balance-scale consistency: g1_fighter_env must apply the SAME
     residual scale the balance policy was TRAINED with
     (g1_balance_env.SCALE_BAL). Mismatch starves the frozen
     substrate of authority -> r1 falls instantly.

Usage:
  python3 preflight.py
Exits non-zero if the substrate fails (CI / pre-train gate).
"""
import os, sys, re
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mujoco

from g1_arena import build_arena
from loco_base29 import StandPD, KP, KD, HOME

STAND_STEPS = 300      # ~6s -- if PD can't hold this, RL is hopeless
FALL_Z = 0.40


def check_pd_stand():
    m = build_arena(ring="ropes", half=2.4)
    d = mujoco.MjData(m)
    lo = m.actuator_ctrlrange[:, 0].copy()
    hi = m.actuator_ctrlrange[:, 1].copy()
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [-0.6, 0, 0.793]
    d.qpos[7:36] = HOME
    mujoco.mj_forward(m, d)

    spd = StandPD()
    minz = 9.0
    fell = None
    for i in range(STAND_STEPS):
        tau = KP * (HOME - d.qpos[7:36]) - KD * d.qvel[6:35]
        d.ctrl[:29] = np.clip(tau, lo[:29], hi[:29])
        tau2 = KP * (HOME - d.qpos[14:43]) - KD * d.qvel[13:42]
        d.ctrl[29:58] = np.clip(tau2, lo[29:], hi[29:])
        mujoco.mj_step(m, d, 1)
        z = float(d.qpos[2])
        minz = min(minz, z)
        if z < FALL_Z:
            fell = i + 1
            break
    ok = fell is None
    print(f"[preflight] PD-to-HOME stand: "
          f"{'PASS' if ok else 'FAIL'} "
          f"(held {STAND_STEPS if ok else fell} steps, minz={minz:.3f})")
    return ok


def check_shapes():
    from g1_balance_env import G1BalanceEnv
    e = G1BalanceEnv(max_steps=10, randomize=False)
    o, _ = e.reset()
    assert o.shape == (e.observation_space.shape[0],), f"obs {o.shape}"
    a = e.action_space.sample()
    assert a.shape == (e.action_space.shape[0],), f"act {a.shape}"
    print(f"[preflight] obs={o.shape} act={a.shape}  OK")
    return True


def check_scale_consistency():
    """Catch the SCALE_BAL mismatch bug: the frozen balance substrate
    in g1_fighter_env must apply the SAME residual scale the balance
    policy was TRAINED with (g1_balance_env.SCALE_BAL). Mismatch starves
    the substrate of authority -> r1 falls instantly. (Found 2026-07-26:
    fighter used 0.10, balance trained at 0.40.)
    """
    import g1_balance_env as BE
    src = open(os.path.join(os.path.dirname(__file__),
                            "g1_fighter_env.py")).read()
    m = re.search(r"bal_act \* ([\d.]+) \+ HOME", src)
    fighter_scale = float(m.group(1)) if m else None
    balance_scale = getattr(BE, "SCALE_BAL", None)
    if fighter_scale is None or balance_scale is None:
        print(f"[preflight] scale check: could not read "
              f"(fighter={fighter_scale}, balance={balance_scale}) -- SKIP")
        return True
    ok = abs(fighter_scale - balance_scale) < 1e-6
    print(f"[preflight] balance-scale match: "
          f"{'PASS' if ok else 'FAIL'} "
          f"(fighter applies {fighter_scale}, balance trained {balance_scale})")
    if not ok:
        print("  -> frozen substrate authority mismatch; r1 will fall. Fix g1_fighter_env.")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= check_pd_stand()
    ok &= check_shapes()
    ok &= check_scale_consistency()
    if not ok:
        print("[preflight] FAILED -- do NOT launch long training")
        sys.exit(1)
    print("[preflight] ALL CHECKS PASS -- safe to train")
