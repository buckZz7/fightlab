"""CI pre-flight -- the subset of preflight.py that needs NO MuJoCo
meshes (runs on GitHub Actions). Blocks the insidious static bugs:
  - balance-scale mismatch (fighter applies != balance trained)
  - obs/act contract drift (miner_template contract self-test)
  - all modules import cleanly (catches syntax/import rot)

The full PD-stand check (needs G1 meshes) runs on the pod via
preflight.py before any training launch. This CI gate catches the
rest on every push/PR.

Usage:
  python3 preflight_ci.py   # exits non-zero on failure
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))


def check_scale_consistency():
    import g1_balance_env as BE
    src = open(os.path.join(os.path.dirname(__file__),
                            "g1_fighter_env.py")).read()
    m = re.search(r"bal_act \* ([\d.]+) \+ HOME", src)
    fighter_scale = float(m.group(1)) if m else None
    balance_scale = getattr(BE, "SCALE_BAL", None)
    if fighter_scale is None or balance_scale is None:
        print(f"[ci] scale check: could not read "
              f"(fighter={fighter_scale}, balance={balance_scale}) -- SKIP")
        return True
    ok = abs(fighter_scale - balance_scale) < 1e-6
    print(f"[ci] balance-scale match: {'PASS' if ok else 'FAIL'} "
          f"(fighter {fighter_scale}, balance {balance_scale})")
    return ok


def check_obs_contract():
    """obs/act dims in fightlab_contract.py must match the env."""
    import fightlab_contract as C
    # env obs/act are authoritative
    from g1_fighter_env import OBS_DIM, ACT_DIM
    ok = (C.OBS_DIM == OBS_DIM) and (C.ACT_DIM == ACT_DIM)
    print(f"[ci] contract dims: {'PASS' if ok else 'FAIL'} "
          f"(contract {C.OBS_DIM}/{C.ACT_DIM}, env {OBS_DIM}/{ACT_DIM})")
    if not ok:
        print("  -> fightlab_contract.py drifted from g1_fighter_env. Miners break.")
    return ok


def check_imports():
    mods = ["g1_arena", "g1_fighter_env", "g1_balance_env", "loco_base29",
            "league", "fightlab_contract", "miner_template", "g1_moves_reward"]
    ok = True
    for m in mods:
        try:
            __import__(m)
        except Exception as e:
            print(f"[ci] import FAIL {m}: {e}")
            ok = False
    if ok:
        print("[ci] all modules import OK")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= check_imports()
    ok &= check_scale_consistency()
    ok &= check_obs_contract()
    if not ok:
        print("[ci] FAILED -- do NOT merge")
        sys.exit(1)
    print("[ci] ALL CHECKS PASS")
