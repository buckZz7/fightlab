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


def _preflight():
    """Gate: refuse to launch long training on a BROKEN substrate.

    Checks the substrate is physically sane + contract-consistent:
      - shapes sane (obs/act)
      - sim does NOT NaN/crash for a short window (physics stable)
      - balance-scale consistency (fighter must match balance)
    NOTE: we do NOT require PD-to-HOME to stand indefinitely -- a G1
    needs ACTIVE balance (what the policy learns). The real stand gate
    is the trained-policy eval at the END of training.
    """
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("preflight",
        os.path.join(os.path.dirname(__file__), "preflight.py"))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    # sanity: shapes + scale (cheap, high-value)
    ok = True
    ok &= pf.check_shapes()
    ok &= pf.check_scale_consistency()
    # physics-stability: sim must not NaN/crash for N steps (PD may sag,
    # that's fine -- we just need no singular/degenerate physics)
    try:
        e = G1BalanceEnv(max_steps=150, randomize=False)
        e.reset()
        for _ in range(150):
            e.step(np.zeros(e.action_space.shape[0]))
        print("[preflight] physics stable: 150 steps no NaN/crash -- OK")
    except Exception as ex:
        print(f"[preflight] physics UNSTABLE: {ex}")
        ok = False
    if not ok:
        print("[preflight] FAILED -- aborting training")
        sys.exit(1)
    print("[preflight] substrate OK -- proceeding to train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", default="models/balance_v1")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--randomize", action="store_true", default=True)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--tb", default="")
    ap.add_argument("--skip_preflight", action="store_true",
                    help="skip the PD-stand pre-flight gate")
    a = ap.parse_args()

    if not a.skip_preflight:
        _preflight()

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

    # REQUIRED stand test: the balance policy is the foundation for the
    # fighter. A model that can't stand is worse than useless (it poisons
    # the fighter env). Refuse to save a non-standing model.
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
    if survived < 750:
        print(f"[eval] FAIL: balance policy stands only {survived} steps (<750 = 1.5s, "
              f"worse than PD baseline 678). NOT saving a broken substrate. "
              f"Fix training and rerun.")
        # remove the bad model so downstream steps don't use it
        import os as _os
        _os.remove(f"{a.out}.zip")
        sys.exit(1)
    print(f"[eval] PASS: balance policy stands {survived} steps -> substrate OK")


if __name__ == "__main__":
    main()
