"""Train the Track B FIGHTER: balance substrate + punches.

Loads models/balance_v1.zip (the stand-still policy) as a FROZEN
substrate, then trains a 17-dim policy (arm residual 14 + walk
cmd 3) with:
  - damage reward (anti-shove)  -> learn to HIT
  - motion-match bonus (G1 Moves clips) -> learn CLEAN punches
  - facing + approach + balance penalty

Usage:
  python3 train_fighter.py --balance models/balance_v1 \\
      --steps 2_000_000 --out models/fighter_v1
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from g1_fighter_env import G1FighterEnv


def _preflight():
    """Gate: refuse to launch the 2M-step fighter run if the frozen
    balance substrate can't stand inside G1FighterEnv. Catches the
    SCALE_BAL / DR / model-quality bugs in ~20s, not after an hour.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight_fighter",
        os.path.join(os.path.dirname(__file__), "preflight_fighter.py"))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    ok = pf.check_fighter_stands()
    if not ok:
        print("[preflight] FAILED -- aborting fighter training "
              "(fix frozen substrate first)")
        sys.exit(1)
    print("[preflight] fighter substrate OK -- proceeding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", default="models/balance_v1")
    ap.add_argument("--opponent", default="")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--out", default="models/fighter_v1")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--randomize", action="store_true", default=True)
    ap.add_argument("--tb", default="")
    ap.add_argument("--skip_preflight", action="store_true",
                    help="skip the fighter-stand pre-flight gate")
    a = ap.parse_args()

    if not a.skip_preflight:
        _preflight()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Parallelize rollouts across CPU cores (MuJoCo sim is CPU-bound).
    from stable_baselines3.common.vec_env import SubprocVecEnv
    def _make():
        opp = a.opponent if a.opponent else None
        return G1FighterEnv(balance_path=a.balance, opponent_path=opp,
                             max_steps=a.max_steps, randomize=a.randomize)
    if a.n_envs > 1:
        env = SubprocVecEnv([_make for _ in range(a.n_envs)])
    else:
        env = _make()

    # env sanity
    try:
        check_env(G1FighterEnv(balance_path=a.balance,
                               opponent_path=(a.opponent or None),
                               max_steps=a.max_steps, randomize=a.randomize),
                  warn=True)
        print("[ok] env check passed")
    except Exception as e:
        print("[warn] env check:", e)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=512,
        n_epochs=12,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=a.tb or None,
        device="auto",
    )
    print(f"[train] balance={a.balance} steps={a.steps:,} out={a.out}")
    model.learn(total_timesteps=a.steps, tb_log_name="fighter")
    model.save(a.out)
    print(f"[saved] {a.out}.zip")


if __name__ == "__main__":
    main()
