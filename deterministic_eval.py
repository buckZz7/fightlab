"""Deterministic league eval for trustless PR decisions.

Runs bouts with FIXED seed, NO domain randomization, and produces
a verifiable bout log (JSON) with per-step HP, damage events, and
final decision. Anyone can re-run and get identical results.

Usage:
  python3 deterministic_eval.py --fighter models/my_fighter.zip
  python3 deterministic_eval.py --fighter models/my_fighter.zip --king models/fighter_v2.zip
"""
import os, sys, json, hashlib, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
from stable_baselines3 import PPO
from g1_fighter_env import G1FighterEnv
from combat_rules import CombatJudge
from bout_fighter import ShadowBoxer

# Fixed seed for ALL bouts — makes results reproducible
EVAL_SEED = 42
EVAL_STEPS = 5000
ROUND_SECONDS = 20.0
ROUNDS = 3


def model_hash(path):
    """SHA256 hash of a model file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_bout(fighter_path, opponent_spec, seed=EVAL_SEED):
    """Run a single deterministic bout. Returns bout log."""
    env = G1FighterEnv(max_steps=EVAL_STEPS, randomize=False)

    # Force deterministic: no randomization, fixed seed
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    judge = CombatJudge(env, round_seconds=ROUND_SECONDS, rounds=ROUNDS)

    # Load fighter
    if fighter_path and os.path.exists(fighter_path):
        p1 = PPO.load(fighter_path)
    else:
        p1 = ShadowBoxer(env, style="red", profile="balanced")

    # Load opponent
    if opponent_spec and opponent_spec.startswith("scripted:"):
        profile = opponent_spec.split(":")[1]
        env.opponent = ShadowBoxer(env, style="blue", profile=profile)
    elif opponent_spec and os.path.exists(opponent_spec):
        env.opponent = PPO.load(opponent_spec)
    else:
        env.opponent = ShadowBoxer(env, style="blue", profile="pd")

    log = {
        "fighter": os.path.basename(fighter_path or "shadowboxer"),
        "fighter_hash": model_hash(fighter_path) if fighter_path and os.path.exists(fighter_path) else "scripted",
        "opponent": opponent_spec,
        "seed": seed,
        "events": [],
        "final_hp": None,
        "result": None,
    }

    for t in range(EVAL_STEPS):
        a1, _ = p1.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)

        # Log damage events
        if env._dmg_dealt[0] > 0 or env._dmg_taken[0] > 0:
            log["events"].append({
                "step": t,
                "hp": [float(env.hp[0]), float(env.hp[1])],
                "dmg_dealt": float(env._dmg_dealt[0]),
                "dmg_taken": float(env._dmg_taken[0]),
            })

        if term or trunc:
            break

    log["final_hp"] = [float(env.hp[0]), float(env.hp[1])]
    card = judge.card()
    log["result"] = {
        "winner": card["winner"],
        "method": card["method"],
        "round_scores": card["round_scores"],
    }

    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fighter", required=True, help="path to fighter .zip")
    ap.add_argument("--king", default=None, help="path to current king .zip")
    ap.add_argument("--out", default="/tmp/eval_result.json")
    ap.add_argument("--entrants", nargs="*", default=[
        "scripted:jabbler", "scripted:defender", "scripted:balanced", "scripted:pd"])
    a = ap.parse_args()

    print(f"[eval] fighter: {a.fighter}")
    print(f"[eval] hash: {model_hash(a.fighter)}")
    print(f"[eval] seed: {EVAL_SEED} (deterministic)")

    results = []
    all_entrants = [a.fighter] + a.entrants
    if a.king:
        all_entrants.append(a.king)

    # Run fighter vs each opponent
    for opp in a.entrants:
        print(f"[eval] vs {opp}...", end=" ", flush=True)
        log = run_bout(a.fighter, opp)
        results.append(log)
        print(f"hp={log['final_hp']} result={log['result']['method']}")

    # Title bout vs king
    if a.king:
        print(f"[eval] TITLE BOUT vs king...", end=" ", flush=True)
        log = run_bout(a.fighter, a.king)
        log["title_bout"] = True
        results.append(log)
        print(f"hp={log['final_hp']} result={log['result']['method']}")

    # Summary
    wins = sum(1 for r in results if r["result"]["winner"] == 0)
    losses = sum(1 for r in results if r["result"]["winner"] == 1)
    draws = len(results) - wins - losses

    summary = {
        "fighter": os.path.basename(a.fighter),
        "fighter_hash": model_hash(a.fighter),
        "seed": EVAL_SEED,
        "total_bouts": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "pass": wins >= 1,
        "bouts": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'✅' if summary['pass'] else '❌'} Result: {wins}W {losses}L {draws}D")
    print(f"[eval] saved {a.out}")

    if not summary["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
