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
EVAL_SEEDS = [42, 123, 777]  # multiple seeds to prevent overfitting
EVAL_STEPS = 5000
ROUND_SECONDS = 20.0
ROUNDS = 3
MAX_MODEL_SIZE = 50 * 1024 * 1024  # 50MB limit
MIN_DAMAGE_TO_PASS = 1.0  # must deal at least 1 damage (not just survive)


def model_hash(path):
    """SHA256 hash of a model file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_bout(fighter_path, opponent_spec, seed=42):
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
    print(f"[eval] seeds: {EVAL_SEEDS} (multi-seed anti-overfit)")

    # Model size check (anti-DoS)
    fsize = os.path.getsize(a.fighter)
    if fsize > MAX_MODEL_SIZE:
        print(f"[eval] REJECTED: model too large ({fsize / 1e6:.1f}MB > {MAX_MODEL_SIZE / 1e6:.0f}MB)")
        sys.exit(1)

    results = []
    all_entrants = [a.fighter] + a.entrants
    if a.king:
        all_entrants.append(a.king)

    # Run fighter vs each opponent across ALL seeds
    for seed in EVAL_SEEDS:
        for opp in a.entrants:
            print(f"[eval] seed={seed} vs {opp}...", end=" ", flush=True)
            log = run_bout(a.fighter, opp, seed=seed)
            log["seed"] = seed
            results.append(log)
            print(f"hp={log['final_hp']} result={log['result']['method']}")

    # Title bout vs king (all seeds)
    if a.king:
        for seed in EVAL_SEEDS:
            print(f"[eval] seed={seed} TITLE BOUT vs king...", end=" ", flush=True)
            log = run_bout(a.fighter, a.king, seed=seed)
            log["title_bout"] = True
            log["seed"] = seed
            results.append(log)
            print(f"hp={log['final_hp']} result={log['result']['method']}")

    # Summary with anti-gaming checks
    wins = sum(1 for r in results if r["result"]["winner"] == 0)
    losses = sum(1 for r in results if r["result"]["winner"] == 1)
    draws = len(results) - wins - losses
    # Must deal actual damage (not just survive opponent falling)
    total_dmg_dealt = sum(sum(e["dmg_dealt"] for e in r["events"]) for r in results)
    # Must win on at least 2 different seeds (anti-overfit)
    seeds_won = set(r["seed"] for r in results if r["result"]["winner"] == 0)

    summary = {
        "fighter": os.path.basename(a.fighter),
        "fighter_hash": model_hash(a.fighter),
        "model_size_mb": round(fsize / 1e6, 1),
        "seeds": EVAL_SEEDS,
        "total_bouts": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_damage_dealt": round(total_dmg_dealt, 2),
        "seeds_won_on": sorted(seeds_won),
        # Gate: ≥1 win + dealt damage + won on ≥2 seeds
        "pass": wins >= 1 and total_dmg_dealt >= MIN_DAMAGE_TO_PASS and len(seeds_won) >= 2,
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
