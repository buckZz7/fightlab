"""CI gate for fighter submissions.

Evaluates a submitted fighter in the league and decides if it
passes the merge threshold. Used by Gittensor PR CI.

Criteria:
  - Must not crash (all bouts complete)
  - Must win at least 1 bout (not a sandbag)
  - ELO >= 1400 (beats the PD baseline)
  - Must not time out (bouts finish in reasonable time)

Usage:
  python3 ci_gate.py --fighter models/fighter_v2.zip
  python3 ci_gate.py --fighter models/fighter_v2.zip --threshold 1500
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import subprocess

DEFAULT_ENTRANTS = [
    "scripted:jabbler", "scripted:defender",
    "scripted:balanced", "scripted:pd"
]
MIN_ELO = 1400
MIN_WINS = 1


def run_league(fighter_path, standings_path, bouts=3, max_steps=5000):
    """Run the league with the submitted fighter + scripted baselines."""
    entrants = [fighter_path] + DEFAULT_ENTRANTS
    cmd = [sys.executable, "league.py",
           "--entrants", *entrants,
           "--pd", "--bouts", str(bouts),
           "--max_steps", str(max_steps),
           "--round_seconds", "20",
           "--out", standings_path]
    print(f"[ci] running league: {fighter_path}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    print(f"[ci] league completed in {elapsed:.0f}s")
    if r.returncode != 0:
        print(f"[ci] league FAILED: {r.stderr[-300:]}")
        return None
    return json.load(open(standings_path))


def evaluate(fighter_path, threshold=MIN_ELO):
    """Run CI gate on a submitted fighter. Returns pass/fail + details."""
    standings_path = "/tmp/ci_standings.json"
    standings = run_league(fighter_path, standings_path)
    if standings is None:
        return {"pass": False, "reason": "league crashed"}

    # Find the submitted fighter in standings
    fighter_name = os.path.basename(fighter_path).replace(".zip", "")
    fighter_entry = None
    for s in standings.get("standings", []):
        if fighter_name in s["name"]:
            fighter_entry = s
            break

    if not fighter_entry:
        return {"pass": False, "reason": "fighter not in standings"}

    elo = fighter_entry["elo"]
    wins = fighter_entry["W"]
    losses = fighter_entry["L"]
    draws = fighter_entry["D"]

    result = {
        "pass": True,
        "fighter": fighter_name,
        "elo": elo,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "king": standings.get("king"),
        "threshold": threshold,
    }

    # Gate 1: ELO threshold
    if elo < threshold:
        result["pass"] = False
        result["reason"] = f"ELO {elo} < threshold {threshold}"

    # Gate 2: Must win at least 1 bout
    if wins < MIN_WINS:
        result["pass"] = False
        result["reason"] = f"only {wins} wins (need >= {MIN_WINS})"

    # Gate 3: Must be king or top-3
    if elo < standings["standings"][0]["elo"] and elo < standings["standings"][2]["elo"] if len(standings["standings"]) > 2 else False:
        result["pass"] = False
        result["reason"] = f"ELO {elo} too low (not top-3)"

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fighter", required=True, help="path to fighter .zip")
    ap.add_argument("--threshold", type=float, default=MIN_ELO,
                    help=f"minimum ELO to pass (default {MIN_ELO})")
    a = ap.parse_args()

    result = evaluate(a.fighter, a.threshold)

    print("=" * 50)
    if result["pass"]:
        print(f"✅ PASS: {result['fighter']}")
        print(f"   ELO: {result['elo']:.1f} (threshold: {result['threshold']})")
        print(f"   Record: {result['wins']}W {result['losses']}L {result['draws']}D")
        print(f"   King: {result['king']}")
        sys.exit(0)
    else:
        print(f"❌ FAIL: {result.get('fighter', 'unknown')}")
        print(f"   Reason: {result['reason']}")
        if "elo" in result:
            print(f"   ELO: {result['elo']:.1f} (threshold: {result['threshold']})")
            print(f"   Record: {result['wins']}W {result['losses']}L {result['draws']}D")
        sys.exit(1)


if __name__ == "__main__":
    main()
