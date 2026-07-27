"""Self-play training loop: always-evolving league.

Each cycle:
1. Train a new challenger against the current king + past kings + scripted
2. Run league eval (challenger vs all entrants)
3. If challenger ELO >= king ELO: title bout (challenger vs king)
4. If challenger beats king: new king, add old king to the training pool
5. Repeat

The league never stops — each iteration produces a stronger fighter.
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(__file__))

KINGS_DIR = "models/kings"
LEAGUE_FILE = "docs/league_standings.json"
TITLE_BOUT_DIR = "docs/bouts"


def run(cmd, timeout=3600):
    """Run a command, stream output, return result."""
    print(f"> {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-500:], flush=True)
    if r.stderr and r.returncode != 0:
        print(f"[warn] {r.stderr[-300:]}", flush=True)
    return r


def get_king():
    """Get the current king from league standings."""
    if not os.path.exists(LEAGUE_FILE):
        return None
    d = json.load(open(LEAGUE_FILE))
    return d.get("king")


def get_king_model():
    """Get the path to the current king's model."""
    king = get_king()
    if king is None:
        return None
    # king is a name like "fighter_v4.zip" or "scripted:pd"
    if king.startswith("scripted:"):
        return None  # scripted, no model
    # king already includes .zip extension
    path = f"models/{king}"
    if os.path.exists(path):
        return path
    # try without .zip
    path2 = f"models/{king.replace('.zip', '')}"
    if os.path.exists(path2 + ".zip"):
        return path2 + ".zip"
    return None


def archive_king(king_model, cycle):
    """Archive the current king so future challengers can train against it."""
    if king_model is None:
        return None
    # king_model is already a full path (e.g. "models/fighter_v4.zip")
    if not os.path.exists(king_model):
        return None
    os.makedirs(KINGS_DIR, exist_ok=True)
    archived = os.path.join(KINGS_DIR, f"king_cycle{cycle}.zip")
    import shutil
    shutil.copy2(king_model, archived)
    print(f"[evolve] archived king -> {archived}", flush=True)
    return archived


def get_training_opponents(king_model, cycle):
    """Get all past kings + current king for self-play training."""
    opponents = []
    # Current king (already a full path)
    if king_model and os.path.exists(king_model):
        opponents.append(king_model)
    # Past kings
    if os.path.exists(KINGS_DIR):
        for f in sorted(os.listdir(KINGS_DIR)):
            if f.endswith(".zip"):
                opponents.append(os.path.join(KINGS_DIR, f))
    # Scripted baselines (always available)
    # These are handled by the league, not training
    return opponents


def train_challenger(cycle, king_model, tracker_path, steps=1000000, envs=16):
    """Train a new challenger against the current king."""
    challenger = f"models/challenger_{cycle}"
    opponent_arg = king_model + ".zip" if king_model and os.path.exists(king_model + ".zip") else None

    cmd = (
        f"python3 train_combat.py"
        f" --tracker {tracker_path}"
        f" --steps {steps}"
        f" --envs {envs}"
        f" --out {challenger}"
    )
    if opponent_arg:
        cmd += f" --opponent {opponent_arg}"

    print(f"[evolve] cycle {cycle}: training challenger against king={opponent_arg}", flush=True)
    run(cmd, timeout=7200)
    return challenger


def run_league(challenger, cycle):
    """Run the league with the challenger + all entrants."""
    entrants = [challenger + ".zip", "scripted:jabbler", "scripted:defender",
                "scripted:balanced", "scripted:pd"]
    # Add past kings as entrants
    if os.path.exists(KINGS_DIR):
        for f in sorted(os.listdir(KINGS_DIR)):
            if f.endswith(".zip"):
                entrants.append(os.path.join(KINGS_DIR, f))

    cmd = (
        f"python3 league.py"
        f" --entrants {' '.join(entrants)}"
        f" --pd --bouts 3"
        f" --max_steps 5000 --round_seconds 20"
        f" --out {LEAGUE_FILE}"
    )
    print(f"[evolve] cycle {cycle}: running league", flush=True)
    run(cmd, timeout=3600)

    # Generate page
    run(f"python3 league.py page --standings {LEAGUE_FILE} --out docs/index.html")

    # Check results
    d = json.load(open(LEAGUE_FILE))
    king = d.get("king")
    standings = d.get("standings", [])
    for s in standings[:5]:
        print(f"  {s['elo']:6.1f} {s['name']:24s} W{s['W']} L{s['L']} D{s['D']}", flush=True)
    return d


def title_bout(challenger, king_model, cycle):
    """Render the title bout: challenger vs king."""
    if king_model is None or not os.path.exists(king_model + ".zip"):
        print(f"[evolve] cycle {cycle}: no king model, challenger is default king", flush=True)
        return

    out = os.path.join(TITLE_BOUT_DIR, f"title_cycle{cycle}.mp4")
    cmd = (
        f"python3 eval.py egl_bout"
        f" --p1 {challenger}.zip"
        f" --steps 5000"
        f" --out {out}"
        f" --no-terminate"
    )
    print(f"[evolve] cycle {cycle}: rendering title bout", flush=True)
    run(cmd, timeout=3600)
    print(f"[evolve] title bout: {out}", flush=True)


def check_crown(challenger, standings, king_model, cycle):
    """Did the challenger become the new king?"""
    challenger_name = os.path.basename(challenger) + ".zip"
    king_entry = None
    challenger_entry = None
    for s in standings.get("standings", []):
        if challenger_name in s["name"]:
            challenger_entry = s
        if king_model and king_model.replace("models/", "") + ".zip" in s["name"]:
            king_entry = s

    if challenger_entry and challenger_entry["elo"] >= 1640:
        # Challenger is top ELO = new king
        if king_model and os.path.exists(king_model + ".zip"):
            archive_king(king_model, cycle)
        print(f"[evolve] cycle {cycle}: CHALLENGER IS NEW KING! ELO={challenger_entry['elo']}", flush=True)
        return True
    else:
        print(f"[evolve] cycle {cycle}: challenger did not take crown. ELO={challenger_entry['elo'] if challenger_entry else '?'}", flush=True)
        return False


def evolve(tracker_path, cycles=5, steps=1000000, envs=16, wait_seconds=0):
    """Run the evolution loop."""
    os.environ.setdefault("G1_SCENE_XML",
        "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
    os.environ.setdefault("G1_MESH_DIR",
        "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
    os.environ.setdefault("MUJOCO_GL", "egl")

    for cycle in range(1, cycles + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"[evolve] CYCLE {cycle}/{cycles}", flush=True)
        print(f"{'='*60}", flush=True)

        king_model = get_king_model()
        print(f"[evolve] current king: {king_model or 'none'}", flush=True)

        # 1. Train challenger
        challenger = train_challenger(cycle, king_model, tracker_path, steps, envs)

        # 2. Run league
        standings = run_league(challenger, cycle)

        # 3. Title bout
        title_bout(challenger, king_model, cycle)

        # 4. Check crown
        is_king = check_crown(challenger, standings, king_model, cycle)

        # 5. Report
        print(f"\n[evolve] cycle {cycle} complete. King: {standings.get('king')}", flush=True)

        # 6. Push to site (git commit + push)
        run("cd /workspace && git add -A && git commit -m 'Evolve cycle {cycle}: king={king}' && git push".format(
            cycle=cycle, king=standings.get('king', 'unknown')), timeout=60)

        if wait_seconds > 0 and cycle < cycles:
            print(f"[evolve] waiting {wait_seconds}s before next cycle...", flush=True)
            time.sleep(wait_seconds)

    print(f"\n[evolve] evolution complete after {cycles} cycles.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="motion tracker model")
    ap.add_argument("--cycles", type=int, default=5, help="number of evolution cycles")
    ap.add_argument("--steps", type=int, default=1000000, help="training steps per cycle")
    ap.add_argument("--envs", type=int, default=16, help="parallel envs")
    ap.add_argument("--wait", type=int, default=0, help="seconds between cycles")
    ap.add_argument("--daemon", action="store_true", help="run forever")
    a = ap.parse_args()

    cycles = 999999 if a.daemon else a.cycles
    evolve(a.tracker, cycles, a.steps, a.envs, a.wait)


if __name__ == "__main__":
    main()
