"""Auto-update the king-of-the-hill league page when a trained
fighter lands. One-shot: runs the league (mixing trained fighters
+ scripted reference), renders bouts, regenerates the page, and
reports a summary. Intended to be driven by a cron that watches
for models/fighter_v1.zip (or newer checkpoints) on the pod.

Run ON THE POD (has mujoco + the trained models):
  python3 league_update.py --standings docs/league_standings.json

Exits 0 with a short summary printed. Safe to run repeatedly --
it re-derives standings each time.
"""
import os, sys, argparse, json, glob, subprocess
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")


def sh(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  [warn]", (r.stderr or r.stdout)[-400:])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_standings.json")
    ap.add_argument("--bouts", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--render-steps", type=int, default=300)
    ap.add_argument("--max-render-bouts", type=int, default=4)
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Entrants: any trained fighter checkpoints + scripted reference.
    trained = sorted(glob.glob("models/fighter_*.zip"))
    trained = [p[:-4] for p in trained]  # strip .zip
    entrants = trained + [
        "scripted:jabbler", "scripted:defender", "scripted:balanced", "scripted:pd"]
    print(f"[league] entrants: {entrants}")

    # 1) run the league
    sh([sys.executable, "league.py", "--entrants", *entrants,
        "--pd", "--bouts", str(a.bouts),
        "--max_steps", str(a.max_steps), "--round_seconds", "20",
        "--out", a.standings])

    # 2) render top bouts
    sh([sys.executable, "render_league_bouts.py",
        "--standings", a.standings, "--pd",
        "--steps", str(a.render_steps), "--out-dir", "docs/bouts",
        "--max-bouts", str(a.max_render_bouts)])

    # 3) regenerate the page
    sh([sys.executable, "gen_league_page.py",
        "--standings", a.standings, "--out", "docs/index.html"])

    # 4) report
    d = json.load(open(a.standings))
    king = d.get("king")
    print(f"[done] king={king}")
    for s in d.get("standings", [])[:3]:
        print(f"  {s['elo']:6.1f} {s['name']:24s} W{s['W']} L{s['L']} D{s['D']}")


if __name__ == "__main__":
    main()
