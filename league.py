"""King-of-the-hill league for FightLab boxing.

Lineage file: models/kings.jsonl — one JSON per reign, append-only.
  {"gen": 0, "path": "models/boxing_gen1.zip", "elo": 1000.0,
   "crowned_at": "...", "cause": "genesis"}

  {"gen": 1, "path": "models/boxing_gen2.zip", "elo": 1016.2,
   "crowned_at": "...", "cause": "dethroned gen0 11-4", "challenger_elo_before": 990.1}

Usage:
  python league.py status
  python league.py crown PATH [--cause TEXT]        # genesis crown (first king)
  python league.py challenge CHALLENGER_PATH [--matches 15]
      -> runs series vs current king; challenger crowned if win rate >= threshold
  python league.py gauntlet PATH                    # eval vs genesis + last 2 kings
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from eval_fight import series

KINGS_FILE = os.path.join(os.path.dirname(__file__), "models", "kings.jsonl")
CROWN_THRESHOLD = 0.60      # challenger must win >= 60% of matches
ELO_K = 32
START_ELO = 1000.0

# Challenge rate-limit (starts lenient, tighten only if variance-farming
# appears in practice): escalating cooldown on consecutive failed challenges.
# 1st fail: 24h, 2nd: 3d, 3rd+: 7d. Resets on a successful crown.
COOLDOWN_LADDER_H = [24, 72, 168]


def load_kings():
    if not os.path.exists(KINGS_FILE):
        return []
    with open(KINGS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_king(entry):
    os.makedirs(os.path.dirname(KINGS_FILE), exist_ok=True)
    with open(KINGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def current_king():
    kings = load_kings()
    return kings[-1] if kings else None


def expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def cmd_status():
    kings = load_kings()
    if not kings:
        print("No king crowned yet. Use: league.py crown PATH --cause genesis")
        return
    print(f"{'gen':>4}  {'elo':>7}  {'crowned_at':<20}  path / cause")
    for k in kings:
        print(f"{k['gen']:>4}  {k['elo']:>7.1f}  {k['crowned_at']:<20}  {k['path']}  ({k['cause']})")
    k = kings[-1]
    print(f"\nCurrent king: gen{k['gen']}  {k['path']}  ELO {k['elo']:.1f}")


def cmd_crown(path, cause="genesis"):
    if not os.path.exists(path):
        sys.exit(f"no such model: {path}")
    king = current_king()
    if king is not None:
        sys.exit(f"king already exists (gen{king['gen']}). Use challenge to dethrone.")
    entry = {
        "gen": 0,
        "path": path,
        "elo": START_ELO,
        "crowned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "cause": cause,
    }
    append_king(entry)
    print(f"Crowned genesis king: {path} @ ELO {START_ELO}")


def cmd_challenge(challenger_path, matches):
    king = current_king()
    if king is None:
        sys.exit("no king to challenge. Crown a genesis king first.")
    if not os.path.exists(challenger_path):
        sys.exit(f"no such model: {challenger_path}")

    print(f"Challenger {challenger_path}")
    print(f"vs king gen{king['gen']} {king['path']} (ELO {king['elo']:.1f})")
    print(f"Series: {matches} matches, need >= {CROWN_THRESHOLD:.0%} to take the crown\n")

    # challenger = red, king = blue
    res = series(challenger_path, king["path"], matches=matches)
    print(json.dumps(res, indent=2))

    win_rate = res["red_wins"] / matches
    e = expected(king["elo"], king["elo"])  # prior: equal
    score = win_rate
    new_challenger_elo = king["elo"] + ELO_K * (score - e)
    new_king_elo = king["elo"] + ELO_K * ((1 - score) - e)

    if win_rate >= CROWN_THRESHOLD:
        gen = king["gen"] + 1
        entry = {
            "gen": gen,
            "path": challenger_path,
            "elo": round(king["elo"] + ELO_K * (score - e) + ELO_K * 0.5, 1),
            "crowned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
            "cause": f"dethroned gen{king['gen']} {res['red_wins']}-{res['blue_wins']}",
        }
        append_king(entry)
        print(f"\nNEW KING: gen{gen} {challenger_path}")
        print(f"ELO {king['elo']:.1f} -> {entry['elo']:.1f}")
    else:
        print(f"\nKing holds. Challenger needed {CROWN_THRESHOLD:.0%}, got {win_rate:.0%}.")
        print(f"King ELO stays {king['elo']:.1f} (challenger est. {new_challenger_elo:.1f})")


def cmd_gauntlet(path, matches=10):
    """Eval a model against genesis + the last two kings (fight-record card)."""
    kings = load_kings()
    if not kings:
        sys.exit("no kings yet.")
    opponents = [kings[0]] + kings[-2:] if len(kings) > 2 else kings
    seen, card = set(), []
    for k in opponents:
        if k["path"] in seen:
            continue
        seen.add(k["path"])
        res = series(path, k["path"], matches=matches)
        card.append({
            "opponent": k["path"],
            "opponent_gen": k["gen"],
            "opponent_elo": k["elo"],
            "wins": res["red_wins"],
            "losses": res["blue_wins"],
            "draws": res["draws"],
            "ko_rate": res["ko_rate"],
        })
        print(f"vs gen{k['gen']} (ELO {k['elo']:.0f}): "
              f"{res['red_wins']}W-{res['blue_wins']}L-{res['draws']}D, KO rate {res['ko_rate']:.0%}")
    out = {"model": path, "card": card}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("crown")
    p.add_argument("path")
    p.add_argument("--cause", default="genesis")
    p = sub.add_parser("challenge")
    p.add_argument("path")
    p.add_argument("--matches", type=int, default=15)
    p = sub.add_parser("gauntlet")
    p.add_argument("path")
    p.add_argument("--matches", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "crown":
        cmd_crown(args.path, args.cause)
    elif args.cmd == "challenge":
        cmd_challenge(args.path, args.matches)
    elif args.cmd == "gauntlet":
        cmd_gauntlet(args.path, args.matches)
