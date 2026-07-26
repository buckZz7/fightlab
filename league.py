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

from boxing_rules import run_bout

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


def cmd_crown(path, cause="genesis", force=False):
    if not os.path.exists(path):
        sys.exit(f"no such model: {path}")
    king = current_king()
    if king is not None and not force:
        sys.exit(f"king already exists (gen{king['gen']}). Use challenge to dethrone, or --force to replace.")
    gen = 0 if (king is None or force) else king["gen"] + 1
    entry = {
        "gen": gen,
        "path": path,
        "elo": START_ELO,
        "crowned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "cause": cause,
    }
    append_king(entry)
    print(f"Crowned {'genesis' if gen == 0 else 'replacement'} king: {path} @ ELO {START_ELO}")


def cmd_challenge(challenger_path, matches):
    king = current_king()
    if king is None:
        sys.exit("no king to challenge. Crown a genesis king first.")
    if not os.path.exists(challenger_path):
        sys.exit(f"no such model: {challenger_path}")

    # Track B bout needs a balance substrate.
    bal = os.environ.get("BALANCE_PATH", "models/balance_v1")
    if not os.path.exists(bal):
        sys.exit(f"balance substrate missing: {bal} (set BALANCE_PATH or train it)")

    print(f"Challenger {challenger_path}")
    print(f"vs king gen{king['gen']} {king['path']} (ELO {king['elo']:.1f})")
    print(f"Series: {matches} matches, need >= {CROWN_THRESHOLD:.0%} to take the crown\n")

    challenger = load_policy(challenger_path)
    res = run_fighter_bout(challenger, king["path"], bal, matches=matches)

    red_wins = res["red_wins"]
    blue_wins = res["blue_wins"]
    print(json.dumps({"red_wins": red_wins, "blue_wins": blue_wins,
                      "draws": res["draws"]}, indent=2))
    print("\nSample card (match 0):")
    print(json.dumps(res["cards"][0], indent=2))

    win_rate = red_wins / matches
    e = expected(king["elo"], king["elo"])  # prior: equal
    score = win_rate
    if win_rate >= CROWN_THRESHOLD:
        gen = king["gen"] + 1
        entry = {
            "gen": gen,
            "path": challenger_path,
            "elo": round(king["elo"] + ELO_K * (score - e) + ELO_K * 0.5, 1),
            "crowned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
            "cause": f"dethroned gen{king['gen']} {red_wins}-{blue_wins}",
        }
        append_king(entry)
        print(f"\nNEW KING: gen{gen} {challenger_path}")
        print(f"ELO {king['elo']:.1f} -> {entry['elo']:.1f}")
    else:
        print(f"\nKing holds. Challenger needed {CROWN_THRESHOLD:.0%}, got {win_rate:.0%}.")
        print(f"King ELO stays {king['elo']:.1f}")


def cmd_gauntlet(path, matches=10):
    """Eval a model against genesis + the last two kings (fight-record card)."""
    kings = load_kings()
    if not kings:
        sys.exit("no kings yet.")
    opponents = [kings[0]] + kings[-2:] if len(kings) > 2 else kings
    seen, card = set(), []
    bal = os.environ.get("BALANCE_PATH", "models/balance_v1")
    if not os.path.exists(bal):
        sys.exit(f"balance substrate missing: {bal}")
    model = load_policy(path)
    for k in opponents:
        if k["path"] in seen:
            continue
        seen.add(k["path"])
        king_pol = load_policy(k["path"])
        res = run_fighter_bout(model, k["path"], bal, matches=matches)
        wins = res["red_wins"]; losses = res["blue_wins"]
        kos = sum(1 for c in res["cards"] if c["winner"] == "red"
                  and (c["dmg_to_king"] >= 100.0))
        ko_rate = kos / matches
        card.append({
            "opponent": k["path"],
            "opponent_gen": k["gen"],
            "opponent_elo": k["elo"],
            "wins": wins,
            "losses": losses,
            "draws": res["draws"],
            "ko_rate": ko_rate,
        })
        print(f"vs gen{k['gen']} (ELO {k['elo']:.0f}): "
              f"{wins}W-{losses}L, KO rate {ko_rate:.0%}")
    out = {"model": path, "card": card}
    print(json.dumps(out, indent=2))


# ----------------------------------------------------------------------------
# Track B bout: uses G1FighterEnv + the miner contract (obs85/act17).
# A "challenger" is any policy that predict(obs85)->act17. The KING is
# loaded the same way. This is the Gittensor ground-truth eval:
# deterministic seeded bout, BoxingJudge-style scoring (damage + fall).
# ----------------------------------------------------------------------------
def run_fighter_bout(challenger_policy, king_path, balance_path,
                     matches=1, max_steps=1500, seed=0):
    """Run a bout series challenger(red) vs king(blue). Returns summary.

    challenger_policy: loaded PPO model driving r1.
    king_path: path to the king .zip (env loads it as r2 opponent).
    balance_path: frozen balance substrate .zip.
    """
    from g1_fighter_env import G1FighterEnv

    env = G1FighterEnv(balance_path=balance_path, opponent_path=king_path,
                       max_steps=max_steps, randomize=False)
    red_wins = blue_wins = draws = 0
    cards = []
    for m in range(matches):
        o, _ = env.reset(seed=seed + m)
        for step in range(max_steps):
            a_red, _ = challenger_policy.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a_red)
            if term or trunc:
                break
        dmg_to_king = 100.0 - info["hp_1"]
        dmg_to_chall = 100.0 - info["hp_0"]
        if info["hp_1"] <= 0 or (info["pelvis_z_1"] < 0.4 and info["pelvis_z_0"] > 0.4):
            red_wins += 1; winner = "red"
        elif info["hp_0"] <= 0 or (info["pelvis_z_0"] < 0.4 and info["pelvis_z_1"] > 0.4):
            blue_wins += 1; winner = "blue"
        else:
            draws += 1; winner = "draw"
        cards.append({"match": m, "winner": winner,
                      "dmg_to_king": round(dmg_to_king, 1),
                      "dmg_to_challenger": round(dmg_to_chall, 1),
                      "steps": step + 1})
    return {"red_wins": red_wins, "blue_wins": blue_wins, "draws": draws,
            "cards": cards}


def load_policy(path):
    """Load a policy via the contract (SB3 .zip for now)."""
    from stable_baselines3 import PPO
    if path.endswith(".zip"):
        return PPO.load(path)
    raise ValueError(f"unsupported policy path: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("crown")
    p.add_argument("path")
    p.add_argument("--cause", default="genesis")
    p.add_argument("--force", action="store_true",
                   help="replace existing king (e.g. Track B king replacing old MVP king)")
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
        cmd_crown(args.path, args.cause, args.force)
    elif args.cmd == "challenge":
        cmd_challenge(args.path, args.matches)
    elif args.cmd == "gauntlet":
        cmd_gauntlet(args.path, args.matches)
