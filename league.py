"""King-of-the-hill league: round-robin bouts + ELO standings.

Fighters (trained PPO policies, or PD baselines, or scripted
ShadowBoxer profiles) enter the ring, fight each other via
G1FighterEnv + BoxingJudge, and we score ELO per bout. The king
is the top-ELO fighter.

Each bout = one 1v1 under boxing rules (3 rounds, KO/fall/decision).

Entrant types (in --entrants):
  models/fighter_v1        -> trained PPO policy (red loads it)
  scripted:jabbler         -> scripted aggressive ShadowBoxer
  scripted:defender        -> scripted guard-heavy ShadowBoxer
  scripted:pd              -> PD-to-HOME baseline (random arm, stands)

Usage:
  python3 league.py --entrants models/fighter_v1 scripted:jabbler \
      scripted:defender scripted:pd --pd --bouts 3 \
      --out league_standings.json
"""
import os, sys, argparse, json, itertools, datetime
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge
from bout_fighter import ShadowBoxer


def _elo(ra, rb, score_a):
    """score_a = 1.0 win, 0.5 draw, 0.0 loss. K=32."""
    k = 32.0
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


class _RandomPD:
    """PD baseline: random arm residual (still stands via PD legs)."""
    def __init__(self, env):
        self.env = env
    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


def _load_entrant(spec, env, for_blue=False):
    """Return a callable predict(obs)->(action,None) for the entrant.
    for_blue=True -> loaded as the env's opponent (r2)."""
    if spec.startswith("scripted:"):
        style = spec.split(":", 1)[1]
        if style == "pd":
            return _RandomPD(env)
        # ShadowBoxer: style 'red'/'blue' (desync) + profile
        # (jabbler aggressive / defender counter / balanced).
        sb_style = "blue" if for_blue else "red"
        profile = style if style in ("jabbler", "defender") else "balanced"
        return ShadowBoxer(env, style=sb_style, profile=profile)
    return PPO.load(spec)


def run_bout(name_a, spec_a, name_b, spec_b, balance, max_steps,
              round_seconds, rounds):
    """One bout: A (red) vs B (blue). balance=None -> PD substrate.
    Blue is loaded as the env's opponent (r2). Returns
    (winner_name or None, method, hp, n_rounds)."""
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=max_steps, randomize=False)
    # load blue as opponent (r2) via the env's opponent hook
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    judge = BoxingJudge(env, round_seconds=round_seconds, rounds=rounds)
    red = _load_entrant(spec_a, env, for_blue=False)
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < max_steps:
        a1 = red.predict(obs, deterministic=True)[0]
        obs, rew, term, trunc, info = judge.step(a1)
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1
    card = judge.card()
    w = card["winner"]
    winner = name_a if w == 0 else (name_b if w == 1 else None)
    return winner, card["method"], card["final_hp"], len(card["round_scores"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrants", nargs="+", required=True,
                    help="fighter model paths (e.g. models/fighter_a)")
    ap.add_argument("--balance", default=None,
                    help="balance model path; omit for PD substrate")
    ap.add_argument("--pd", action="store_true",
                    help="use PD-to-HOME substrate")
    ap.add_argument("--bouts", type=int, default=3,
                    help="bouts per pair (round-robin)")
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--round_seconds", type=float, default=30.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="league_standings.json")
    a = ap.parse_args()

    balance = None if a.pd else a.balance
    entrants = {os.path.basename(p): p for p in a.entrants}
    names = list(entrants)
    elo = {n: 1500.0 for n in names}
    record = {n: {"W": 0, "L": 0, "D": 0} for n in names}

    print(f"[league] {len(names)} entrants, {a.bouts} bouts/pair, "
          f"substrate={'PD' if a.pd else balance}")
    results = []
    for na, nb in itertools.combinations(names, 2):
        for b in range(a.bouts):
            w, method, hp, nr = run_bout(na, entrants[na], nb, entrants[nb],
                                           balance, a.max_steps,
                                           a.round_seconds, a.rounds)
            if w == na:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 1.0)
                record[na]["W"] += 1; record[nb]["L"] += 1
                score = "A win"
            elif w == nb:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 0.0)
                record[na]["L"] += 1; record[nb]["W"] += 1
                score = "B win"
            else:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 0.5)
                record[na]["D"] += 1; record[nb]["D"] += 1
                score = "draw"
            results.append({"red": na, "blue": nb, "bout": b + 1,
                             "winner": w, "method": method,
                             "hp": hp, "rounds": nr,
                             "elo_red": round(elo[na], 1),
                             "elo_blue": round(elo[nb], 1)})
            print(f"  {na} vs {nb} #{b+1}: {score} ({method}) "
                  f"hp={hp} elo={elo[na]:.0f}/{elo[nb]:.0f}")

    # standings sorted by ELO
    standings = sorted(
        [{"name": n, "elo": round(elo[n], 1), **record[n]}
         for n in names],
        key=lambda x: -x["elo"])
    king = standings[0]["name"] if standings else None
    out = {
        "king": king,
        "standings": standings,
        "results": results,
        "substrate": "PD" if a.pd else balance,
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[league] KING: {king}  elo={standings[0]['elo'] if standings else 0}")
    print(f"[league] standings -> {a.out}")
    for s in standings:
        print(f"   {s['elo']:6.1f}  {s['name']:24s}  "
              f"W{s['W']} L{s['L']} D{s['D']}")


if __name__ == "__main__":
    main()
