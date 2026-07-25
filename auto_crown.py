"""Auto-crown pipeline: test a new policy against the current king and
crown it if it wins a boxing-rules series.

Usage:
  python auto_crown.py models/boxing_gen3.zip --matches 5 --min-winrate 0.6

This is the CI for the King-of-the-Hill MVP: every new generation runs this
against the standing king. If it wins >= min-winrate, it takes the crown and
the league records the reign.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime, timezone
from league import (current_king, load_kings, CROWN_THRESHOLD, ELO_K,
                    expected, append_king)
from boxing_rules import run_bout
from g1_selfplay_env import make_g1_selfplay_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("challenger")
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--min-winrate", type=float, default=CROWN_THRESHOLD)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-seconds", type=float, default=30.0)
    args = ap.parse_args()

    king = current_king()
    if king is None:
        print("No king yet. Use: league.py crown PATH --cause genesis")
        return

    print(f"Challenger: {args.challenger}")
    print(f"Reigning king: gen{king['gen']} {king['path']} (ELO {king['elo']:.1f})")
    print(f"Series: {args.matches} bouts, need >= {args.min_winrate:.0%}\n")

    red_wins = 0
    cards = []
    for m in range(args.matches):
        res = run_bout(
            lambda: make_g1_selfplay_env(
                opponent_path2=king["path"], randomize=False, max_steps=2000),
            args.challenger, king["path"],
            rounds=args.rounds, round_seconds=args.round_seconds)
        cards.append(res)
        if res["red_wins"] > 0:
            red_wins += 1
        print(f"  bout {m+1}: winner={'RED' if res['red_wins'] else 'BLUE'} "
              f"by {res['method']} | pts {res['red_points']:.0f}-{res['blue_points']:.0f} "
              f"| HP {res['final_hp'][0]:.0f}-{res['final_hp'][1]:.0f}")

    win_rate = red_wins / args.matches
    print(f"\nWin rate: {win_rate:.0%} (need {args.min_winrate:.0%})")

    if win_rate >= args.min_winrate:
        # ELO update + crown
        e = expected(king["elo"], king["elo"])  # prior equal
        new_elo = round(king["elo"] + ELO_K * (win_rate - e) + ELO_K * 0.5, 1)
        gen = king["gen"] + 1
        entry = {
            "gen": gen,
            "path": args.challenger,
            "elo": new_elo,
            "crowned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
            "cause": f"dethroned gen{king['gen']} {red_wins}-{args.matches-red_wins} via auto_crown",
        }
        append_king(entry)
        print(f"\n*** NEW KING: gen{gen} {args.challenger} (ELO {new_elo:.1f}) ***")
    else:
        print(f"\nKing holds. Challenger {args.challenger} stays unranked.")


def timezone_utc():
    from datetime import timezone
    return timezone.utc


if __name__ == "__main__":
    main()
