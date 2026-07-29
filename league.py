#!/usr/bin/env python3
"""FightLab league — king-of-the-hill ledger (fightlab-league=2).

No ELO. The record is: reigns, title fights, submission decisions.

Commands:
  king                     Show the current king.
  history                  Show reigns and title fights.
  record-title-fight       Record a title-fight series result.
  record-submission        Register a policy submission (challenger pool).
  crown                    Archive the current king's weights to kings/.

State: league_state.json (format fightlab-league=2), atomic writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

DEFAULT_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "league_state.json")
FORMAT = "fightlab-league=2"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "format": FORMAT,
            "current_king": None,
            "reigns": [],
            "title_fights": [],
            "submissions": [],
        }
    with open(path) as f:
        s = json.load(f)
    s.setdefault("format", FORMAT)
    s.setdefault("current_king", None)
    s.setdefault("reigns", [])
    s.setdefault("title_fights", [])
    s.setdefault("submissions", [])
    return s


def save_state(state: dict, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def cmd_king(state, _args):
    king = state.get("current_king")
    if not king:
        print("No king crowned yet.")
        return
    reign = next((r for r in reversed(state["reigns"]) if r["king"] == king and not r.get("ended_at")), None)
    print(f"Current king: {king}")
    if reign:
        print(f"  crowned:  {reign.get('crowned_at')}")
        print(f"  defenses: {reign.get('defenses', 0)}")


def cmd_history(state, _args):
    print("== Reigns ==")
    for r in state["reigns"]:
        span = f"{r.get('crowned_at', '?')} -> {r.get('ended_at') or 'present'}"
        print(f"  {r['king']}  ({span})  defenses: {r.get('defenses', 0)}")
    print("\n== Title fights ==")
    for tf in state["title_fights"]:
        keep = "champion keeps" if tf.get("champion_keeps") else "NEW CHAMPION"
        print(f"  {tf['event']} ({tf.get('date', '?')}): {tf['champion']} vs {tf['challenger']} -> {tf['result']} [{keep}]")


def cmd_record_title_fight(state, args):
    if not state.get("current_king"):
        state["current_king"] = args.champion
        state["reigns"].append({"king": args.champion, "crowned_at": _now(), "ended_at": None, "defenses": 0})
    tf = {
        "event": args.event,
        "date": args.date or _now()[:10],
        "champion": args.champion,
        "challenger": args.challenger,
        "result": args.result,
        "champion_keeps": not args.new_champion,
        "payload_sha256": args.hash or "",
    }
    state["title_fights"].append(tf)
    if args.new_champion:
        for r in reversed(state["reigns"]):
            if r["king"] == args.champion and not r.get("ended_at"):
                r["ended_at"] = _now()
                break
        state["current_king"] = args.challenger
        state["reigns"].append({"king": args.challenger, "crowned_at": _now(), "ended_at": None, "defenses": 0})
        print(f"NEW KING: {args.challenger}")
    else:
        for r in reversed(state["reigns"]):
            if r["king"] == args.champion and not r.get("ended_at"):
                r["defenses"] = r.get("defenses", 0) + 1
                break
        print(f"Recorded: {args.champion} keeps the belt ({args.result})")


def cmd_record_submission(state, args):
    state["submissions"].append({
        "name": args.name, "policy": args.policy,
        "status": args.status, "submitted_at": _now()[:10],
    })
    print(f"Submission recorded: {args.name} ({args.status})")


def cmd_crown(state, args):
    king = state.get("current_king")
    if not king:
        print("No king to archive.")
        return
    slug = king.lower().replace(" ", "-")
    out = os.path.join(args.kings_dir, slug)
    os.makedirs(out, exist_ok=True)
    meta = {
        "king": king,
        "archived_at": _now(),
        "reigns": [r for r in state["reigns"] if r["king"] == king],
    }
    with open(os.path.join(out, "king.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Archived king metadata -> {out}/king.json")


def main() -> int:
    p = argparse.ArgumentParser(description="FightLab king-of-the-hill ledger")
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--kings-dir", default=os.path.join(os.path.dirname(DEFAULT_STATE), "kings"))
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("king")
    sub.add_parser("history")

    sp = sub.add_parser("record-title-fight")
    sp.add_argument("event")
    sp.add_argument("champion")
    sp.add_argument("challenger")
    sp.add_argument("result", help="e.g. '3-2', '2-2-1 draw'")
    sp.add_argument("--new-champion", action="store_true")
    sp.add_argument("--hash", default="")
    sp.add_argument("--date", default="")

    sp = sub.add_parser("record-submission")
    sp.add_argument("name")
    sp.add_argument("policy")
    sp.add_argument("--status", default="challenger")

    sub.add_parser("crown")

    args = p.parse_args()
    state = load_state(args.state)
    cmd = {
        "king": cmd_king,
        "history": cmd_history,
        "record-title-fight": cmd_record_title_fight,
        "record-submission": cmd_record_submission,
        "crown": cmd_crown,
    }[args.command]
    cmd(state, args)
    if args.command.startswith("record"):
        save_state(state, args.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
