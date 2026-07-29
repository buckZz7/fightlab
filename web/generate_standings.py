#!/usr/bin/env python3
"""Generate league standings JSON from league_state.json.

Reads the real league state (fighters + bouts) and emits a JSON file
that the FightLab landing page fetches on load to replace the mockup
standings table with live data.

Standard library only — no third-party deps (per AGENTS.md).

Usage:
    python3 web/generate_standings.py
    python3 web/generate_standings.py --state path/to/league_state.json --out path/to/standings.json
"""

import argparse
import datetime
import json
import os
import sys
from typing import Any

# Resolve default paths relative to this script so it works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join(HERE, "..", "league_state.json")
DEFAULT_OUT = os.path.join(HERE, "standings.json")


def _utc_now_iso() -> str:
    """Current UTC time in ISO-8601 with a Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from_epoch(ts: float) -> str:
    """Convert an epoch timestamp to ISO-8601 UTC with a Z suffix."""
    try:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _compute_streak(fighter: dict[str, Any], bouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the current win/loss/draw streak for a fighter.

    Walks bouts in chronological order (oldest first) and tracks the
    consecutive same-result run ending at the most recent bout for this
    fighter. Returns {"kind": "W"|"L"|"D"|"-", "count": int, "cls": css-class}.
    """
    name = fighter.get("name")
    if not name or not bouts:
        return {"kind": "-", "count": 0, "cls": "streak-flat", "label": "— 0"}

    # Bouts are appended chronologically; iterate newest -> oldest.
    streak_kind = "-"
    streak_count = 0
    for bout in reversed(bouts):
        a = bout.get("fighter_a")
        b = bout.get("fighter_b")
        if name not in (a, b):
            continue
        result = bout.get("result")
        if result == "draw":
            kind = "D"
        elif result == "a_wins" and name == a:
            kind = "W"
        elif result == "b_wins" and name == b:
            kind = "W"
        else:
            kind = "L"

        if streak_count == 0:
            streak_kind = kind
            streak_count = 1
        elif kind == streak_kind:
            streak_count += 1
        else:
            break

    if streak_count == 0:
        cls = "streak-flat"
        label = "— 0"
    elif streak_kind == "W":
        cls = "streak-up"
        label = f"W {streak_count}"
    elif streak_kind == "L":
        cls = "streak-down"
        label = f"L {streak_count}"
    elif streak_kind == "D":
        cls = "streak-flat"
        label = f"D {streak_count}"
    else:
        cls = "streak-flat"
        label = "— 0"

    return {"kind": streak_kind, "count": streak_count, "cls": cls, "label": label}


def _win_rate(wins: int, losses: int, draws: int) -> str:
    """Win rate as a percentage string with one decimal. Draws excluded from denominator."""
    decided = wins + losses
    if decided == 0:
        return "—"
    rate = (wins / decided) * 100.0
    return f"{rate:.1f}%"


def build_standings(state: dict[str, Any]) -> dict[str, Any]:
    """Build the standings payload from the raw league state."""
    fighters_map: dict[str, Any] = state.get("fighters", {}) or {}
    bouts: list[dict[str, Any]] = state.get("bouts", []) or []

    rows = []
    for name, f in fighters_map.items():
        wins = int(f.get("wins", 0))
        losses = int(f.get("losses", 0))
        draws = int(f.get("draws", 0))
        bouts_count = int(f.get("bouts", 0))
        elo = float(f.get("elo", 1000.0))
        created_at = f.get("created_at")

        streak = _compute_streak(f, bouts)

        rows.append({
            "name": name,
            "elo": round(elo, 1),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "bouts": bouts_count,
            "win_rate": _win_rate(wins, losses, draws),
            "record": f"{wins}–{losses}" + (f"–{draws}" if draws > 0 else ""),
            "streak_label": streak["label"],
            "streak_cls": streak["cls"],
            "policy_path": f.get("policy_path", ""),
            "created_at": _iso_from_epoch(created_at) if created_at else "",
        })

    # Rank by ELO desc, then by wins desc, then name asc.
    rows.sort(key=lambda r: (-r["elo"], -r["wins"], r["name"]))

    # King = rank 1 with at least one bout. If no bouts yet, no king.
    has_bouts = len(bouts) > 0
    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["is_king"] = has_bouts and i == 0

    total_bouts = len(bouts)
    total_fighters = len(rows)

    return {
        "generated_at": _utc_now_iso(),
        "season": "Season 01",
        "total_fighters": total_fighters,
        "total_bouts": total_bouts,
        "has_bouts": has_bouts,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate FightLab standings JSON from league_state.json"
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE,
        help=f"Path to league_state.json (default: {DEFAULT_STATE})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the JSON output"
    )
    args = parser.parse_args(argv)

    state_path = os.path.abspath(args.state)
    out_path = os.path.abspath(args.out)

    if not os.path.exists(state_path):
        print(f"error: state file not found: {state_path}", file=sys.stderr)
        return 1

    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {state_path}: {exc}", file=sys.stderr)
        return 1

    payload = build_standings(state)

    indent = 2 if args.pretty else None
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")

    print(f"standings written: {out_path}")
    print(
        f"  fighters={payload['total_fighters']} bouts={payload['total_bouts']} "
        f"generated_at={payload['generated_at']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
