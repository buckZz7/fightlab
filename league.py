#!/usr/bin/env python3
"""FightLab league system.

A self-contained ELO league for autonomous humanoid combat (Unitree G1).

Design:
  - JSON file is the single source of truth (league_state.json).
  - Each fighter has a miner-chosen name, a policy path, and an ELO rating.
  - Bouts update ELO via standard expected-score math with a K-factor.
  - When a fighter reaches the top ELO, they can be crowned king; their
    policy weights are archived to kings/ with a metadata sidecar.

Usage as a library:
    from league import League
    lg = League("/path/to/league_state.json")
    lg.add_fighter("IronFist", "/policies/ironfist.pt")
    lg.record_bout("IronFist", "StoneHand", winner="IronFist")
    lg.standings()

Usage from CLI:
    python league.py --state league_state.json add IronFist /policies/ironfist.pt
    python league.py --state league_state.json bout IronFist StoneHand --winner IronFist
    python league.py --state league_state.json standings
    python league.py --state league_state.json schedule --rounds 1
    python league.py --state league_state.json crown
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

# --- Constants ---------------------------------------------------------------

DEFAULT_ELO = 1000.0
K_FACTOR = 32.0           # Standard chess-style K-factor.
ELO_DIVISOR = 400.0       # Denominator in the expected-score formula.

KINGS_DIR_NAME = "kings"
KINGS_META_NAME = "king.json"


# --- Data models ------------------------------------------------------------


def _now() -> float:
    """UTC unix timestamp (seconds). Kept as a function so tests can patch it."""
    return time.time()


@dataclass
class Fighter:
    """A miner-submitted fighter and its standing in the league."""

    name: str
    policy_path: str
    elo: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0
    bouts: int = 0
    created_at: float = field(default_factory=_now)

    def win_rate(self) -> float:
        played = self.wins + self.losses + self.draws
        if played == 0:
            return 0.0
        # Draws count as half a win, standard convention.
        return (self.wins + 0.5 * self.draws) / played

    def record(self) -> str:
        return f"{self.wins}-{self.losses}-{self.draws}"


@dataclass
class Bout:
    """A completed bout between two fighters."""

    bout_id: str
    fighter_a: str
    fighter_b: str
    winner: Optional[str]      # None means draw
    score_a: float = 0.0
    score_b: float = 0.0
    timestamp: float = field(default_factory=_now)


# --- ELO math ----------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for A against B, in [0, 1]."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_DIVISOR))


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Return new (rating_a, rating_b) after a bout.

    `score_a` is 1.0 for A win, 0.0 for B win, 0.5 for draw.
    `score_b` is implicitly `1 - score_a`.
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    sb = 1.0 - score_a
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * (sb - eb)
    return round(new_a, 2), round(new_b, 2)


# --- Round-robin scheduler ---------------------------------------------------


def round_robin(names: list[str], double: bool = False) -> list[tuple[str, str]]:
    """Generate a round-robin schedule.

    Uses the circle method. If `double`, each pair plays home and away
    (both orderings). Otherwise single round (each pair once).

    Args:
        names: fighter names.
        double: if True, generate home/away (both orderings) for each pair.

    Returns:
        List of (home, away) tuples in play order.
    """
    if len(names) < 2:
        return []
    pool = list(names)
    if len(pool) % 2 != 0:
        pool.append(None)  # bye

    n = len(pool)
    half = n // 2
    fixed = pool[0]
    rotating = pool[1:]
    bouts: list[tuple[str, str]] = []

    for _ in range(n - 1):
        round_fighters = [fixed] + rotating
        for i in range(half):
            a = round_fighters[i]
            b = round_fighters[-1 - i]
            if a is None or b is None:
                continue  # bye
            bouts.append((a, b))
        # rotate: keep first of rotating fixed, shift the rest
        rotating = [rotating[-1]] + rotating[:-1]

    if double:
        bouts = bouts + [(b, a) for (a, b) in bouts]

    return bouts


# --- League ------------------------------------------------------------------


class League:
    """The FightLab league. State persists to a JSON file."""

    def __init__(self, state_path: str | Path, kings_dir: str | Path | None = None):
        self.state_path = Path(state_path)
        if kings_dir is None:
            # Default: a sibling `kings` directory next to the state file.
            kings_dir = self.state_path.parent / KINGS_DIR_NAME
        self.kings_dir = Path(kings_dir)
        self.kings_dir.mkdir(parents=True, exist_ok=True)
        self.fighters: dict[str, Fighter] = {}
        self.bouts: list[Bout] = []
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        with self.state_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.fighters = {
            name: Fighter(**f) for name, f in data.get("fighters", {}).items()
        }
        self.bouts = [Bout(**b) for b in data.get("bouts", [])]

    def _save(self) -> None:
        data = {
            "fighters": {n: asdict(f) for n, f in self.fighters.items()},
            "bouts": [asdict(b) for b in self.bouts],
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.state_path)

    # -- fighter management --------------------------------------------------

    def add_fighter(self, name: str, policy_path: str) -> Fighter:
        """Register a new fighter. Raises if the name is taken."""
        if name in self.fighters:
            raise ValueError(f"Fighter '{name}' already exists")
        fighter = Fighter(name=name, policy_path=policy_path)
        self.fighters[name] = fighter
        self._save()
        return fighter

    def remove_fighter(self, name: str) -> None:
        """Remove a fighter from the league."""
        if name not in self.fighters:
            raise ValueError(f"Fighter '{name}' not found")
        del self.fighters[name]
        self._save()

    def get_fighter(self, name: str) -> Fighter:
        if name not in self.fighters:
            raise ValueError(f"Fighter '{name}' not found")
        return self.fighters[name]

    # -- bouts ---------------------------------------------------------------

    def record_bout(
        self,
        fighter_a: str,
        fighter_b: str,
        winner: Optional[str] = None,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Bout:
        """Record a completed bout and update ELO + records.

        `winner` is one of fighter_a, fighter_b, or None (draw).
        Updates both fighters' ELO and W/L/D counts, and persists state.

        Returns the created Bout.
        """
        if fighter_a not in self.fighters:
            raise ValueError(f"Fighter '{fighter_a}' not found")
        if fighter_b not in self.fighters:
            raise ValueError(f"Fighter '{fighter_b}' not found")
        if fighter_a == fighter_b:
            raise ValueError("A fighter cannot fight themselves")

        fa = self.fighters[fighter_a]
        fb = self.fighters[fighter_b]

        if winner is None:
            score_a_elo = 0.5
        elif winner == fighter_a:
            score_a_elo = 1.0
        elif winner == fighter_b:
            score_a_elo = 0.0
        else:
            raise ValueError(
                f"winner must be '{fighter_a}', '{fighter_b}', or None (draw); "
                f"got '{winner}'"
            )

        new_a, new_b = elo_update(fa.elo, fb.elo, score_a_elo)
        fa.elo, fb.elo = new_a, new_b
        fa.bouts += 1
        fb.bouts += 1
        if winner is None:
            fa.draws += 1
            fb.draws += 1
        elif winner == fighter_a:
            fa.wins += 1
            fb.losses += 1
        else:
            fa.losses += 1
            fb.wins += 1

        bout = Bout(
            bout_id=uuid.uuid4().hex[:12],
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
        )
        self.bouts.append(bout)
        self._save()
        return bout

    def bout_history(self, fighter: Optional[str] = None) -> list[Bout]:
        """Return bout history, optionally filtered to one fighter."""
        if fighter is None:
            return list(self.bouts)
        return [
            b for b in self.bouts
            if b.fighter_a == fighter or b.fighter_b == fighter
        ]

    # -- standings -----------------------------------------------------------

    def standings(self) -> list[dict[str, Any]]:
        """Return fighters ranked by ELO (descending).

        Each row: rank, name, elo, record (W-L-D), win_rate, bouts, policy_path.
        """
        rows = []
        for f in self.fighters.values():
            rows.append({
                "name": f.name,
                "elo": f.elo,
                "wins": f.wins,
                "losses": f.losses,
                "draws": f.draws,
                "bouts": f.bouts,
                "win_rate": round(f.win_rate(), 3),
                "record": f.record(),
                "policy_path": f.policy_path,
            })
        rows.sort(key=lambda r: r["elo"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        return rows

    def king(self) -> Optional[Fighter]:
        """Return the current top-ELO fighter, or None if the league is empty."""
        if not self.fighters:
            return None
        return max(self.fighters.values(), key=lambda f: f.elo)

    # -- king archiving ------------------------------------------------------

    def crown_king(self) -> Optional[dict[str, Any]]:
        """Archive the current top-ELO fighter as king.

        Copies their policy file (if it exists) into kings/<name>/
        and writes a metadata sidecar (king.json). If no policy file exists,
        only the metadata is written (the weights are expected to arrive
        out-of-band in the real pipeline).

        Returns the metadata dict, or None if the league is empty.
        """
        king = self.king()
        if king is None:
            return None

        slug = _slug(king.name)
        dest_dir = self.kings_dir / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy weights if the policy file exists locally.
        archived_path: Optional[str]
        if king.policy_path and Path(king.policy_path).is_file():
            dest_file = dest_dir / Path(king.policy_path).name
            shutil.copy2(king.policy_path, dest_file)
            archived_path = str(dest_file)
        else:
            archived_path = None

        meta = {
            "fighter_name": king.name,
            "elo": king.elo,
            "record": king.record(),
            "wins": king.wins,
            "losses": king.losses,
            "draws": king.draws,
            "bouts": king.bouts,
            "policy_path": king.policy_path,
            "archived_path": archived_path,
            "crowned_at": _now(),
        }
        meta_path = dest_dir / KINGS_META_NAME
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return meta

    # -- scheduling ----------------------------------------------------------

    def schedule(
        self,
        names: Optional[Iterable[str]] = None,
        double: bool = False,
    ) -> list[tuple[str, str]]:
        """Generate a round-robin schedule.

        Args:
            names: fighter names to schedule. Defaults to all current fighters.
            double: if True, home-and-away (both orderings) for each pair.

        Returns:
            List of (home, away) bout tuples in play order.
        """
        if names is None:
            names = list(self.fighters.keys())
        names = list(names)
        for n in names:
            if n not in self.fighters:
                raise ValueError(f"Fighter '{n}' not found")
        return round_robin(names, double=double)


# --- Helpers ----------------------------------------------------------------


def _slug(name: str) -> str:
    """Filesystem-safe slug for a fighter name."""
    keep = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
        elif ch in " .":
            keep.append("-")
    slug = "".join(keep)
    # collapse runs of '-'
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "fighter"


# --- CLI --------------------------------------------------------------------


def _print_standings(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No fighters registered.")
        return
    header = f"{'#':>2}  {'Name':<20} {'ELO':>7}  {'Record':<9}  {'W%':>5}  {'Bouts':>5}  Policy"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['rank']:>2}  {r['name']:<20} {r['elo']:>7.1f}  "
            f"{r['record']:<9}  {r['win_rate']:>5.3f}  {r['bouts']:>5}  {r['policy_path']}"
        )


def _print_schedule(bouts: list[tuple[str, str]]) -> None:
    if not bouts:
        print("No bouts scheduled.")
        return
    print(f"{'#':>3}  {'Home':<20} {'Away':<20}")
    print("-" * 46)
    for i, (a, b) in enumerate(bouts, 1):
        print(f"{i:>3}  {a:<20} {b:<20}")


def _print_bouts(bouts: list[Bout]) -> None:
    if not bouts:
        print("No bouts recorded.")
        return
    print(f"{'ID':<12}  {'A':<16} {'B':<16} {'Winner':<16}  {'Score':<11}  Timestamp")
    print("-" * 92)
    for b in bouts:
        winner = b.winner if b.winner is not None else "draw"
        score = f"{b.score_a:g}-{b.score_b:g}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(b.timestamp)) + "Z"
        print(
            f"{b.bout_id:<12}  {b.fighter_a:<16} {b.fighter_b:<16} "
            f"{winner:<16}  {score:<11}  {ts}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="league",
        description="FightLab league system — ELO rankings, scheduling, king archiving.",
    )
    p.add_argument(
        "--state",
        default="league_state.json",
        help="Path to the league state JSON file (default: league_state.json).",
    )
    p.add_argument(
        "--kings-dir",
        default=None,
        help="Directory for archived king weights (default: <state_dir>/kings).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("add", help="Register a new fighter.")
    sp.add_argument("name", help="Fighter name (miner-chosen).")
    sp.add_argument("policy_path", help="Path to the policy weights file.")

    sp = sub.add_parser("remove", help="Remove a fighter.")
    sp.add_argument("name")

    sp = sub.add_parser("bout", help="Record a bout result and update ELO.")
    sp.add_argument("fighter_a")
    sp.add_argument("fighter_b")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--winner", default=None, help="Name of the winner (default: draw).")
    sp.add_argument("--score-a", type=float, default=0.0, help="Score for fighter A.")
    sp.add_argument("--score-b", type=float, default=0.0, help="Score for fighter B.")

    sp = sub.add_parser("standings", help="Print current ELO standings.")
    sp.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    sp = sub.add_parser("schedule", help="Print a round-robin schedule.")
    sp.add_argument("--rounds", type=int, default=1, choices=[1, 2],
                    help="1 = single round (default), 2 = home and away.")
    sp.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    sp = sub.add_parser("history", help="Print bout history.")
    sp.add_argument("fighter", nargs="?", default=None,
                    help="Filter to one fighter (optional).")

    sp = sub.add_parser("crown", help="Archive the current top-ELO fighter as king.")
    sp.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    sp = sub.add_parser("king", help="Show the current top-ELO fighter.")
    sp.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    sp = sub.add_parser("show", help="Show one fighter's details.")
    sp.add_argument("name")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    lg = League(args.state, kings_dir=args.kings_dir)

    if args.command == "add":
        f = lg.add_fighter(args.name, args.policy_path)
        print(f"Added fighter '{f.name}' (ELO {f.elo:.0f}, policy: {f.policy_path})")

    elif args.command == "remove":
        lg.remove_fighter(args.name)
        print(f"Removed fighter '{args.name}'.")

    elif args.command == "bout":
        bout = lg.record_bout(
            args.fighter_a, args.fighter_b,
            winner=args.winner,
            score_a=args.score_a, score_b=args.score_b,
        )
        fa = lg.get_fighter(args.fighter_a)
        fb = lg.get_fighter(args.fighter_b)
        w = bout.winner if bout.winner else "draw"
        print(f"Bout {bout.bout_id}: {args.fighter_a} vs {args.fighter_b} -> {w}")
        print(f"  {fa.name}: ELO {fa.elo:.1f} ({fa.record()})")
        print(f"  {fb.name}: ELO {fb.elo:.1f} ({fb.record()})")

    elif args.command == "standings":
        rows = lg.standings()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            _print_standings(rows)

    elif args.command == "schedule":
        bouts = lg.schedule(double=(args.rounds == 2))
        if args.json:
            print(json.dumps([{"home": a, "away": b} for a, b in bouts], indent=2))
        else:
            _print_schedule(bouts)

    elif args.command == "history":
        bouts = lg.bout_history(args.fighter)
        _print_bouts(bouts)

    elif args.command == "crown":
        meta = lg.crown_king()
        if meta is None:
            print("No fighters to crown.")
            return 1
        if args.json:
            print(json.dumps(meta, indent=2))
        else:
            print(f"Crowned king: {meta['fighter_name']}")
            print(f"  ELO: {meta['elo']:.1f}  Record: {meta['record']}  Bouts: {meta['bouts']}")
            if meta.get("archived_path"):
                print(f"  Weights archived to: {meta['archived_path']}")
            else:
                print("  (No local policy file found; metadata written only.)")
            print(f"  Metadata: {lg.kings_dir / _slug(meta['fighter_name']) / KINGS_META_NAME}")

    elif args.command == "king":
        k = lg.king()
        if k is None:
            print("No fighters registered.")
            return 1
        if args.json:
            print(json.dumps(asdict(k), indent=2))
        else:
            print(f"King: {k.name}  ELO {k.elo:.1f}  Record {k.record()}  Bouts {k.bouts}")

    elif args.command == "show":
        f = lg.get_fighter(args.name)
        print(json.dumps(asdict(f), indent=2))

    else:
        build_arg_parser().print_help()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
