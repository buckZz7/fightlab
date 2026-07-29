#!/usr/bin/env python3
"""FightLab Miner SDK -- submit, evaluate, and track fighting policies.

FightLab is autonomous humanoid combat RL for the Unitree G1 in MuJoCo.
Miners train their own policies and submit them to a trustless league where
bouts are refereed by eval_harness.py and rankings are maintained by
league.py. This module is the single entry point a miner needs.

Capabilities
------------
1. FighterSubmission  -- validate a policy file, package it with metadata.
2. submit()           -- register a fighter with the league.
3. evaluate()         -- run the trustless eval harness against the king or
                          a named opponent, optionally recording the result.
4. check_status()     -- query the league for ELO, rank, and record.
5. CLI                -- fightlab-sdk submit|status|challenge ...

Environment interface (what miners build against)
-------------------------------------------------
DEPRECATED: The 41D obs / 17D action interface documented below is the
v1 interface. It is kept for backwards compatibility with the v1
walker-based stack. The v2 pipeline uses a 32-D latent action space over
23-DoF PD targets (see docs/v2-pipeline.md). New miners should target
the v2 latent interface once the first v2 king is crowned.

The combat environment (fight_env.py on the training pod, not bundled here)
exposes a 41-dimensional observation and a 17-dimensional action.

  Action (17D), contiguous, roughly [-1, 1] per component:
    [0:3]   velocity commands  -- (vx, vy, yaw_rate) for the locomotion
                                 controller. The G1's balance is maintained
                                 by a separate walker.onnx policy; these
                                 three values steer the walker.
    [3:17]  arm joint targets  -- 7 per arm (shoulder pitch/roll/yaw,
                                 elbow, wrist roll/pitch, gripper or
                                 forearm). Left arm [3:10], right arm [10:17].

  Observation (41D):
    [0:3]   root linear velocity  (x, y, z), body frame
    [3:6]   root angular velocity (roll, pitch, yaw rate), body frame
    [6:9]   projected gravity     (sin/cos of tilt in 3 axes)
    [9:23]  arm joint positions   14 values (7 left + 7 right), radians
    [23:37] arm joint velocities  14 values (7 left + 7 right), rad/s
    [37:39] opponent relative position (x, y), arena frame
    [39:41] opponent relative heading  (cos, sin of bearing)

  Note: the exact observation layout should be verified against the
  local fight_env.py on your training pod. The dimensions (41 obs, 17 act)
  and the action split (3 vel + 14 arm) are stable; the per-index semantics
  above are the documented contract.

Design
------
- Standard library only (hashlib, json, argparse, pathlib, datetime, sys).
- Imports league.py and eval_harness.py from the same directory.
- All timestamps are UTC, ISO-8601 with a Z suffix.
- No emojis, no external dependencies, no network calls.

Usage as a library
------------------
    from miner_sdk import FighterSubmission, submit, evaluate, check_status

    # 1. Package and validate
    sub = FighterSubmission(
        policy_path="/policies/ironfist.pt",
        fighter_name="IronFist",
        miner_name="alice",
    )
    print(sub.sha256, sub.size_bytes)

    # 2. Register with the league
    fighter = submit(sub, state_path="league_state.json")

    # 3. Evaluate against the current king
    result = evaluate("IronFist", state_path="league_state.json")

    # 4. Check standing
    status = check_status("IronFist", state_path="league_state.json")
    print(status["elo"], status["rank"], status["record"])

Usage from CLI
--------------
    fightlab-sdk submit /policies/ironfist.pt --name IronFist --miner alice
    fightlab-sdk status IronFist
    fightlab-sdk challenge StoneHand --name IronFist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# --- Local imports (league.py and eval_harness.py live alongside this file) --

_THIS_DIR = Path(__file__).resolve().parent
# Root (league.py, miner_sdk.py) and engine/ (eval_harness.py, baselines.py,
# real_bout_runner.py) must both be importable.
for _p in (_THIS_DIR, _THIS_DIR / "engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from league import League  # noqa: E402
from eval_harness import (  # noqa: E402
    EvalHarness,
    BoutConfig,
    BoutResult,
    MockBoutRunner,
    PolicyFileError,
    PolicySizeError,
    DEFAULT_SIZE_CAP_BYTES,
    DEFAULT_SEEDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_DAMAGE_THRESHOLD,
)


# --- Constants ---------------------------------------------------------------

OBS_DIM = 41
ACT_DIM = 17
VEL_ACTION_DIM = 3
ARM_ACTION_DIM = 14

DEFAULT_STATE_PATH = str(_THIS_DIR / "league_state.json")
DEFAULT_SIZE_LIMIT = DEFAULT_SIZE_CAP_BYTES  # 50 MB


# --- UTC timestamp helper ----------------------------------------------------


def _now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a Z suffix.

    Example: 2026-07-27T14:33:01.123456Z
    """
    t = time.time()
    secs = int(t)
    micros = int((t - secs) * 1_000_000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + f".{micros:06d}Z"


def _ts_to_iso(ts: float) -> str:
    """Convert a UTC unix timestamp (seconds) to ISO-8601 with a Z suffix."""
    secs = int(ts)
    micros = int((ts - secs) * 1_000_000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + f".{micros:06d}Z"


# --- File hashing (streaming SHA-256) ----------------------------------------


def _hash_file(path: str | Path) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for a file, read in 1 MiB chunks."""
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# --- Errors ------------------------------------------------------------------


class FighterNotRegisteredError(Exception):
    """Raised when a fighter name is not found in the league."""


# --- FighterSubmission -------------------------------------------------------


@dataclass
class FighterSubmission:
    """A validated policy submission packaged with metadata.

    Validates that the policy file exists and is under the size limit,
    computes a SHA-256 hash, and records the submission timestamp in UTC.

    Attributes:
        policy_path:  Absolute or relative path to the policy weights file.
        fighter_name: Miner-chosen name for this fighter (unique in the league).
        miner_name:   Name of the miner submitting the policy.
        size_limit:   Maximum allowed file size in bytes (default 50 MB).
        sha256:       SHA-256 hex digest of the policy file (computed on init).
        size_bytes:   Size of the policy file in bytes (computed on init).
        submitted_at: UTC ISO-8601 timestamp when this object was created.
    """

    policy_path: str
    fighter_name: str
    miner_name: str
    size_limit: int = DEFAULT_SIZE_LIMIT
    sha256: str = field(default="", init=False)
    size_bytes: int = field(default=0, init=False)
    submitted_at: str = field(default_factory=_now_utc_iso, init=False)

    def __post_init__(self) -> None:
        self._validate()
        self.sha256, self.size_bytes = _hash_file(self.policy_path)
        if self.size_bytes > self.size_limit:
            raise PolicySizeError(
                f"Policy '{self.policy_path}' is {self.size_bytes} bytes, "
                f"exceeds limit {self.size_limit} bytes "
                f"({self.size_limit / (1024 * 1024):.1f} MB)."
            )

    def _validate(self) -> None:
        p = Path(self.policy_path)
        if not p.is_file():
            raise PolicyFileError(f"Policy file not found: {self.policy_path}")
        if not os.access(p, os.R_OK):
            raise PolicyFileError(f"Policy file not readable: {self.policy_path}")

    def metadata(self) -> dict[str, Any]:
        """Return the submission metadata as a JSON-serializable dict."""
        return {
            "fighter_name": self.fighter_name,
            "miner_name": self.miner_name,
            "policy_path": str(Path(self.policy_path).resolve()),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "size_limit": self.size_limit,
            "submitted_at": self.submitted_at,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
        }

    def to_json(self) -> str:
        """Serialize metadata to a pretty JSON string (sorted keys)."""
        return json.dumps(self.metadata(), indent=2, sort_keys=True)

    def __repr__(self) -> str:
        return (
            f"FighterSubmission(fighter_name={self.fighter_name!r}, "
            f"miner_name={self.miner_name!r}, "
            f"sha256={self.sha256[:12]}..., "
            f"size={self.size_bytes} bytes)"
        )


# --- Core functions ----------------------------------------------------------


def submit(
    submission: FighterSubmission,
    state_path: str = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    """Register a fighter with the league.

    Calls League.add_fighter under the hood. If the fighter name already
    exists, the league raises ValueError; this function lets it propagate.

    Args:
        submission:  A validated FighterSubmission.
        state_path:   Path to the league state JSON file.

    Returns:
        A dict with the fighter's initial league state (name, elo, record,
        policy_path, created_at).
    """
    lg = League(state_path)
    fighter = lg.add_fighter(submission.fighter_name, str(Path(submission.policy_path).resolve()))
    return {
        "name": fighter.name,
        "elo": fighter.elo,
        "wins": fighter.wins,
        "losses": fighter.losses,
        "draws": fighter.draws,
        "bouts": fighter.bouts,
        "policy_path": fighter.policy_path,
        "created_at": _ts_to_iso(fighter.created_at),
    }


def evaluate(
    fighter_name: str,
    state_path: str = DEFAULT_STATE_PATH,
    opponent: Optional[str] = None,
    seeds: Optional[list[int]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
    record_bout: bool = True,
    signing_key: Optional[str] = None,
) -> dict[str, Any]:
    """Run the trustless eval harness against the king or a named opponent.

    Looks up both fighters' policy paths from the league, runs a multi-seed
    bout via EvalHarness, and optionally records the result in the league
    (updating ELO and W/L/D records).

    Args:
        fighter_name:    The submitting fighter's name (fighter A in the bout).
        state_path:       Path to the league state JSON file.
        opponent:        Opponent's name. If None, evaluates against the
                         current king (top ELO). Raises if the league is
                         empty or the only fighter is the submitter.
        seeds:           List of seeds for the bout. Defaults to [1..5].
        max_steps:       Maximum steps per seed bout.
        damage_threshold: Minimum total damage for a win to count.
        record_bout:     If True, record the result in the league (update ELO).
        signing_key:     Optional HMAC key for signed bout results.

    Returns:
        A dict combining the bout result summary and (if recorded) the
        updated ELO for both fighters.

    Raises:
        FighterNotRegisteredError: if fighter_name or opponent is not in the league.
        ValueError: if no opponent can be determined.
    """
    lg = League(state_path)

    # Resolve the submitting fighter.
    if fighter_name not in lg.fighters:
        raise FighterNotRegisteredError(f"Fighter '{fighter_name}' not found")
    fighter = lg.get_fighter(fighter_name)

    # Resolve the opponent.
    if opponent is not None:
        if opponent not in lg.fighters:
            raise FighterNotRegisteredError(f"Opponent '{opponent}' not found")
        opponent_fighter = lg.get_fighter(opponent)
    else:
        # Evaluate against the current king.
        king = lg.king()
        if king is None:
            raise ValueError("League is empty; no opponent to evaluate against.")
        if king.name == fighter_name:
            # Only one fighter or the submitter is already king -- pick the
            # next best opponent.
            others = [f for f in lg.fighters.values() if f.name != fighter_name]
            if not others:
                raise ValueError(
                    f"'{fighter_name}' is the only fighter; no opponent available."
                )
            opponent_fighter = max(others, key=lambda f: f.elo)
        else:
            opponent_fighter = king

    if fighter_name == opponent_fighter.name:
        raise ValueError("A fighter cannot fight themselves.")

    # Build the eval config.
    if seeds is None:
        seeds = list(range(1, DEFAULT_SEEDS + 1))

    cfg = BoutConfig(
        seeds=seeds,
        max_steps=max_steps,
        damage_threshold=damage_threshold,
    )
    harness = EvalHarness(config=cfg, runner=MockBoutRunner(), signing_key=signing_key)

    # Run the bout: fighter is A, opponent is B.
    result = harness.run_bout(fighter.policy_path, opponent_fighter.policy_path)

    # Determine the league-level winner name.
    agg = result.aggregate
    winner_name: Optional[str]
    if agg["overall_winner"] == "A":
        winner_name = fighter_name
    elif agg["overall_winner"] == "B":
        winner_name = opponent_fighter.name
    else:
        winner_name = None  # draw or no contest

    # Record the bout in the league (updates ELO + W/L/D).
    updated_elo: dict[str, float] = {}
    if record_bout:
        lg.record_bout(
            fighter_name,
            opponent_fighter.name,
            winner=winner_name,
            score_a=agg["total_damage_a"],
            score_b=agg["total_damage_b"],
        )
        updated_elo = {
            fighter_name: lg.get_fighter(fighter_name).elo,
            opponent_fighter.name: lg.get_fighter(opponent_fighter.name).elo,
        }

    return {
        "bout_id": result.bout_id,
        "fighter": fighter_name,
        "opponent": opponent_fighter.name,
        "winner": winner_name or "draw",
        "confidence": agg["confidence"],
        "seeds_won_fighter": agg["seeds_won_a"],
        "seeds_won_opponent": agg["seeds_won_b"],
        "draws": agg["draws"],
        "total_damage_fighter": agg["total_damage_a"],
        "total_damage_opponent": agg["total_damage_b"],
        "damage_gate_passed": agg["damage_gate_passed"],
        "gate_note": agg["gate_note"],
        "payload_sha256": result.payload_sha256,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "elo_after": updated_elo,
    }


def check_status(
    fighter_name: str,
    state_path: str = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    """Query the league for a fighter's current ELO, rank, and record.

    Args:
        fighter_name: The fighter's name.
        state_path:    Path to the league state JSON file.

    Returns:
        A dict with: name, elo, rank, wins, losses, draws, bouts, win_rate,
        record, policy_path, is_king, created_at (UTC ISO-8601).

    Raises:
        FighterNotRegisteredError: if the fighter is not in the league.
    """
    lg = League(state_path)
    if fighter_name not in lg.fighters:
        raise FighterNotRegisteredError(f"Fighter '{fighter_name}' not found")

    standings = lg.standings()
    row = next((r for r in standings if r["name"] == fighter_name), None)
    if row is None:
        raise FighterNotRegisteredError(f"Fighter '{fighter_name}' not found in standings")

    king = lg.king()
    is_king = king is not None and king.name == fighter_name

    fighter = lg.get_fighter(fighter_name)
    recent_bouts = lg.bout_history(fighter_name)
    last_opponent: Optional[str] = None
    last_result: Optional[str] = None
    last_ts: Optional[str] = None
    if recent_bouts:
        last = recent_bouts[-1]
        last_opponent = last.fighter_b if last.fighter_a == fighter_name else last.fighter_a
        if last.winner is None:
            last_result = "draw"
        else:
            last_result = "win" if last.winner == fighter_name else "loss"
        last_ts = _ts_to_iso(last.timestamp)

    return {
        "name": row["name"],
        "elo": row["elo"],
        "rank": row["rank"],
        "wins": row["wins"],
        "losses": row["losses"],
        "draws": row["draws"],
        "bouts": row["bouts"],
        "win_rate": row["win_rate"],
        "record": row["record"],
        "policy_path": row["policy_path"],
        "is_king": is_king,
        "created_at": _ts_to_iso(fighter.created_at),
        "last_bout": {
            "opponent": last_opponent,
            "result": last_result,
            "timestamp": last_ts,
        } if last_opponent else None,
    }


# --- CLI ---------------------------------------------------------------------


def _cmd_submit(args: argparse.Namespace) -> int:
    try:
        sub = FighterSubmission(
            policy_path=args.policy_path,
            fighter_name=args.name,
            miner_name=args.miner,
            size_limit=int(args.size_limit_mb * 1024 * 1024),
        )
    except PolicyFileError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except PolicySizeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Submission validated:")
    print(f"  Fighter:   {sub.fighter_name}")
    print(f"  Miner:     {sub.miner_name}")
    print(f"  Policy:    {sub.policy_path}")
    print(f"  SHA-256:   {sub.sha256}")
    print(f"  Size:      {sub.size_bytes} bytes ({sub.size_bytes / (1024*1024):.2f} MB)")
    print(f"  Timestamp: {sub.submitted_at} (UTC)")

    try:
        result = submit(sub, state_path=args.state)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nRegistered with league:")
    print(f"  Name:      {result['name']}")
    print(f"  ELO:       {result['elo']:.0f}")
    print(f"  Record:    0-0-0")
    print(f"  Created:   {result['created_at']} (UTC)")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    try:
        status = check_status(args.name, state_path=args.state)
    except FighterNotRegisteredError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    king_tag = " [KING]" if status["is_king"] else ""
    print(f"Fighter: {status['name']}{king_tag}")
    print(f"  ELO:       {status['elo']:.1f}")
    print(f"  Rank:      #{status['rank']}")
    print(f"  Record:    {status['record']}  (W-L-D)")
    print(f"  Win rate:  {status['win_rate']:.3f}")
    print(f"  Bouts:     {status['bouts']}")
    print(f"  Policy:    {status['policy_path']}")
    print(f"  Created:   {status['created_at']} (UTC)")
    if status["last_bout"]:
        lb = status["last_bout"]
        print(f"  Last bout: {lb['result']} vs {lb['opponent']} at {lb['timestamp']} (UTC)")
    else:
        print(f"  Last bout: (none)")
    return 0


def _cmd_challenge(args: argparse.Namespace) -> int:
    try:
        result = evaluate(
            fighter_name=args.name,
            state_path=args.state,
            opponent=args.opponent,
            seeds=args.seeds,
            max_steps=args.max_steps,
            damage_threshold=args.damage_threshold,
            record_bout=not args.dry_run,
        )
    except FighterNotRegisteredError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    winner = result["winner"]
    gate = "PASS" if result["damage_gate_passed"] else "FAIL"
    print(f"Bout result:")
    print(f"  Fighter:   {result['fighter']} vs Opponent: {result['opponent']}")
    print(f"  Winner:    {winner}")
    print(f"  Confidence:{result['confidence']:.4f}")
    print(f"  Seeds:     {result['seeds_won_fighter']}-{result['seeds_won_opponent']}-{result['draws']} (fighter-opponent-draws)")
    print(f"  Damage:    {result['total_damage_fighter']:.1f} / {result['total_damage_opponent']:.1f}")
    print(f"  Gate:      {gate}")
    if result["gate_note"]:
        print(f"  Gate note: {result['gate_note']}")
    print(f"  SHA-256:   {result['payload_sha256'][:24]}...")
    print(f"  Started:   {result['started_at']} (UTC)")
    print(f"  Completed: {result['completed_at']} (UTC)")

    if result["elo_after"]:
        print(f"\nELO after bout:")
        for name, elo in result["elo_after"].items():
            print(f"  {name}: {elo:.1f}")
    elif args.dry_run:
        print(f"\n(dry run -- bout not recorded in league)")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Print the environment interface documentation."""
    print("FightLab Environment Interface")
    print("=" * 60)
    print()
    print(f"Observation space: {OBS_DIM}D (continuous)")
    print(f"Action space:      {ACT_DIM}D (continuous, approx [-1, 1])")
    print()
    print("Action layout (17D):")
    print("  [0:3]   velocity commands (vx, vy, yaw_rate)")
    print("  [3:10]  left arm joint targets (7 DOF)")
    print("  [10:17] right arm joint targets (7 DOF)")
    print()
    print("Observation layout (41D):")
    print("  [0:3]   root linear velocity (x, y, z), body frame")
    print("  [3:6]   root angular velocity (roll, pitch, yaw), body frame")
    print("  [6:9]   projected gravity (3-axis tilt)")
    print("  [9:23]  arm joint positions (14: 7 left + 7 right), radians")
    print("  [23:37] arm joint velocities (14: 7 left + 7 right), rad/s")
    print("  [37:39] opponent relative position (x, y), arena frame")
    print("  [39:41] opponent relative heading (cos, sin of bearing)")
    print()
    print("Balance: maintained by walker.onnx (not controlled by policy).")  # v1 only; v2 uses latent space
    print("Scene:   scene_2bot.xml (two Unitree G1 humanoids).")
    print()
    print("Note: verify per-index semantics against local fight_env.py.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fightlab-sdk",
        description="FightLab Miner SDK -- submit, evaluate, and track fighting policies.",
    )
    p.add_argument(
        "--state",
        default=DEFAULT_STATE_PATH,
        help=f"Path to the league state JSON (default: {DEFAULT_STATE_PATH}).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # submit
    sp = sub.add_parser("submit", help="Validate and register a policy with the league.")
    sp.add_argument("policy_path", help="Path to the policy weights file.")
    sp.add_argument("--name", required=True, help="Fighter name (miner-chosen, unique).")
    sp.add_argument("--miner", required=True, help="Miner name.")
    sp.add_argument(
        "--size-limit-mb", type=float, default=DEFAULT_SIZE_LIMIT / (1024 * 1024),
        help=f"Max policy file size in MB (default: {DEFAULT_SIZE_LIMIT / (1024*1024):.0f}).",
    )
    sp.set_defaults(func=_cmd_submit)

    # status
    sp = sub.add_parser("status", help="Show a fighter's ELO, rank, and record.")
    sp.add_argument("name", help="Fighter name.")
    sp.set_defaults(func=_cmd_status)

    # challenge
    sp = sub.add_parser("challenge", help="Evaluate against the king or a named opponent.")
    sp.add_argument("opponent", help="Opponent name (use 'king' for the current king).")
    sp.add_argument("--name", required=True, help="Your fighter's name.")
    sp.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds for the bout (default: 1 2 3 4 5).",
    )
    sp.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    sp.add_argument("--damage-threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD)
    sp.add_argument(
        "--dry-run", action="store_true",
        help="Run the bout but do not record it in the league.",
    )
    sp.set_defaults(func=_cmd_challenge)

    # info
    sp = sub.add_parser("info", help="Print the environment interface documentation.")
    sp.set_defaults(func=_cmd_info)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Handle 'king' as a synonym for the current king in challenge.
    if args.command == "challenge" and args.opponent.lower() == "king":
        lg = League(args.state)
        king = lg.king()
        if king is None:
            print("ERROR: league is empty, no king to challenge.", file=sys.stderr)
            return 1
        args.opponent = king.name

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
