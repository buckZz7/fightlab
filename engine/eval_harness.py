#!/usr/bin/env python3
"""FightLab trustless evaluation harness — the referee.

Overview
========
This module is the *referee* for FightLab, the autonomous humanoid combat
platform (Unitree G1 in MuJoCo). Miners submit fighting policies; the
platform must evaluate them **trustlessly** — no party can fake a bout
result. This harness produces cryptographically verifiable bout results
that any third party can independently recompute and check.

Design requirements honoured
----------------------------
1. **Anti-tampering.** Every bout result is a JSON payload whose
   `payload_sha256` is the SHA-256 of the canonical (sorted-key, no-space)
   serialization of that payload. The payload includes the seed list, both
   policy file hashes, the env-config hash, the per-seed results, and the
   aggregate. Any edit to any field invalidates the hash. An optional
   HMAC-SHA256 signature (computed when a signing key is supplied) adds
   authentication on top of integrity.

2. **Multi-seed.** Each bout runs across N seeds (default 5) for
   statistical confidence. Per-seed results and an aggregate are both
   reported.

3. **Size limit.** Submitted policy files must not exceed a configurable
   size cap (default 50 MB). Oversized submissions are rejected before
   any bout is run.

4. **Damage gate.** A fighter must deal at least `damage_threshold`
   total damage across the seed set for a "win" to count as a valid win.
   This prevents passive/avoidance strategies that win on technicalities
   (e.g. surviving longer without engaging). A fighter that "wins" the
   seed majority but fails the damage gate is downgraded to a draw with
   `damage_gate_passed=False` and a `gate_note` explaining why.

5. **UTC timestamps.** Every timestamp is UTC ISO-8601 with a `Z`
   suffix (e.g. `2026-07-27T14:33:01.123456Z`).

Pluggable bout execution
-----------------------
The actual MuJoCo bout execution is a pluggable function injected into
the harness (a `BoutRunner` protocol). This file ships a `MockBoutRunner`
that deterministically synthesizes results from the policy file hashes
and seed, so the harness can be tested end-to-end without MuJoCo. In
production, swap in a real runner that drives `fight_env`.

Usage as a library
------------------
    from eval_harness import EvalHarness, BoutConfig, MockBoutRunner

    cfg = BoutConfig(seeds=[1, 2, 3, 4, 5], max_steps=1000, damage_threshold=50.0)
    harness = EvalHarness(config=cfg, runner=MockBoutRunner())
    result = harness.run_bout("/policies/ironfist.pt", "/policies/stonehand.pt")
    print(result.aggregate.overall_winner)
    print(result.payload_sha256)

    # Verify integrity of a serialized result
    assert EvalHarness.verify(result.to_json())

Usage from CLI
--------------
    # Run a bout with the mock runner and print the signed result JSON
    python eval_harness.py run /policies/a.pt /policies/b.pt --seeds 1 2 3 4 5

    # Run and save to a file
    python eval_harness.py run /policies/a.pt /policies/b.pt -o bout_result.json

    # Verify a saved result file (checks payload hash matches)
    python eval_harness.py verify bout_result.json

    # Run with a signing key (HMAC-SHA256 over the payload hash)
    python eval_harness.py run a.pt b.pt --signing-key secret

Library + CLI, single file, standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


# --- Constants ---------------------------------------------------------------

DEFAULT_SEEDS = 5
DEFAULT_MAX_STEPS = 1000
DEFAULT_DAMAGE_THRESHOLD = 50.0
DEFAULT_SIZE_CAP_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "obs_dim": 41,
    "act_dim": 17,
    "sim_dt": 0.0025,
    "max_steps": DEFAULT_MAX_STEPS,
    "opponent": "sandbag",
}


# --- Timestamps (UTC, ISO-8601 with Z) --------------------------------------


def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string with a Z suffix."""
    # time.time() is UTC on POSIX; format with microseconds for precision.
    t = time.time()
    return _ts_to_iso(t)


def _ts_to_iso(t: float) -> str:
    """Convert a UTC unix timestamp to ISO-8601 with a Z suffix."""
    # gmtime doesn't carry sub-second; format manually to keep microseconds.
    secs = int(t)
    micros = int((t - secs) * 1_000_000)
    tm = time.gmtime(secs)
    return time.strftime("%Y-%m-%dT%H:%M:%S", tm) + f".{micros:06d}Z"


# --- Canonical JSON (for stable hashing) ------------------------------------


def _canonical_json(obj: Any) -> str:
    """Serialize `obj` to a canonical JSON string: sorted keys, no spaces.

    This is the form hashed for integrity. Sorting keys makes the hash
    independent of dict insertion order across Python versions or
    serialization libraries.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(data: bytes | str) -> str:
    """SHA-256 hex digest of bytes or a UTF-8 string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str | Path) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for a file, read in streaming chunks."""
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# --- Data models -------------------------------------------------------------


@dataclass
class BoutConfig:
    """Configuration for a bout (the rules of engagement)."""

    seeds: list[int] = field(default_factory=lambda: list(range(1, DEFAULT_SEEDS + 1)))
    max_steps: int = DEFAULT_MAX_STEPS
    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD
    size_cap_bytes: int = DEFAULT_SIZE_CAP_BYTES
    env_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ENV_CONFIG))

    def env_config_hash(self) -> str:
        """SHA-256 of the canonical env config (max_steps included here too)."""
        # Force max_steps in env_config to match the bout-level max_steps so
        # the env hash reflects what was actually simulated.
        cfg = dict(self.env_config)
        cfg["max_steps"] = self.max_steps
        return _sha256_hex(_canonical_json(cfg))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "max_steps": self.max_steps,
            "damage_threshold": self.damage_threshold,
            "size_cap_bytes": self.size_cap_bytes,
            "env_config": dict(self.env_config),
        }


@dataclass
class PolicyInfo:
    """A submitted policy: path, SHA-256, and byte size."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass
class SeedResult:
    """Result of a single seeded bout.

    - winner: 'A', 'B', or None (draw / no contest).
    - damage_dealt_a / damage_dealt_b: damage each fighter dealt this seed.
    - steps_survived_a / steps_survived_b: steps each fighter stayed upright.
    - score_a / score_b: an env-defined scalar score (higher is better).
    - terminated: True if the bout ended by KO/termination (not by step cap).
    """

    seed: int
    winner: Optional[str]
    damage_dealt_a: float
    damage_dealt_b: float
    steps_survived_a: int
    steps_survived_b: int
    score_a: float
    score_b: float
    terminated: bool
    timestamp: str = field(default_factory=_now_utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoutAggregate:
    """Aggregate across all seeds."""

    overall_winner: Optional[str]  # 'A', 'B', None (draw / no contest)
    confidence: float  # fraction of seeds won by the overall winner, in [0, 1]
    seeds_won_a: int
    seeds_won_b: int
    draws: int
    total_damage_a: float
    total_damage_b: float
    damage_gate_passed: bool
    gate_note: str  # '' if gate passed, else explanation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoutResult:
    """The signed, verifiable bout result."""

    bout_id: str
    config: dict[str, Any]
    env_config_hash: str
    policy_a: dict[str, Any]
    policy_b: dict[str, Any]
    seed_results: list[dict[str, Any]]
    aggregate: dict[str, Any]
    started_at: str
    completed_at: str

    # Integrity + (optional) authentication.
    payload_sha256: str = ""
    hmac_sha256: str = ""
    signed_at: str = ""

    def payload(self) -> dict[str, Any]:
        """The signed payload (everything except the signature fields)."""
        return {
            "bout_id": self.bout_id,
            "config": self.config,
            "env_config_hash": self.env_config_hash,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "seed_results": self.seed_results,
            "aggregate": self.aggregate,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.payload()
        d["payload_sha256"] = self.payload_sha256
        d["hmac_sha256"] = self.hmac_sha256
        d["signed_at"] = self.signed_at
        return d

    def to_json(self) -> str:
        """Pretty JSON with sorted keys (stable, human-auditable)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# --- Pluggable bout runner ---------------------------------------------------


@runtime_checkable
class BoutRunner(Protocol):
    """Protocol for bout executors.

    A concrete runner simulates one seeded bout between two policies and
    returns a `SeedResult`. The real implementation drives `fight_env`;
    `MockBoutRunner` is provided for testing without MuJoCo.
    """

    def run_bout(
        self,
        policy_a_path: str,
        policy_b_path: str,
        policy_a_sha256: str,
        policy_b_sha256: str,
        seed: int,
        max_steps: int,
        env_config: dict[str, Any],
    ) -> SeedResult: ...


# --- Mock bout runner (deterministic, no MuJoCo) -----------------------------


class MockBoutRunner:
    """Deterministic mock bout runner for testing without MuJoCo.

    Synthesizes a SeedResult from the policy hashes and seed so that:
    - Results are reproducible (same inputs -> same outputs).
    - Different policies produce different results.
    - Damage and steps vary per seed.
    - The 'better' policy (by a pseudo-random function of the hashes) tends
      to win, but with per-seed noise so multi-seed aggregation matters.

    This is NOT a real fight — it only exercises the harness plumbing and
    proves the integrity / aggregation / damage-gate logic.
    """

    def run_bout(
        self,
        policy_a_path: str,
        policy_b_path: str,
        policy_a_sha256: str,
        policy_b_sha256: str,
        seed: int,
        max_steps: int,
        env_config: dict[str, Any],
    ) -> SeedResult:
        # Derive two pseudo-strength values in [0, 1) from each policy's hash
        # and the seed, so the same (policy, seed) always yields the same
        # strength but different seeds shift it.
        def _strength(sha: str) -> float:
            h = hashlib.sha256(f"{sha}:{seed}".encode("utf-8")).digest()
            return int.from_bytes(h[:8], "big") / (2**64)

        sa = _strength(policy_a_sha256)
        sb = _strength(policy_b_sha256)

        # Per-seed noise so the stronger fighter doesn't win every seed.
        noise = (int.from_bytes(
            hashlib.sha256(f"{seed}:noise".encode()).digest()[:4], "big"
        ) / 2**32) - 0.5  # in [-0.5, 0.5)

        diff = (sa - sb) + noise  # positive => A is stronger this seed

        if abs(diff) < 0.05:
            winner: Optional[str] = None  # draw
        elif diff > 0:
            winner = "A"
        else:
            winner = "B"

        # Damage: winner deals more; loser still deals some. Scale with
        # strength so two strong policies produce a high-damage bout.
        base_damage = 40.0 + 60.0 * (sa + sb) / 2.0
        if winner == "A":
            dmg_a = base_damage * (0.6 + 0.4 * sa)
            dmg_b = base_damage * (0.2 + 0.2 * sb)
        elif winner == "B":
            dmg_b = base_damage * (0.6 + 0.4 * sb)
            dmg_a = base_damage * (0.2 + 0.2 * sa)
        else:
            dmg_a = base_damage * 0.4
            dmg_b = base_damage * 0.4

        # Steps survived: loser goes down earlier; draws go the distance.
        if winner == "A":
            steps_a = max_steps
            steps_b = int(max_steps * (0.3 + 0.3 * sb))
            terminated = True
        elif winner == "B":
            steps_b = max_steps
            steps_a = int(max_steps * (0.3 + 0.3 * sa))
            terminated = True
        else:
            steps_a = max_steps
            steps_b = max_steps
            terminated = False

        # Score: a scalar combining damage and survival (env-defined).
        score_a = dmg_a + 0.01 * steps_a
        score_b = dmg_b + 0.01 * steps_b

        return SeedResult(
            seed=seed,
            winner=winner,
            damage_dealt_a=round(dmg_a, 2),
            damage_dealt_b=round(dmg_b, 2),
            steps_survived_a=steps_a,
            steps_survived_b=steps_b,
            score_a=round(score_a, 2),
            score_b=round(score_b, 2),
            terminated=terminated,
        )


# --- Aggregation -------------------------------------------------------------


def _aggregate(
    seed_results: list[SeedResult], damage_threshold: float
) -> BoutAggregate:
    """Aggregate per-seed results into an overall verdict with a damage gate.

    Damage gate: the overall winner must have dealt >= damage_threshold
    total damage across the seed set. If they "win" the seed majority but
    failed the gate, the result is downgraded to a draw with a note.
    """
    seeds_won_a = sum(1 for r in seed_results if r.winner == "A")
    seeds_won_b = sum(1 for r in seed_results if r.winner == "B")
    draws = sum(1 for r in seed_results if r.winner is None)
    total_damage_a = round(sum(r.damage_dealt_a for r in seed_results), 2)
    total_damage_b = round(sum(r.damage_dealt_b for r in seed_results), 2)

    # Raw overall winner by seed majority (A wins ties via... no — ties are
    # draws at the seed level; overall is decided by strict majority).
    n = len(seed_results)
    if seeds_won_a > seeds_won_b and seeds_won_a > n / 2:
        raw_winner: Optional[str] = "A"
    elif seeds_won_b > seeds_won_a and seeds_won_b > n / 2:
        raw_winner = "B"
    else:
        raw_winner = None  # no strict majority -> draw/no contest

    # Damage gate: a winner must have dealt >= threshold total damage.
    gate_passed = True
    gate_note = ""
    overall_winner = raw_winner
    if raw_winner == "A" and total_damage_a < damage_threshold:
        gate_passed = False
        gate_note = (
            f"Fighter A won {seeds_won_a}/{n} seeds but dealt "
            f"{total_damage_a:.2f} total damage < threshold "
            f"{damage_threshold:.2f}; downgraded to draw."
        )
        overall_winner = None
    elif raw_winner == "B" and total_damage_b < damage_threshold:
        gate_passed = False
        gate_note = (
            f"Fighter B won {seeds_won_b}/{n} seeds but dealt "
            f"{total_damage_b:.2f} total damage < threshold "
            f"{damage_threshold:.2f}; downgraded to draw."
        )
        overall_winner = None

    # Confidence: fraction of seeds won by the overall winner (0 if draw).
    if overall_winner == "A":
        confidence = seeds_won_a / n
    elif overall_winner == "B":
        confidence = seeds_won_b / n
    else:
        # For a draw, confidence is the max share any fighter had (so a
        # near-draw reads as low confidence).
        confidence = max(seeds_won_a, seeds_won_b) / n if n else 0.0

    return BoutAggregate(
        overall_winner=overall_winner,
        confidence=round(confidence, 4),
        seeds_won_a=seeds_won_a,
        seeds_won_b=seeds_won_b,
        draws=draws,
        total_damage_a=total_damage_a,
        total_damage_b=total_damage_b,
        damage_gate_passed=gate_passed,
        gate_note=gate_note,
    )


# --- Errors ------------------------------------------------------------------


class PolicySizeError(Exception):
    """Raised when a submitted policy exceeds the size cap."""


class PolicyFileError(Exception):
    """Raised when a policy file is missing or unreadable."""


# --- The harness -------------------------------------------------------------


class EvalHarness:
    """The trustless referee. Runs bouts, signs results, verifies them.

    Args:
        config: a `BoutConfig` (rules of engagement).
        runner: a `BoutRunner` (the bout executor). Defaults to
            `MockBoutRunner` so the harness works out-of-the-box for tests.
        signing_key: optional HMAC key. If provided, results carry an
            `hmac_sha256` over the `payload_sha256` for authentication.
            In trustless verification, anyone with the key can confirm
            the result was signed by a holder of the key; without the key
            the payload hash still proves integrity.
    """

    def __init__(
        self,
        config: Optional[BoutConfig] = None,
        runner: Optional[BoutRunner] = None,
        signing_key: Optional[str] = None,
    ):
        self.config = config or BoutConfig()
        self.runner: BoutRunner = runner if runner is not None else MockBoutRunner()
        self.signing_key = signing_key

    # -- policy validation --------------------------------------------------

    def _validate_and_hash_policy(self, path: str | Path) -> PolicyInfo:
        """Check the policy file/bundle exists, is under the size cap, and hash it.

        A directory is a *policy bundle* (docs/policy-contract.md): its hash is
        the SHA-256 over the sorted relative file paths + each file's bytes,
        so the whole bundle (policy.py + weights + manifest) is pinned.
        """
        p = Path(path)
        # Scripted baselines (handled inside the runner, no file to hash).
        if str(path) == "sandbag" or str(path).startswith("scripted:"):
            return PolicyInfo(path=str(path), sha256=_sha256_hex(str(path).encode()), size_bytes=0)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file())
            if not files:
                raise PolicyFileError(f"Policy bundle is empty: {path}")
            h = hashlib.sha256()
            total = 0
            for f in files:
                rel = f.relative_to(p).as_posix().encode()
                h.update(rel)
                data = f.read_bytes()
                h.update(data)
                total += len(data)
            if total > self.config.size_cap_bytes:
                raise PolicySizeError(
                    f"Policy bundle '{path}' is {total} bytes, exceeds cap "
                    f"{self.config.size_cap_bytes} bytes."
                )
            return PolicyInfo(path=str(p), sha256=h.hexdigest(), size_bytes=total)
        if not p.is_file():
            raise PolicyFileError(f"Policy file not found: {path}")
        sha256, size_bytes = _hash_file(p)
        if size_bytes > self.config.size_cap_bytes:
            raise PolicySizeError(
                f"Policy '{path}' is {size_bytes} bytes, exceeds cap "
                f"{self.config.size_cap_bytes} bytes "
                f"({self.config.size_cap_bytes / (1024*1024):.1f} MB)."
            )
        return PolicyInfo(path=str(p), sha256=sha256, size_bytes=size_bytes)

    # -- the main entry -----------------------------------------------------

    def run_bout(
        self,
        policy_a_path: str | Path,
        policy_b_path: str | Path,
    ) -> BoutResult:
        """Run a full multi-seed bout and return a signed, verifiable result.

        Raises:
            PolicyFileError: if a policy file is missing.
            PolicySizeError: if a policy file exceeds the size cap.
        """
        started_at = _now_utc()
        bout_id = uuid.uuid4().hex

        # 1. Validate + hash both policies (size cap enforced here).
        policy_a = self._validate_and_hash_policy(policy_a_path)
        policy_b = self._validate_and_hash_policy(policy_b_path)

        # 2. Run each seed.
        seed_results: list[SeedResult] = []
        for seed in self.config.seeds:
            r = self.runner.run_bout(
                policy_a_path=policy_a.path,
                policy_b_path=policy_b.path,
                policy_a_sha256=policy_a.sha256,
                policy_b_sha256=policy_b.sha256,
                seed=seed,
                max_steps=self.config.max_steps,
                env_config=self.config.env_config,
            )
            seed_results.append(r)

        # 3. Aggregate (with damage gate).
        agg = _aggregate(seed_results, self.config.damage_threshold)

        completed_at = _now_utc()

        result = BoutResult(
            bout_id=bout_id,
            config=self.config.to_dict(),
            env_config_hash=self.config.env_config_hash(),
            policy_a=policy_a.to_dict(),
            policy_b=policy_b.to_dict(),
            seed_results=[r.to_dict() for r in seed_results],
            aggregate=agg.to_dict(),
            started_at=started_at,
            completed_at=completed_at,
        )

        # 4. Sign: payload SHA-256 (integrity) + optional HMAC (auth).
        result.payload_sha256 = _sha256_hex(_canonical_json(result.payload()))
        if self.signing_key is not None:
            result.hmac_sha256 = hmac.new(
                self.signing_key.encode("utf-8"),
                result.payload_sha256.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        result.signed_at = _now_utc()

        return result

    # -- verification (static, no instance state needed) --------------------

    @staticmethod
    def verify(result_json: str | dict, signing_key: Optional[str] = None) -> bool:
        """Verify a serialized bout result's integrity.

        Recomputes the payload SHA-256 from the payload fields and checks
        it matches the stored `payload_sha256`. If `signing_key` is given,
        also verifies the `hmac_sha256` over the payload hash.

        Args:
            result_json: a JSON string or an already-parsed dict.
            signing_key: optional HMAC key to authenticate the signature.

        Returns:
            True if the payload hash (and, if checked, the HMAC) match.

        Raises:
            ValueError: if the result is malformed or verification fails
                (with a message explaining what mismatched).
        """
        if isinstance(result_json, str):
            d = json.loads(result_json)
        else:
            d = dict(result_json)

        required = {
            "bout_id", "config", "env_config_hash", "policy_a", "policy_b",
            "seed_results", "aggregate", "started_at", "completed_at",
            "payload_sha256",
        }
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"Malformed result: missing fields: {sorted(missing)}")

        payload_keys = [
            "bout_id", "config", "env_config_hash", "policy_a", "policy_b",
            "seed_results", "aggregate", "started_at", "completed_at",
        ]
        payload = {k: d[k] for k in payload_keys}
        recomputed = _sha256_hex(_canonical_json(payload))
        if recomputed != d["payload_sha256"]:
            raise ValueError(
                f"Payload hash mismatch: stored={d['payload_sha256']} "
                f"recomputed={recomputed}. Result may have been tampered."
            )

        if signing_key is not None:
            if not d.get("hmac_sha256"):
                raise ValueError("signing_key given but result has no hmac_sha256.")
            expected = hmac.new(
                signing_key.encode("utf-8"),
                d["payload_sha256"].encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, d["hmac_sha256"]):
                raise ValueError("HMAC signature mismatch: result not signed by key holder.")

        return True


# --- CLI ---------------------------------------------------------------------


def _write_policy_file(path: str, content: bytes) -> None:
    """Helper for the self-test: write a fake policy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = BoutConfig(
        seeds=args.seeds,
        max_steps=args.max_steps,
        damage_threshold=args.damage_threshold,
        size_cap_bytes=int(args.size_cap_mb * 1024 * 1024),
    )
    harness = EvalHarness(config=cfg, runner=MockBoutRunner(), signing_key=args.signing_key)
    try:
        result = harness.run_bout(args.policy_a, args.policy_b)
    except (PolicyFileError, PolicySizeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out = result.to_json()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
            fh.write("\n")
        print(f"Signed bout result written to {args.output}", file=sys.stderr)
    else:
        print(out)

    # Print a one-line summary to stderr (keeps stdout pure JSON).
    agg = result.aggregate
    winner = agg["overall_winner"] or "draw"
    print(
        f"[result] winner={winner} confidence={agg['confidence']} "
        f"seeds A/B/draw={agg['seeds_won_a']}/{agg['seeds_won_b']}/{agg['draws']} "
        f"dmg A/B={agg['total_damage_a']:.1f}/{agg['total_damage_b']:.1f} "
        f"gate={'PASS' if agg['damage_gate_passed'] else 'FAIL'} "
        f"hash={result.payload_sha256[:16]}...",
        file=sys.stderr,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    with open(args.file, "r", encoding="utf-8") as fh:
        content = fh.read()
    try:
        ok = EvalHarness.verify(content, signing_key=args.signing_key)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if ok:
        d = json.loads(content)
        agg = d.get("aggregate", {})
        print(
            f"OK: payload_sha256 verified. "
            f"winner={agg.get('overall_winner') or 'draw'} "
            f"gate={'PASS' if agg.get('damage_gate_passed') else 'FAIL'}",
            file=sys.stderr,
        )
        return 0
    print("FAIL: verification returned False.", file=sys.stderr)
    return 1


def _cmd_selftest(args: argparse.Namespace) -> int:
    """End-to-end self-test: create fake policies, run a bout, verify it."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="fightlab_eval_test_"))
    pa = tmp / "policy_a.pt"
    pb = tmp / "policy_b.pt"
    pc = tmp / "policy_c.pt"  # for size-limit test

    # Distinct contents so the mock runner produces distinct strengths.
    pa.write_bytes(b"IRONFIST-WEIGHTS-" * 1024)
    pb.write_bytes(b"STONEHAND-WEIGHTS-" * 1024)

    cfg = BoutConfig(
        seeds=[1, 2, 3, 4, 5],
        max_steps=500,
        damage_threshold=20.0,
        size_cap_bytes=10 * 1024 * 1024,
    )
    harness = EvalHarness(config=cfg, runner=MockBoutRunner(), signing_key="test-key")

    # --- Run a normal bout ---
    result = harness.run_bout(pa, pb)
    assert result.payload_sha256, "payload_sha256 should be set"
    assert result.hmac_sha256, "hmac_sha256 should be set (signing_key given)"
    assert result.aggregate["damage_gate_passed"] in (True, False)

    # --- Verify integrity (no key) ---
    j = result.to_json()
    assert EvalHarness.verify(j), "integrity check should pass"

    # --- Verify HMAC (with key) ---
    assert EvalHarness.verify(j, signing_key="test-key"), "HMAC check should pass"

    # --- Verify HMAC fails with wrong key ---
    try:
        EvalHarness.verify(j, signing_key="wrong-key")
        assert False, "wrong key should have failed"
    except ValueError:
        pass  # expected

    # --- Tamper detection: flip one bit in the aggregate ---
    d = json.loads(j)
    d["aggregate"]["total_damage_a"] = d["aggregate"]["total_damage_a"] + 1000.0
    tampered = json.dumps(d, sort_keys=True)
    try:
        EvalHarness.verify(tampered)
        assert False, "tampered result should fail verification"
    except ValueError:
        pass  # expected

    # --- Size limit: oversized policy is rejected ---
    big = tmp / "big.pt"
    big.write_bytes(b"\x00" * (cfg.size_cap_bytes + 1))
    try:
        harness.run_bout(pa, big)
        assert False, "oversized policy should have been rejected"
    except PolicySizeError:
        pass  # expected

    # --- Determinism: same inputs -> same payload hash ---
    r2 = harness.run_bout(pa, pb)
    # Mock runner is deterministic per (sha, seed), and timestamps differ,
    # so the *payload* hash differs only because of timestamps. Check the
    # per-seed damage is identical instead (the simulated fight is stable).
    damages_1 = [s["damage_dealt_a"] for s in result.seed_results]
    damages_2 = [s["damage_dealt_a"] for s in r2.seed_results]
    assert damages_1 == damages_2, "mock runner should be deterministic"

    print(f"[selftest] OK -- all assertions passed. tmpdir={tmp}", file=sys.stderr)
    print(f"[selftest] sample winner={result.aggregate['overall_winner'] or 'draw'} "
          f"hash={result.payload_sha256[:16]}...", file=sys.stderr)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_harness",
        description="FightLab trustless evaluation harness — the referee.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # run
    sp = sub.add_parser("run", help="Run a multi-seed bout and emit a signed result.")
    sp.add_argument("policy_a", help="Path to fighter A's policy weights.")
    sp.add_argument("policy_b", help="Path to fighter B's policy weights.")
    sp.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(1, DEFAULT_SEEDS + 1)),
        help=f"Seeds to run (default: 1..{DEFAULT_SEEDS}).",
    )
    sp.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    sp.add_argument("--damage-threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD)
    sp.add_argument("--size-cap-mb", type=float, default=DEFAULT_SIZE_CAP_BYTES / (1024 * 1024))
    sp.add_argument("-o", "--output", default=None, help="Write result JSON to this path.")
    sp.add_argument("--signing-key", default=None, help="HMAC signing key (optional).")
    sp.set_defaults(func=_cmd_run)

    # verify
    sp = sub.add_parser("verify", help="Verify a saved bout result's integrity/signature.")
    sp.add_argument("file", help="Path to a bout result JSON file.")
    sp.add_argument("--signing-key", default=None, help="HMAC key to check the signature.")
    sp.set_defaults(func=_cmd_verify)

    # selftest
    sp = sub.add_parser("selftest", help="Run the built-in end-to-end self-test.")
    sp.set_defaults(func=_cmd_selftest)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
