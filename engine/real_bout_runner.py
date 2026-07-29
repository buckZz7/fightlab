#!/usr/bin/env python3
"""FightLab real bout runner — bridges the eval harness to MuJoCo.

Implements the `BoutRunner` protocol from `eval_harness.py` using the
real `G1FighterEnv` combat environment (Unitree G1 in MuJoCo).

The trained policies (models/fighter_v*.zip) were trained against
`G1FighterEnv` (85-D observation, 17-D action = [arm_14 | walk_3]).
This runner loads that env, drives r1 with policy A externally, sets
policy B as the opponent (r2), runs a single seeded bout to KO or step
cap, and returns a `SeedResult` the harness can aggregate and sign.

Scripted opponents (sandbag / scripted:*) are supported as fallbacks
when a policy path is not a `.zip` file, via `ShadowBoxer`.

Usage as a library (from eval_harness):
    from real_bout_runner import RealBoutRunner
    from eval_harness import EvalHarness, BoutConfig

    cfg = BoutConfig(seeds=[1,2,3], max_steps=1500, damage_threshold=20.0)
    harness = EvalHarness(config=cfg, runner=RealBoutRunner())
    result = harness.run_bout("models/fighter_v4.zip", "models/fighter_v3.zip")

CLI (standalone single-bout smoke test):
    python3 real_bout_runner.py models/fighter_v4.zip models/fighter_v3.zip \
        --seed 42 --max-steps 200
    python3 real_bout_runner.py sandbag sandbag --seed 1 --max-steps 100

Notes:
    - Env: G1FighterEnv(max_steps, randomize=False). No domain
      randomization so results are reproducible per seed.
    - No balance_path: the default PD-to-HOME keeps the robots standing
      (confirmed by deterministic_eval.py on the pod). A balance policy
      is only needed for richer footwork, not for valid bouts.
    - Determinism: np.random and Python random are seeded before each
      bout; env.reset(seed=...) is called; policies run deterministic.
    - Damage: wrist-to-torso contact, gated by relative velocity
      (see G1FighterEnv._update_damage). HP starts at 100, KO at 0 or
      pelvis-z < 0.4 (fall).
    - Scoring: per-step reward from G1FighterEnv._compute_reward
      (RoboStriker weights) accumulated over the bout, plus a terminal
      component. Higher is better.
"""
from __future__ import annotations

import argparse
import math
import os
import random as _pyrandom
import sys
from typing import Any, Optional

import numpy as np

# --- Pod environment bootstrap (scene/mesh paths for G1FighterEnv) ------------
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "G1_SCENE_XML",
    os.path.join(_WORKSPACE, "unitree_mujoco", "unitree_robots", "g1", "scene_29dof.xml"),
)
os.environ.setdefault(
    "G1_MESH_DIR",
    os.path.join(_WORKSPACE, "unitree_mujoco", "unitree_robots", "g1", "meshes"),
)

# Late imports (after sys.path / env vars are set)
from stable_baselines3 import PPO  # noqa: E402

from g1_fighter_env import G1FighterEnv, MAX_HP  # noqa: E402
from bout_fighter import ShadowBoxer  # noqa: E402

# eval_harness lives in /opt/data/fightlab on the orchestrator; on the pod
# it may not be present, so import defensively and fall back to a local
# SeedResult dataclass that matches the protocol shape.
try:
    from eval_harness import SeedResult  # noqa: E402
except Exception:  # pragma: no cover - pod-only path
    from dataclasses import dataclass, field  # noqa: E402
    import time  # noqa: E402

    def _now_utc() -> str:
        t = time.time()
        secs = int(t)
        micros = int((t - secs) * 1_000_000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + f".{micros:06d}Z"

    @dataclass
    class SeedResult:  # type: ignore[no-redef]
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

        def to_dict(self):
            from dataclasses import asdict
            return asdict(self)


# --- Constants ---------------------------------------------------------------
KO_HP = 0.0
FALL_Z = 0.4  # pelvis below this = fall (matches G1FighterEnv)


# --- Policy loading ----------------------------------------------------------


class _NullPolicy:
    """Stand-still policy: always returns zeros (sandbag)."""

    def predict(self, obs, deterministic=True):
        return np.zeros(17, dtype=np.float64), None


def _load_policy(spec: str, env: G1FighterEnv, slot: int) -> Any:
    """Load a fighter policy from a path/spec.

    Supported specs:
      - "<path>.zip"          : stable-baselines3 PPO model
      - "sandbag"             : stand-still (zeros)
      - "scripted:<profile>"  : ShadowBoxer with the given profile
                                (jabbler, defender, balanced, pd, ...)

    `slot` is 0 for r1 (red, driven externally) or 1 for r2 (blue,
    set as env.opponent). ShadowBoxer style is derived from the slot.
    """
    if spec is None or spec == "" or spec == "sandbag":
        return _NullPolicy()
    if spec.endswith(".zip") and os.path.exists(spec):
        return PPO.load(spec)
    if spec.startswith("scripted:"):
        profile = spec.split(":", 1)[1]
        style = "red" if slot == 0 else "blue"
        return ShadowBoxer(env, style=style, profile=profile)
    # If it looks like a path but doesn't exist, fall back to sandbag
    # with a warning so a missing checkpoint doesn't crash a multi-seed run.
    if spec.endswith(".zip"):
        sys.stderr.write(
            f"[real_bout_runner] WARNING: policy not found: {spec}; "
            f"using sandbag\n"
        )
        return _NullPolicy()
    # Unknown spec -> treat as scripted profile name
    style = "red" if slot == 0 else "blue"
    return ShadowBoxer(env, style=style, profile=spec)


# --- The runner --------------------------------------------------------------


class RealBoutRunner:
    """Real BoutRunner: drives G1FighterEnv in MuJoCo.

    Implements the `BoutRunner` protocol from eval_harness.py:
        run_bout(policy_a_path, policy_b_path, policy_a_sha256,
                 policy_b_sha256, seed, max_steps, env_config) -> SeedResult

    The sha256 args are accepted for protocol conformance but not used
    by the runner (the harness handles integrity/hashing; the runner
    only needs the paths and the seed).
    """

    def __init__(self, randomize: bool = False, verbose: bool = False):
        """Args:
            randomize: if True, enable G1FighterEnv domain randomization
                (friction, PD gains). Default False for reproducible eval.
            verbose: print per-step progress to stderr.
        """
        self.randomize = randomize
        self.verbose = verbose

    def run_bout(
        self,
        policy_a_path: str,
        policy_b_path: str,
        policy_a_sha256: str = "",
        policy_b_sha256: str = "",
        seed: int = 42,
        max_steps: int = 1500,
        env_config: Optional[dict] = None,
    ) -> SeedResult:
        """Run a single seeded bout and return a SeedResult."""
        # Seed everything for determinism.
        np.random.seed(seed)
        _pyrandom.seed(seed)

        # Build the env. randomize=False for trustless reproducibility.
        env = G1FighterEnv(max_steps=max_steps, randomize=self.randomize)

        # Load policies. r1 (A) is driven externally via step(); r2 (B)
        # is set as env.opponent so the env calls it inside step().
        policy_a = _load_policy(policy_a_path, env, slot=0)
        policy_b = _load_policy(policy_b_path, env, slot=1)
        env.opponent = policy_b

        # Reset with the seed.
        obs, _ = env.reset(seed=seed)

        # Accumulators
        total_reward_a = 0.0
        # G1FighterEnv only returns r1's reward; r2's reward is the
        # symmetric negative-ish term. We approximate r2's score from
        # its damage and survival (the env doesn't return r2's reward).
        # The harness only uses score_a/score_b as scalar tiebreakers,
        # so an approximation is fine.
        dmg_a_total = 0.0
        dmg_b_total = 0.0
        steps_a_upright = 0
        steps_b_upright = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            # r1 action from policy A
            action_a, _ = policy_a.predict(obs, deterministic=True)
            action_a = np.clip(action_a, -1.0, 1.0)

            # Step the env (this also steps r2 via env.opponent)
            obs, reward, terminated, truncated, info = env.step(action_a)
            total_reward_a += float(reward)

            # Damage bookkeeping (per-step deltas)
            dmg_a_total += float(env._dmg_dealt[0])
            dmg_b_total += float(env._dmg_dealt[1])

            # Survival tracking (pelvis above fall threshold)
            z0 = env._pelvis_z(0)
            z1 = env._pelvis_z(1)
            if z0 >= FALL_Z and env.hp[0] > KO_HP:
                steps_a_upright += 1
            if z1 >= FALL_Z and env.hp[1] > KO_HP:
                steps_b_upright += 1

            if self.verbose and (step % 100 == 0 or terminated or truncated):
                sys.stderr.write(
                    f"  step {step}: hp=[{env.hp[0]:.1f},{env.hp[1]:.1f}] "
                    f"z=[{z0:.3f},{z1:.3f}] dmg_a={dmg_a_total:.2f} "
                    f"dmg_b={dmg_b_total:.2f} term={terminated} "
                    f"trunc={truncated}\n"
                )

            if terminated or truncated:
                break

        # --- Decide the winner -------------------------------------------
        hp_a = float(env.hp[0])
        hp_b = float(env.hp[1])
        z0 = env._pelvis_z(0)
        z1 = env._pelvis_z(1)
        a_down = (hp_a <= KO_HP) or (z0 < FALL_Z)
        b_down = (hp_b <= KO_HP) or (z1 < FALL_Z)

        winner: Optional[str]
        if a_down and not b_down:
            winner = "B"
        elif b_down and not a_down:
            winner = "A"
        elif a_down and b_down:
            # Both down on the same step -> draw (simultaneous KO)
            winner = None
        else:
            # Nobody KO'd -> decision by HP margin. A small margin is a draw.
            margin = hp_a - hp_b
            if abs(margin) < 5.0:
                winner = None
            elif margin > 0:
                winner = "A"
            else:
                winner = "B"

        # terminated flag: True if the bout ended by KO/fall (not step cap)
        ended_by_ko = terminated and not truncated

        # --- Scores -------------------------------------------------------
        # r1 score: accumulated env reward (RoboStriker shaping + terminal).
        score_a = round(total_reward_a, 4)
        # r2 score: approximate symmetric score from damage + survival.
        # (G1FighterEnv only computes r1's reward; this is a scalar
        # tiebreaker only, never used for the win decision.)
        score_b = round(dmg_b_total * 1.0 + 0.01 * steps_b_upright, 4)

        # If the bout went the distance (no KO), cap survival at max_steps.
        if not a_down:
            steps_a_upright = env.step_count
        if not b_down:
            steps_b_upright = env.step_count

        result = SeedResult(
            seed=seed,
            winner=winner,
            damage_dealt_a=round(dmg_a_total, 2),
            damage_dealt_b=round(dmg_b_total, 2),
            steps_survived_a=steps_a_upright,
            steps_survived_b=steps_b_upright,
            score_a=score_a,
            score_b=score_b,
            terminated=ended_by_ko,
        )

        if self.verbose:
            sys.stderr.write(
                f"[bout] seed={seed} winner={winner} "
                f"dmg_a={result.damage_dealt_a} dmg_b={result.damage_dealt_b} "
                f"steps_a={result.steps_survived_a} steps_b={result.steps_survived_b} "
                f"score_a={result.score_a} score_b={result.score_b} "
                f"ko={ended_by_ko}\n"
            )

        # Close the env (frees MuJoCo model/data).
        try:
            env.close()
        except Exception:
            pass

        return result


# --- CLI (standalone single-bout smoke test) ---------------------------------


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="FightLab real bout runner — run a single MuJoCo bout.",
    )
    ap.add_argument("policy_a", help="Path/spec for fighter A (r1). e.g. models/fighter_v4.zip, sandbag, scripted:jabbler")
    ap.add_argument("policy_b", help="Path/spec for fighter B (r2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--verbose", action="store_true", help="Print per-step progress to stderr.")
    a = ap.parse_args(argv)

    runner = RealBoutRunner(verbose=a.verbose)
    result = runner.run_bout(
        policy_a_path=a.policy_a,
        policy_b_path=a.policy_b,
        seed=a.seed,
        max_steps=a.max_steps,
    )
    # Print as JSON-ish to stdout
    print(f"seed={result.seed}")
    print(f"winner={result.winner}")
    print(f"damage_dealt_a={result.damage_dealt_a}")
    print(f"damage_dealt_b={result.damage_dealt_b}")
    print(f"steps_survived_a={result.steps_survived_a}")
    print(f"steps_survived_b={result.steps_survived_b}")
    print(f"score_a={result.score_a}")
    print(f"score_b={result.score_b}")
    print(f"terminated={result.terminated}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
