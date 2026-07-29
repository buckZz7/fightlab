#!/usr/bin/env python3
"""FightLab baseline policies -- starter templates for miners.

Two minimal policies that conform to the FightLab action/observation
interface (41D obs, 17D action). These are intended as scaffolding: copy
one, replace the logic with your trained network, and submit via
miner_sdk.py.

  RandomPolicy   -- uniform random actions within the valid range.
  ScriptedPolicy -- a hand-coded strategy: advance toward the opponent
                    and cycle through guard / jab / cross arm targets.

Both are pure-Python (no numpy required) but will work with numpy arrays
if the environment passes them. The policy interface is a single callable:

    action = policy(obs)   # obs: list[float] of length 41
                          # action: list[float] of length 17

Action space (17D):
  [0:3]   velocity commands  (vx, vy, yaw_rate) in [-1, 1]
  [3:10]  left arm targets   (7 DOF) in [-1, 1]
  [10:17] right arm targets  (7 DOF) in [-1, 1]

Observation space (41D):
  [0:3]   root linear velocity
  [3:6]   root angular velocity
  [6:9]   projected gravity
  [9:23]  arm joint positions (14)
  [23:37] arm joint velocities (14)
  [37:39] opponent relative position (x, y)
  [39:41] opponent relative heading (cos, sin)

Usage
-----
    from baselines import RandomPolicy, ScriptedPolicy

    policy = ScriptedPolicy()
    obs = env.reset()
    action = policy(obs)
    env.step(action)

Or generate a dummy policy file for testing the SDK:

    python baselines.py write /tmp/random_policy.json
    python baselines.py info
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from typing import Any, Optional


# --- Constants ---------------------------------------------------------------

OBS_DIM = 41
ACT_DIM = 17
VEL_DIM = 3      # [vx, vy, yaw_rate]
ARM_DIM = 14     # 7 per arm
LEFT_ARM_SLICE = slice(3, 10)
RIGHT_ARM_SLICE = slice(10, 17)

# Opponent position indices in the observation.
OPP_POS_X = 37
OPP_POS_Y = 38
OPP_HEADING_COS = 39
OPP_HEADING_SIN = 40

# Action value bounds (the environment clips, but we respect them).
ACTION_MIN = -1.0
ACTION_MAX = 1.0


# --- Helpers -----------------------------------------------------------------


def _to_list(obs: Any) -> list[float]:
    """Convert obs to a plain Python list (handles numpy arrays)."""
    if hasattr(obs, "tolist"):
        return list(obs.tolist())
    if isinstance(obs, (list, tuple)):
        return [float(x) for x in obs]
    raise TypeError(f"Cannot convert observation of type {type(obs)} to list")


def _clamp(value: float, lo: float = ACTION_MIN, hi: float = ACTION_MAX) -> float:
    return max(lo, min(hi, value))


# --- Policy interface --------------------------------------------------------


class Policy:
    """Base class for FightLab policies.

    Subclasses implement __call__(obs) -> action.
    """

    name: str = "base"

    def __call__(self, obs: Any) -> list[float]:
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each episode. Override if needed."""
        pass


# --- Random policy -----------------------------------------------------------


class RandomPolicy(Policy):
    """Uniformly random actions within [-1, 1] for all 17 dimensions.

    This is the weakest possible baseline. It exists to:
    - Verify the action/observation plumbing end-to-end.
    - Provide a floor for ELO (a policy worse than random has a bug).
    - Serve as a template for miners to copy and replace with real logic.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def __call__(self, obs: Any) -> list[float]:
        return [self._rng.uniform(ACTION_MIN, ACTION_MAX) for _ in range(ACT_DIM)]

    def reset(self) -> None:
        pass


# --- Scripted policy ---------------------------------------------------------


# Guard position: arms raised in front of the torso.
# 7 DOF per arm: [shoulder_pitch, shoulder_roll, shoulder_yaw,
#                 elbow, wrist_roll, wrist_pitch, gripper]
# Values are normalized targets in [-1, 1].
_GUARD_LEFT = [-0.3, 0.2, 0.0, -0.5, 0.0, -0.3, 0.0]
_GUARD_RIGHT = [-0.3, -0.2, 0.0, -0.5, 0.0, -0.3, 0.0]

# Jab: left arm extends forward.
_JAB_LEFT = [-0.8, 0.1, 0.0, -0.1, 0.0, -0.1, 0.0]

# Cross: right arm extends forward with rotation.
_CROSS_RIGHT = [-0.9, -0.3, 0.2, -0.05, 0.1, -0.1, 0.0]

# Hook: arm comes from the side.
_HOOK_LEFT = [-0.2, 0.6, 0.3, -0.3, 0.2, -0.2, 0.0]


class ScriptedPolicy(Policy):
    """A simple hand-coded fighting policy.

    Strategy:
    1. Face the opponent by commanding a yaw rate proportional to the
       bearing error (from the opponent's relative heading in the obs).
    2. Advance toward the opponent at a fixed forward velocity when
       facing them; strafe slightly otherwise.
    3. Cycle through a guard / jab / cross / hook sequence on a fixed
       step counter, with the guard held between strikes.

    This policy does not learn. It is a readable template that produces
    purposeful-looking behavior and establishes a modest ELO floor.
    """

    name = "scripted"

    # Strike cycle: (arm_state, duration_in_steps)
    _STRIKE_CYCLE = [
        ("jab", 8),
        ("guard", 12),
        ("cross", 8),
        ("guard", 12),
        ("hook", 8),
        ("guard", 16),
    ]

    def __init__(self, forward_speed: float = 0.6, strafe_speed: float = 0.2) -> None:
        self.forward_speed = _clamp(forward_speed)
        self.strafe_speed = _clamp(strafe_speed)
        self._step = 0
        self._cycle_idx = 0
        self._phase_step = 0

    def reset(self) -> None:
        self._step = 0
        self._cycle_idx = 0
        self._phase_step = 0

    def __call__(self, obs: Any) -> list[float]:
        o = _to_list(obs)
        if len(o) < OBS_DIM:
            # Pad or truncate defensively (should not happen in practice).
            if len(o) < ACT_DIM:
                raise ValueError(f"Observation too short: {len(o)} < {ACT_DIM}")
            o = o + [0.0] * (OBS_DIM - len(o))

        # --- Locomotion: face the opponent ---
        opp_cx = o[OPP_HEADING_COS]
        opp_sy = o[OPP_HEADING_SIN]
        bearing = math.atan2(opp_sy, opp_cx)  # radians, 0 = straight ahead
        # Proportional yaw control: turn toward the opponent.
        yaw_rate = _clamp(bearing * 1.5)

        facing = abs(bearing) < 0.5  # roughly facing the opponent
        if facing:
            vx = self.forward_speed
            vy = 0.0
        else:
            vx = self.forward_speed * 0.3
            vy = self.strafe_speed * (1.0 if bearing > 0 else -1.0)

        # --- Arms: cycle through strikes ---
        action = self._arm_action()

        return [
            _clamp(vx), _clamp(vy), _clamp(yaw_rate),
            *action[:ARM_DIM],
        ]

    def _arm_action(self) -> list[float]:
        """Return the 14D arm action for the current strike phase."""
        phase, duration = self._STRIKE_CYCLE[self._cycle_idx]

        if phase == "guard":
            arms = list(_GUARD_LEFT) + list(_GUARD_RIGHT)
        elif phase == "jab":
            arms = list(_JAB_LEFT) + list(_GUARD_RIGHT)
        elif phase == "cross":
            arms = list(_GUARD_LEFT) + list(_CROSS_RIGHT)
        elif phase == "hook":
            arms = list(_HOOK_LEFT) + list(_GUARD_RIGHT)
        else:
            arms = list(_GUARD_LEFT) + list(_GUARD_RIGHT)

        # Advance the strike cycle.
        self._phase_step += 1
        self._step += 1
        if self._phase_step >= duration:
            self._phase_step = 0
            self._cycle_idx = (self._cycle_idx + 1) % len(self._STRIKE_CYCLE)

        return arms


# --- Serialization (for generating dummy policy files) -----------------------


def policy_to_metadata(policy: Policy) -> dict[str, Any]:
    """Return a JSON-serializable metadata dict for a policy.

    This is for generating test/dummy policy files. Real submissions use
    trained weights (ONNX, TorchScript, etc.), not this metadata.
    """
    return {
        "name": policy.name,
        "type": type(policy).__name__,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "action_layout": {
            "velocity": [0, 3],
            "left_arm": [3, 10],
            "right_arm": [10, 17],
        },
        "observation_layout": {
            "root_lin_vel": [0, 3],
            "root_ang_vel": [3, 6],
            "projected_gravity": [6, 9],
            "arm_joint_pos": [9, 23],
            "arm_joint_vel": [23, 37],
            "opponent_pos": [37, 39],
            "opponent_heading": [39, 41],
        },
    }


# --- CLI ---------------------------------------------------------------------


def _cmd_info() -> int:
    print("FightLab Baseline Policies")
    print("=" * 50)
    print()
    print("Available policies:")
    print("  RandomPolicy   -- uniform random actions (ELO floor)")
    print("  ScriptedPolicy -- hand-coded guard/jab/cross/hook cycle")
    print()
    print("Interface:")
    print(f"  obs:   {OBS_DIM}D list[float]")
    print(f"  action: {ACT_DIM}D list[float]")
    print()
    print("Action layout:")
    print(f"  [0:3]   velocity (vx, vy, yaw_rate)")
    print(f"  [3:10]  left arm (7 DOF)")
    print(f"  [10:17] right arm (7 DOF)")
    print()
    print("Observation layout:")
    print(f"  [0:3]   root linear velocity")
    print(f"  [3:6]   root angular velocity")
    print(f"  [6:9]   projected gravity")
    print(f"  [9:23]  arm joint positions (14)")
    print(f"  [23:37] arm joint velocities (14)")
    print(f"  [37:39] opponent relative position (x, y)")
    print(f"  [39:41] opponent relative heading (cos, sin)")
    return 0


def _cmd_write(path: str, policy_name: str) -> int:
    """Write a dummy policy file (metadata JSON) for SDK testing."""
    if policy_name == "random":
        policy: Policy = RandomPolicy(seed=42)
    elif policy_name == "scripted":
        policy = ScriptedPolicy()
    else:
        print(f"Unknown policy '{policy_name}'. Use 'random' or 'scripted'.", file=sys.stderr)
        return 1

    meta = policy_to_metadata(policy)
    # Add a content hash so the file is a realistic "policy artifact".
    content = json.dumps(meta, sort_keys=True).encode("utf-8")
    meta["sha256"] = hashlib.sha256(content).hexdigest()
    meta["size_bytes"] = len(content)

    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {policy_name} policy metadata to {p}")
    print(f"  SHA-256: {meta['sha256']}")
    print(f"  Size:    {meta['size_bytes']} bytes")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        print("Usage: baselines.py [info|write <path> [random|scripted]]")
        return 0

    cmd = argv[0]
    if cmd == "info":
        return _cmd_info()
    elif cmd == "write":
        if len(argv) < 2:
            print("Usage: baselines.py write <path> [random|scripted]", file=sys.stderr)
            return 1
        policy_name = argv[2] if len(argv) > 2 else "random"
        return _cmd_write(argv[1], policy_name)
    else:
        print(f"Unknown command '{cmd}'. Use 'info' or 'write'.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
