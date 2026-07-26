"""Motion-match reward: reward the bot for throwing punch SHAPES
that match the G1 Moves reference clips.

The fighting bot learns the *form* of a real jab/low-punch/
rapid-punch from the reference trajectories, not just "extend arm
until contact". This gives clean, human-like strikes instead of
flailing. Pure reward shaping -- no balance model needed
to define it; it activates once the bot can move its arms.

Reference clips (g1moves/motion/*.npz): joint_pos (T,29), fps.
We only use arm joints 15:29 as the target shape.
"""
import os
import numpy as np

MOVE_FILES = {
    "jab": "M_ShortMove12_quickjab",
    "lowpunch": "M_Move2_lowpunch",
    "rapidpunch": "M_Move7_rapidpunch",
}


def load_refs(motion_dir):
    """Load arm (15:29) trajectories + fps from the G1 Moves clips."""
    refs = {}
    for name, fn in MOVE_FILES.items():
        p = os.path.join(motion_dir, "motion", f"{fn}.npz")
        if not os.path.exists(p):
            continue
        m = np.load(p)
        jp = m["joint_pos"].astype(np.float32)
        refs[name] = {
            "arm": jp[:, 15:29].copy(),          # (T, 14) arm joints
            "fps": float(m.get("fps", 60.0)),
            "T": jp.shape[0],
        }
    return refs


def _sample(ref, t, dt):
    """Sample the ref arm pose at time t (seconds)."""
    fi = int(t * ref["fps"]) % ref["T"]
    return ref["arm"][fi]


def motion_match_bonus(refs, name, arm_qpos, t, dt, scale=0.5):
    """Bonus for matching the named punch's arm shape.

    arm_qpos: (14,) current arm joints (15:29).
    Returns 0..scale (higher = closer shape match).
    name=None -> no bonus.
    """
    if name is None or name not in refs:
        return 0.0
    ref = refs[name]
    target = _sample(ref, t, dt)
    # per-joint error, Gaussian falloff
    err = np.mean((arm_qpos - target) ** 2)
    return float(scale * np.exp(-err / 0.10))


class MoveCoach:
    """Tracks which punch is 'active' and feeds motion-match bonus.

    A fight policy emits a discrete punch command (or we cycle). For now
    the coach is fed (name, t_in_punch) each step by the env/rollout
    and returns the bonus. Keeps the reward modular.
    """

    def __init__(self, motion_dir):
        self.refs = load_refs(motion_dir)
        self.active = None
        self.t = 0.0

    def reset(self):
        self.active = None
        self.t = 0.0

    def start(self, name):
        self.active = name
        self.t = 0.0

    def step(self, arm_qpos, dt):
        if self.active is None:
            return 0.0
        self.t += dt
        b = motion_match_bonus(self.refs, self.active, arm_qpos, self.t, dt)
        return b
