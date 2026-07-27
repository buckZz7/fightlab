"""Backward-compat shim. The MoveCoach / motion-match reward now lives
in g1_fighter_env.py (inlined). This module re-exports it so existing
imports (`from g1_moves_reward import MoveCoach`) keep working —
including train_combat.py, which cannot be modified.

Do not add new code here. New code should import directly from
g1_fighter_env.
"""
from g1_fighter_env import (
    MoveCoach, load_refs, motion_match_bonus, MOVE_FILES,
)

__all__ = ["MoveCoach", "load_refs", "motion_match_bonus", "MOVE_FILES"]
