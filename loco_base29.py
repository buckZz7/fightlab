"""Backward-compat shim. The StandPD/KP/KD/HOME definitions now live
in g1_fighter_env.py (inlined). This module re-exports them so
existing imports (`from loco_base29 import StandPD, KP, KD, HOME`)
keep working — including train_combat.py, which cannot be modified.

Do not add new code here. New code should import directly from
g1_fighter_env.
"""
from g1_fighter_env import StandPD, KP, KD, HOME, SCALE, LocoBase29

__all__ = ["StandPD", "KP", "KD", "HOME", "SCALE", "LocoBase29"]
