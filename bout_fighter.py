"""Backward-compat shim. The ShadowBoxer opponent now lives in combat.py
(consolidated with the rules engine). This module re-exports it so
existing imports (`from bout_fighter import ShadowBoxer`) keep
working — including train_combat.py, which cannot be modified.

Do not add new code here. New code should import directly from combat.
"""
from combat import ShadowBoxer

__all__ = ["ShadowBoxer"]
