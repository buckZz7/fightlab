#!/usr/bin/env python3
"""FightLab style battery — assess a fighter against scripted archetypes.

Internal tool (not part of the public league contract). Runs a policy bundle
against a set of scripted opponent styles and produces a scorecard JSON:
per-archetype damage ratio, knockdowns, ground time, win rate.

Archetypes are contract-conformant policies (predict(obs73) -> 29 targets)
with scripted behavior — no training. They expose different weaknesses:
  pressure  — always advances and swings; tests back-foot fighting
  counter   — waits, punishes committed lunges; tests overcommitment
  shell     — pure defense, never initiates; tests guard-breaking
  chaos     — random bursts + retreats; tests robustness
  mirror    — the fighter's own snapshot; tests self-dominance

Usage:
  python style_battery.py <bundle_dir> [--seeds 1 2 3] [--max-steps 300] [--out scorecard.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_bout_runner import ContractBoutRunner, DEFAULT_POSE_29, OBS_DIM, ACT_DIM


# ---------------------------------------------------------------------------
# Scripted archetype policies (contract-conformant)
# ---------------------------------------------------------------------------

def _guard():
    return DEFAULT_POSE_29.copy()


class _Base:
    """Shared helpers for scripted policies."""

    def __init__(self):
        self._t = 0

    def _tick(self):
        self._t += 1
        return self._t

    @staticmethod
    def _opp_dir(obs):
        # goal offense slice: opp torso - ego fists, ego frame (3+3) at 61:67
        off = obs[61:67]
        d = (off[0:3] + off[3:6]) / 2.0
        n = np.linalg.norm(d)
        return d / max(n, 1e-6), n

    def predict(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PressurePolicy(_Base):
    """Always advance + swing. High punch output, no defense."""

    def predict(self, obs):
        t = self._tick()
        a = _guard()
        # legs: forward drive (hip pitch oscillation ~ walking)
        ph = 0.35 * np.sin(t * 0.25)
        a[0] += ph; a[6] += ph          # hip pitch L/R
        a[3] += abs(ph) * 0.5; a[9] += abs(ph) * 0.5  # knees
        # arms: constant alternating hooks (shoulder pitch swings)
        sw = 0.9 * np.sin(t * 0.5)
        a[15] += sw                      # left shoulder pitch
        a[22] -= sw                      # right shoulder pitch
        a[18] = 0.6 + 0.3 * np.sin(t * 0.5)   # left elbow pump
        a[25] = 0.6 - 0.3 * np.sin(t * 0.5)   # right elbow pump
        return a.astype(np.float32)


class CounterPolicy(_Base):
    """Holds guard, then fires a fast cross when the opponent commits
    (opp fist comes close = lunge detected)."""

    def __init__(self):
        super().__init__()
        self._cool = 0

    def predict(self, obs):
        t = self._tick()
        a = _guard()
        _, dist = self._opp_dir(obs)
        self._cool = max(0, self._cool - 1)
        if dist < 0.45 and self._cool == 0:
            self._cool = 12  # ~0.4s commit window
        if self._cool > 0:
            # fast right cross: right shoulder pitch forward + elbow extend
            k = self._cool / 12.0
            a[22] -= 1.1 * k      # right shoulder pitch (punch out)
            a[25] = 0.15          # right elbow near-extended
            a[15] += 0.3          # left guard up
        else:
            # slight retreat drift
            a[0] -= 0.05; a[6] -= 0.05
        return a.astype(np.float32)


class ShellPolicy(_Base):
    """Tight guard, crouched, never initiates. Hard to hit clean."""

    def predict(self, obs):
        a = _guard()
        # crouch: hips + knees bent
        a[0] -= 0.25; a[6] -= 0.25
        a[3] += 0.45; a[9] += 0.45
        a[4] -= 0.15; a[10] -= 0.15
        # guard tight: elbows up, shoulders rolled in
        a[15] += 0.35; a[22] += 0.35
        a[18] = 1.1; a[25] = 1.1   # elbows fully tucked
        return a.astype(np.float32)


class ChaosPolicy(_Base):
    """Random aggressive bursts interleaved with retreats."""

    def __init__(self, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._mode = 0   # 0 idle, 1 burst, 2 retreat
        self._left = 0

    def predict(self, obs):
        t = self._tick()
        if self._left <= 0:
            self._mode = int(self._rng.integers(0, 3))
            self._left = int(self._rng.integers(8, 25))
        self._left -= 1
        a = _guard()
        if self._mode == 1:  # burst: flail forward
            ph = 0.5 * np.sin(t * 0.6)
            a[0] += ph; a[6] += ph
            a[15] += 0.8 * np.sin(t * 0.9)
            a[22] -= 0.8 * np.cos(t * 0.9)
        elif self._mode == 2:  # retreat
            a[0] -= 0.2; a[6] -= 0.2
        return a.astype(np.float32)


ARCHETYPES = {
    "pressure": PressurePolicy,
    "counter": CounterPolicy,
    "shell": ShellPolicy,
    "chaos": ChaosPolicy,
    # "mirror" handled specially (the fighter's own bundle as opponent)
}


# ---------------------------------------------------------------------------
# Bundle loading (mirrors contract_bout_runner.load_policy)
# ---------------------------------------------------------------------------

def load_bundle(spec: str):
    policy_py = os.path.join(spec, "policy.py")
    module_name = "fightlab_battery_" + os.path.basename(spec.rstrip("/"))
    mod_spec = importlib.util.spec_from_file_location(module_name, policy_py)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.load(spec)


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

def run_battery(bundle_dir: str, seeds, max_steps: int, include_mirror: bool = True):
    """Run the fighter against every archetype; return a scorecard dict."""
    fighter = load_bundle(bundle_dir)
    runner = ContractBoutRunner()
    card = {"bundle": bundle_dir, "seeds": seeds, "max_steps": max_steps, "archetypes": {}}

    opponents = dict(ARCHETYPES)
    if include_mirror:
        opponents["mirror"] = None  # sentinel: use the fighter's own bundle

    for name, cls in opponents.items():
        wins = 0; losses = 0; draws = 0
        dmg_for = 0.0; dmg_against = 0.0
        kd_for = 0; kd_against = 0
        gt_self = 0; gt_opp = 0
        for seed in seeds:
            # We drive archetype B by wrapping it as a "policy" object.
            # The runner loads policies by path; to inject a scripted policy we
            # monkey-patch via a temp wrapper dir is overkill — instead run the
            # runner's bout directly with the loaded objects through a shim.
            r = _run_bout_with_objects(runner, fighter, cls, name, seed, max_steps, bundle_dir)
            dmg_for += r["dmg_a"]; dmg_against += r["dmg_b"]
            gt_self += (max_steps - r["steps_a"]); gt_opp += (max_steps - r["steps_b"])
            if r["winner"] == "A":
                wins += 1
            elif r["winner"] == "B":
                losses += 1
            else:
                draws += 1
        n = len(seeds)
        card["archetypes"][name] = {
            "record": f"{wins}-{losses}-{draws}",
            "win_rate": round(wins / n, 3),
            "dmg_ratio": round(dmg_for / max(dmg_against, 1e-6), 2),
            "dmg_for": round(dmg_for, 1),
            "dmg_against": round(dmg_against, 1),
            "ground_time_self": gt_self,
            "ground_time_opp": gt_opp,
        }
        print(f"  {name:9s}  {wins}-{losses}-{draws}  dmg {dmg_for:.0f}:{dmg_against:.0f}  gt {gt_self}:{gt_opp}", flush=True)

    # Overall style profile: weakest archetype = improvement target
    rates = {k: v["win_rate"] for k, v in card["archetypes"].items()}
    card["weakest"] = min(rates, key=rates.get)
    card["strongest"] = max(rates, key=rates.get)
    return card


def _run_bout_with_objects(runner, fighter, arch_cls, name, seed, max_steps, bundle_dir):
    """Run one bout: fighter (A) vs archetype (B).

    The ContractBoutRunner loads policies from paths. For scripted archetypes
    we pass an in-memory policy object by temporarily monkey-patching the
    runner's load_policy. For 'mirror', B is the fighter's own bundle.
    """
    import contract_bout_runner as cbr

    orig_load = cbr.load_policy
    try:
        if name == "mirror":
            cbr.load_policy = lambda spec: load_bundle(bundle_dir) if spec == "__fighter__" else orig_load(spec)
            r = runner.run_bout("__fighter__", "__fighter__", seed=seed, max_steps=max_steps)
        else:
            arch = arch_cls()
            cbr.load_policy = lambda spec: fighter if spec == "__fighter__" else (arch if spec == "__arch__" else orig_load(spec))
            r = runner.run_bout("__fighter__", "__arch__", seed=seed, max_steps=max_steps)
    finally:
        cbr.load_policy = orig_load

    return {
        "winner": r.winner,
        "dmg_a": r.damage_dealt_a,
        "dmg_b": r.damage_dealt_b,
        "steps_a": r.steps_survived_a,
        "steps_b": r.steps_survived_b,
    }


def main():
    ap = argparse.ArgumentParser(description="FightLab style battery")
    ap.add_argument("bundle", help="fighter bundle directory")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"Style battery: {args.bundle}  seeds={args.seeds}  steps={args.max_steps}")
    card = run_battery(args.bundle, args.seeds, args.max_steps, include_mirror=not args.no_mirror)
    card["overall_win_rate"] = round(
        sum(v["win_rate"] for v in card["archetypes"].values()) / len(card["archetypes"]), 3
    )
    print(f"\nOverall win rate: {card['overall_win_rate']}  weakest: {card['weakest']}  strongest: {card['strongest']}")

    out = args.out or os.path.join(args.bundle, "scorecard.json")
    with open(out, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Scorecard -> {out}")


if __name__ == "__main__":
    main()
