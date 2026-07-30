#!/usr/bin/env python3
"""FightLab contract bout runner — referees bouts under docs/policy-contract.md.

Drives the standard two-robot MuJoCo scene (mjlab-parity position-actuator
G1) with policy bundles that expose the contract interface:

    Policy.predict(obs_61) -> action_23   (joint position targets, rad)

Training-stack agnostic: a bundle is a directory with manifest.json +
policy.py + weights. policy.py must define `load(path) -> Policy` where
Policy has `.predict(obs)`. Latent decoders, normalizers, LSTMs, etc. live
inside the bundle and are opaque to this runner.

Bout semantics (RULESET): HP 100 each, KO at 0 or pelvis-z < 0.4 (fall =
knockdown -> ref stand-up, bout continues to step cap). Wrist-to-torso
contact gated by relative velocity deals damage. Winner by KO, else HP
margin; <5 HP margin = draw.

Implements the `BoutRunner` protocol from eval_harness.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random as _pyrandom
import sys
from typing import Any, Optional

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCENE = os.environ.get(
    "FIGHTLAB_SCENE_XML",
    os.path.join(_HERE, "assets", "scene_2bot_mjlab.xml"),
)

# Contract constants (docs/policy-contract.md Appendix A/B) — contract v2, 29-DoF
JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
DEFAULT_POSE_29 = np.array([
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,
    0.0, 0.0, 0.0,
    0.20, 0.20, 0.0, 0.9, 0.0, 0.0, 0.0,
    0.20, -0.20, 0.0, 0.9, 0.0, 0.0, 0.0,
], dtype=np.float32)
OBS_DIM = 73
ACT_DIM = 29

KO_HP = 0.0
FALL_Z = 0.4
SPAWN_Z = 0.75
SEPARATION = 0.6
DECIM = 4            # 120 Hz physics / 30 Hz policy
CONTACT_THRESH = 0.35
HIT_VEL_THRESH = 1.0   # m/s projected toward opponent; falling-into peaks ~0.75
DMG_SCALE = 5.0
DMG_CAP = 20.0
SETTLE_GRACE_STEPS = 30  # 1 s at 30 Hz: no damage scored after a (re)spawn


# --- Policy loading ----------------------------------------------------------


class _SandbagPolicy:
    """Stand-still in guard pose (the v0 baseline)."""

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return DEFAULT_POSE_29.copy()


def load_policy(spec: str) -> Any:
    """Load a policy bundle or the 'sandbag' baseline.

    Bundle = directory containing manifest.json + policy.py with load(path).
    """
    if spec in (None, "", "sandbag"):
        return _SandbagPolicy()
    if os.path.isdir(spec):
        policy_py = os.path.join(spec, "policy.py")
        if not os.path.exists(policy_py):
            raise FileNotFoundError(f"bundle missing policy.py: {spec}")
        module_name = "fightlab_bundle_" + os.path.basename(spec.rstrip("/"))
        mod_spec = importlib.util.spec_from_file_location(module_name, policy_py)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return mod.load(spec)
    raise ValueError(
        f"unknown policy spec '{spec}': expected a bundle directory or 'sandbag'"
    )


# --- MuJoCo helpers ----------------------------------------------------------


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    import mujoco

    rot = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, quat_wxyz)
    return rot.reshape(3, 3)


class _Robot:
    """Per-robot address table + obs extraction for one prefix (A_/B_)."""

    def __init__(self, model, prefix: str, opp_prefix: str):
        import mujoco

        self.prefix = prefix
        self.opp_prefix = opp_prefix

        def jid(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

        def bid(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        self.free_jnt = jid(f"{prefix}floating_base_joint")
        self.qpos_adr = model.jnt_qposadr[self.free_jnt]
        self.qvel_adr = model.jnt_dofadr[self.free_jnt]
        self.ctrl_qpos = np.array(
            [model.jnt_qposadr[jid(f"{prefix}{n}")] for n in JOINT_NAMES_29]
        )
        self.ctrl_qvel = np.array(
            [model.jnt_dofadr[jid(f"{prefix}{n}")] for n in JOINT_NAMES_29]
        )
        self.act_adr = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{n}")
             for n in JOINT_NAMES_29]
        )
        self.pelvis = bid(f"{prefix}pelvis")
        self.torso = bid(f"{prefix}torso_link")
        lf = bid(f"{prefix}left_wrist_pitch_link")
        self.lfist = lf if lf >= 0 else bid(f"{prefix}left_wrist_roll_link")
        rf = bid(f"{prefix}right_wrist_pitch_link")
        self.rfist = rf if rf >= 0 else bid(f"{prefix}right_wrist_roll_link")

    def obs(self, model, data, opp: "_Robot") -> np.ndarray:
        quat = data.qpos[self.qpos_adr + 3:self.qpos_adr + 7]
        rot = _quat_to_rot(quat)
        ang_vel_b = data.qvel[self.qvel_adr + 3:self.qvel_adr + 6].astype(np.float32)
        jp = data.qpos[self.ctrl_qpos].astype(np.float32)
        jv = data.qvel[self.ctrl_qvel].astype(np.float32)
        jp_rel = jp - DEFAULT_POSE_29
        off_l = rot.T @ (data.xpos[opp.torso] - data.xpos[self.lfist])
        off_r = rot.T @ (data.xpos[opp.torso] - data.xpos[self.rfist])
        def_l = rot.T @ (data.xpos[opp.pelvis] - data.xpos[self.torso])
        def_r = rot.T @ (data.xpos[opp.pelvis] - data.xpos[self.torso])
        goal = np.concatenate([off_l, off_r, def_l, def_r]).astype(np.float32)
        return np.concatenate([ang_vel_b, jp_rel, jv, goal]).astype(np.float32)

    def pelvis_z(self, data) -> float:
        return float(data.qpos[self.qpos_adr + 2])

    def fist_positions(self, data):
        return data.xpos[self.lfist].copy(), data.xpos[self.rfist].copy()

    def root_velocity(self, model, data):
        import mujoco

        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, v, 0)
        return v[:3].copy()

    def fist_velocities(self, model, data):
        import mujoco

        vl = np.zeros(6)
        vr = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.lfist, vl, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.rfist, vr, 0)
        return vl[:3].copy(), vr[:3].copy()


# --- The runner --------------------------------------------------------------


class ContractBoutRunner:
    """Real BoutRunner over the contract scene (docs/policy-contract.md)."""

    def __init__(self, scene_xml: str = _SCENE, verbose: bool = False,
                 record_path: Optional[str] = None, record_fps: int = 30):
        self.scene_xml = scene_xml
        self.verbose = verbose
        self.record_path = record_path
        self.record_fps = record_fps

    def run_bout(
        self,
        policy_a_path: str,
        policy_b_path: str,
        policy_a_sha256: str = "",
        policy_b_sha256: str = "",
        seed: int = 42,
        max_steps: int = 1500,
        env_config: Optional[dict] = None,
    ):
        import mujoco

        # Local import to avoid a hard dependency at module import time when
        # the harness's own SeedResult is unavailable.
        try:
            from eval_harness import SeedResult
        except Exception:  # pragma: no cover
            from dataclasses import dataclass, field
            import time as _t

            def _now() -> str:
                s = int(_t.time())
                return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(s))

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
                timestamp: str = field(default_factory=_now)

        np.random.seed(seed)
        _pyrandom.seed(seed)

        model = mujoco.MjModel.from_xml_path(
            os.path.join(_HERE, "assets", "scene_2bot_broadcast.xml")
            if self.record_path else self.scene_xml
        )
        model.opt.timestep = 1.0 / 120.0
        data = mujoco.MjData(model)

        A = _Robot(model, "A_", "B_")
        B = _Robot(model, "B_", "A_")

        # Reset both robots to the guard pose, facing each other. Seeded spawn
        # jitter (docs/policy-contract.md §6): per-seed offsets in spawn
        # distance, lateral offset, facing angle, and joint pose make each seed
        # a DIFFERENT but still fully deterministic bout.
        rng = np.random.default_rng(seed)
        sep_jitter = rng.uniform(-0.05, 0.05)
        lat_a, lat_b = rng.uniform(-0.05, 0.05, 2)
        yaw_a, yaw_b = rng.uniform(-0.1745, 0.1745, 2)  # ±10 deg
        joint_jitter_a = rng.uniform(-0.02, 0.02, 29)
        joint_jitter_b = rng.uniform(-0.02, 0.02, 29)

        def _yaw_quat(yaw: float, flip: bool) -> tuple:
            # rotation about Z by yaw, plus 180-deg flip for robot B
            ang = yaw + (np.pi if flip else 0.0)
            return (np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2))

        mujoco.mj_resetData(model, data)
        for rb, sign, quat, lat, jj in (
            (A, -1, _yaw_quat(yaw_a, False), lat_a, joint_jitter_a),
            (B, +1, _yaw_quat(yaw_b, True), lat_b, joint_jitter_b),
        ):
            data.qpos[rb.qpos_adr:rb.qpos_adr + 3] = [
                sign * (SEPARATION + sep_jitter) / 2, lat, SPAWN_Z]
            data.qpos[rb.qpos_adr + 3:rb.qpos_adr + 7] = quat
            data.qpos[rb.ctrl_qpos] = DEFAULT_POSE_29 + jj
        mujoco.mj_forward(model, data)

        policy_a = load_policy(policy_a_path)
        policy_b = load_policy(policy_b_path)

        # Optional video recording (fixed broadcast camera).
        renderer = None
        writer = None
        if self.record_path:
            import imageio
            renderer = mujoco.Renderer(model, height=540, width=960)
            writer = imageio.get_writer(self.record_path, fps=self.record_fps, codec="libx264")

        hp_a, hp_b = 100.0, 100.0
        dmg_a_total = 0.0  # damage A deals to B
        dmg_b_total = 0.0
        steps_a_upright = 0
        steps_b_upright = 0
        ground_a = 0  # steps spent below fall threshold (ground time)
        ground_b = 0
        score_a = 0.0
        terminated = False
        settle_steps = 0  # grace period after spawn: no damage scored

        for step in range(max_steps):
            mujoco.mj_forward(model, data)
            obs_a = A.obs(model, data, B)
            obs_b = B.obs(model, data, A)

            act_a = np.asarray(policy_a.predict(obs_a), dtype=np.float64)
            act_b = np.asarray(policy_b.predict(obs_b), dtype=np.float64)

            data.ctrl[A.act_adr] = act_a
            data.ctrl[B.act_adr] = act_b

            for _ in range(DECIM):
                mujoco.mj_step(model, data)

            if writer is not None:
                renderer.update_scene(data, camera=-1)
                writer.append_data(renderer.render())

            # --- Damage: wrist-torso contact gated by punch velocity projected
            # TOWARD the opponent (matches training hit detection). Skipped for
            # the first 0.5 s (15 steps) after a (re)spawn so the PD settle
            # transient doesn't register phantom hits. ---
            settle_steps += 1
            if settle_steps > SETTLE_GRACE_STEPS:
              for atk, dfn, hp_name in ((A, B, "b"), (B, A, "a")):
                # Upright gate: hits only count if the ATTACKER is standing.
                # A falling body crashing into a static fist is not a strike.
                if atk.pelvis_z(data) < FALL_Z + 0.1:
                    continue
                lf, rf = atk.fist_positions(data)
                lvl, rvl = atk.fist_velocities(model, data)
                # Punch speed = arm-swing velocity (fist minus the attacker's own
                # root velocity) projected toward the defender's torso. Walking
                # into a static fist nets ~0; only a real arm strike scores.
                root_v = atk.root_velocity(model, data)
                torso = data.xpos[dfn.torso]
                best = 0.0
                for fist, fvel in ((lf, lvl), (rf, rvl)):
                    d_vec = torso - fist
                    d = np.linalg.norm(d_vec)
                    if d < CONTACT_THRESH and d > 1e-6:
                        swing = fvel - root_v
                        proj = float(np.dot(swing, d_vec / d))
                        best = max(best, proj)
                if best > HIT_VEL_THRESH:
                    dmg = min(best * DMG_SCALE, DMG_CAP)
                    if hp_name == "b":
                        hp_b = max(KO_HP, hp_b - dmg)
                        dmg_a_total += dmg
                        score_a += dmg
                    else:
                        hp_a = max(KO_HP, hp_a - dmg)
                        dmg_b_total += dmg

            za, zb = A.pelvis_z(data), B.pelvis_z(data)
            a_down = za < FALL_Z
            b_down = zb < FALL_Z
            if not a_down:
                steps_a_upright += 1
            else:
                ground_a += 1
            if not b_down:
                steps_b_upright += 1
            else:
                ground_b += 1

            # v5: no ref stand-ups. A downed robot stays down until it gets up
            # itself or the bout ends. Bout ends only on KO (HP -> 0).
            if hp_a <= KO_HP or hp_b <= KO_HP:
                terminated = True
                break

            if self.verbose and step % 100 == 0:
                sys.stderr.write(
                    f"  step {step}: hp=[{hp_a:.0f},{hp_b:.0f}] z=[{za:.2f},{zb:.2f}] "
                    f"dmg_a={dmg_a_total:.1f} dmg_b={dmg_b_total:.1f}\n"
                )

        # --- Winner (v5 ruleset) ---
        # KO: opponent HP to zero. Decision at the cap: (1) HP margin >= 5,
        # (2) ground-time margin >= 90 steps (3 s), else draw.
        a_out = hp_a <= KO_HP
        b_out = hp_b <= KO_HP
        if a_out and not b_out:
            winner = "B"
        elif b_out and not a_out:
            winner = "A"
        elif a_out and b_out:
            winner = None  # double KO = draw
        else:
            hp_margin = hp_a - hp_b
            if abs(hp_margin) >= 5.0:
                winner = "A" if hp_margin > 0 else "B"
            else:
                gt_margin = ground_b - ground_a  # positive = A spent less time down
                if abs(gt_margin) >= 90:
                    winner = "A" if gt_margin > 0 else "B"
                else:
                    winner = None

        score_b = dmg_b_total

        result = SeedResult(
            seed=seed,
            winner=winner,
            damage_dealt_a=round(dmg_a_total, 2),
            damage_dealt_b=round(dmg_b_total, 2),
            steps_survived_a=steps_a_upright,
            steps_survived_b=steps_b_upright,
            score_a=round(score_a, 4),
            score_b=round(score_b, 4),
            terminated=terminated,
        )
        if self.verbose:
            sys.stderr.write(
                f"[bout] seed={seed} winner={winner} hp=[{hp_a:.0f},{hp_b:.0f}] "
                f"dmg_a={result.damage_dealt_a} dmg_b={result.damage_dealt_b}\n"
            )
        if writer is not None:
            writer.close()
            renderer.close()
        return result


# --- CLI ---------------------------------------------------------------------


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FightLab contract bout runner")
    ap.add_argument("policy_a", help="bundle dir for fighter A, or 'sandbag'")
    ap.add_argument("policy_b", help="bundle dir for fighter B, or 'sandbag'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--scene", default=_SCENE)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    runner = ContractBoutRunner(scene_xml=a.scene, verbose=a.verbose)
    r = runner.run_bout(a.policy_a, a.policy_b, seed=a.seed, max_steps=a.max_steps)
    print(json.dumps({
        "seed": r.seed, "winner": r.winner,
        "damage_dealt_a": r.damage_dealt_a, "damage_dealt_b": r.damage_dealt_b,
        "steps_survived_a": r.steps_survived_a, "steps_survived_b": r.steps_survived_b,
        "score_a": r.score_a, "score_b": r.score_b, "terminated": r.terminated,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
