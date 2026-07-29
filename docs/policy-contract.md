# docs/policy-contract.md — FightLab Policy & Bout Contract

**Version:** 2.0
**Last updated:** 2026-07-29T00:00:00Z
**Status:** authoritative for new submissions. Supersedes v1 (61D/23D).

This document defines the ONLY things a miner must conform to for their
policy to fight in the league. FightLab is deliberately training-stack
agnostic: miners may train in Isaac Lab, mjlab, Genesis, or their own
stack. The league cares only about what arrives at the bout — a black box
that maps a standard observation to standard joint position targets.

**v2 changes (29-DoF):** action space grows from 23 to 29 joint targets
(wrists are now controlled, enabling palm-plant get-ups, blocks, and
open-hand work); observation grows from 61 to 73 dims (wrist pos/vel
added). v1 bundles are deprecated; `fightlab-contract=2` is required.

---

## 1. The contract in one sentence

> Given the standard 73-D observation from the standard two-robot MuJoCo
> scene, at 30 Hz, return a 29-D vector of joint position targets.

Everything else — how you trained, what architecture you use, whether a
latent decoder lives inside your bundle — is your business.

---

## 2. The standard scene (physics is neutral)

Bouts are refereed in plain CPU MuJoCo using the standard scene,
`engine/assets/scene_2bot.xml`. This scene is a physical description of
the Unitree G1 robot — NOT a training-stack choice:

- Two G1 29-DoF humanoids, spawned 0.6 m apart, facing each other
  (A at x=-0.3 facing +X, B at x=+0.3 facing -X), pelvis height 0.75 m,
  both in the standard guard pose (Section 4).
- **Actuators:** one MuJoCo `<position>` actuator per controlled joint.
  The position actuator's gain IS the proportional gain kp; the
  derivative gain kd is carried in the joint's `dof_damping`; effort is
  limited by `ctrlrange`. This mirrors how the real G1's onboard PD
  works and is numerically stable under MuJoCo's implicit integrator.
- **Solver:** `implicitfast` integrator, Newton solver, 100 iterations,
  50 line-search iterations. Timestep 1/120 s. Policy/control rate 30 Hz
  (decimation 4).
- Per-joint kp / kd / effort-limit / armature values are pinned in
  Appendix A and match the Unitree G1 hardware datasheet. They are part
  of the contract — miners must not assume different gains.

The scene file is hashed alongside the policy bundle for reproducibility.

## 3. Observation contract (61-D)

`obs = concat(ang_vel_body[3], joint_pos_rel[29], joint_vel[29], goal[12])`

All quantities are for the observing robot ("ego") and expressed in the
ego root (pelvis) body frame unless noted.

| Slice | Field | Dim | Definition |
|-------|-------|-----|------------|
| 0:3   | `ang_vel_body` | 3 | Root angular velocity, body frame (rad/s). |
| 3:32  | `joint_pos_rel` | 29 | Controlled joint positions minus the default guard pose (rad), in `JOINT_NAMES_29` order (Appendix B). |
| 32:61 | `joint_vel` | 29 | Controlled joint velocities (rad/s), same order. |
| 61:67 | `goal_offense` | 6 | Opponent torso position minus ego left/right fist position, in ego frame (3+3, m). |
| 67:73 | `goal_defense` | 6 | Opponent pelvis position minus ego torso position, in ego frame (3+3, m). |

- Fists are the `left/right_wrist_pitch_link` bodies. Torso is
  `torso_link`. Pelvis is the root body.
- The observation is NOT normalized inside the env. Any observation
  normalization is part of the miner's bundle (ship the statistics).
- Both robots receive the same structured observation from their own
  perspective (roles mirrored).

## 4. Action contract (29-D)

`action = 29-D vector of joint position targets (rad), in JOINT_NAMES_29 order.`

- The env applies these directly to the position actuators. All 29
  joints are controlled, including the 6 wrist joints (roll/pitch/yaw
  per hand).
- Targets are clamped to the joint's soft position limits.
- There is no action filtering, smoothing, or torque computation inside
  the env — the position actuators handle PD implicitly. Miners who want
  action smoothing must do it inside their bundle.
- Control rate: 30 Hz. Each action is held for 4 physics steps.

### Default guard pose

Both robots spawn at this pose (rad, `JOINT_NAMES_29` order; wrists at 0):

```
[-0.20, 0.0, 0.0, 0.50, -0.30, 0.0,   # left leg
 -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,   # right leg
  0.0,   0.0, 0.0,                    # waist
  0.20,  0.20, 0.0, 0.90, 0.0, 0.0, 0.0,   # left arm (guard) + wrist
  0.20, -0.20, 0.0, 0.90, 0.0, 0.0, 0.0]   # right arm (guard) + wrist
```

`joint_pos_rel` in the observation is relative to this pose.

---

## 5. Policy bundle format

A submission is a directory:

```
<fighter>/
  manifest.json     # name, version, sha256 of policy file, contract version
  policy.py         # exposes `load(path) -> Policy`
  <weights>         # any file(s) your policy.py loads (pt, onnx, zip, ...)
```

`policy.py` must define:

```python
class Policy:
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (73,) float32. Returns (29,) joint position targets (rad)."""
```

The contract version is `fightlab-contract=2`. The harness validates the
interface, hashes the weights, and pins the scene/obs/action versions
into the signed result.

## 6. Determinism and trust

- The referee env runs with no domain randomization, fixed seeds, fixed
  reset. A bout is fully determined by (policy bundle hash, seed, scene
  hash, contract version).
- **Seeded spawn jitter.** Each seed draws (from `np.random.default_rng(seed)`):
  spawn separation ±0.05 m (shared), lateral offset ±0.05 m per robot,
  facing yaw ±10° per robot, and joint pose offsets ±0.02 rad per
  controlled joint. Same seed → same jitter → same bout, reproducible by
  replay; different seeds → genuinely different trajectories. Ref
  stand-ups after knockdowns use the exact un-jittered spawn.
- Policies are run deterministically at eval (no action sampling). If a
  policy is stochastic internally (e.g. samples a prior), it must use
  its mean/frozen path at eval.
- See [EVAL-TRUST.md](../EVAL-TRUST.md) for the full trust model.

## 7. What is NOT part of the contract

- Training stack (Isaac / mjlab / Genesis / anything).
- Architecture (MLP, LSTM, latent-decoder, diffusion — your choice).
- Internal latent spaces, decoders, priors, normalizers. These live
  inside your bundle and are opaque to the league.
- The warmup/self-play curricula in `docs/v2-pipeline.md`. That doc is
  the reference recipe the first king was trained with, not a rule.

---

## Appendix A — Per-joint actuator constants

| Joints | kp | kd | effort (Nm) | armature |
|--------|----|----|-------------|----------|
| hip_pitch, hip_yaw, waist_yaw | 40.179 | 2.558 | 88 | 0.01018 |
| hip_roll, knee | 99.098 | 6.309 | 139 | 0.02510 |
| ankle_pitch, ankle_roll | 28.501 | 1.814 | 50 | 0.00722 |
| waist_roll, waist_pitch | 28.501 | 1.814 | 50 | 0.00722 |
| shoulder_*, elbow, wrist_roll | 14.251 | 0.907 | 25 | 0.00361 |
| wrist_pitch, wrist_yaw | 16.778 | 1.068 | 5 | 0.00361 |

## Appendix B — JOINT_NAMES_29 order

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

(All suffixed `_joint` in the scene. Order matches the mjlab G1 XML.)
