# FightLab Miner Spec (v1)

FightLab is a **Gittensor-style** repo: miners submit *challengers*
(humanoid boxing policies) that fight the reigning **KING** in a
regulated soft-square ring. The core hosts the arena + eval; miners
bear the training cost on their own hardware.

This document is the **submission contract**. Match it exactly or
your challenger cannot be evaluated.

> Code reference: `fightlab_contract.py` (import `OBS_DIM`,
> `ACT_DIM`, `OBS_LAYOUT`, `ACT_LAYOUT`, `Challenger`,
> `validate_policy`).

---

## 1. What a Challenger is

A policy `π : OBS(85) → ACT(17)` that controls **one** G1 robot
(r1, red corner) in a 2-bot bout. The opponent (r2, blue corner)
is the KING (or a sandbag during warmup).

**The core provides a FROZEN balance substrate** (`balance_v1`).
Every challenger stands on the same legs. You compete on
**striking + (future) footwork**, not locomotion. This is the
fairness boundary — and it keeps the subnet about *combat*, not
*who has the best balance RL*.

---

## 2. Observation — `OBS_DIM = 85` (float64)

| idx | field | dim | meaning |
|-----|-------|-----|---------|
| 0–3 | `quat` | 4 | torso orientation (x,y,z,w) |
| 4–6 | `angvel` | 3 | torso angular velocity (rad/s) |
| 7–35 | `jrel` | 29 | joint angles − HOME (stand pose) |
| 36–64 | `jvel` | 29 | joint velocities (rad/s) |
| 65 | `hp_self` | 1 | your HP (0–100) |
| 66 | `hp_opp` | 1 | opponent HP (0–100) |
| 67–69 | `rel` | 3 | opponent pelvis − your pelvis |
| 70 | `pelvis_z` | 1 | your pelvis height (m) |
| 71–84 | `residuals` | 14 | smoothed arm-action feedback (env) |

`HOME` is the native G1 stand pose (29 joints). Joint order =
the model's actuated joints (arms at 15–28).

---

## 3. Action — `ACT_DIM = 17` (float64, ∈ [−1, 1])

| idx | field | dim | meaning |
|-----|-------|-----|---------|
| 0–13 | `arm_residual` | 14 | arm-joint offsets. Env smooths (lerp 0.25) × 0.15, adds to HOME[15:29]. |
| 14–16 | `walk_cmd` | 3 | (vx, vy, wz) × (0.5, 0.3, 1.0). **See V1 limits.** |

---

## 4. Training

You may train anywhere, as long as your policy consumes `OBS(85)`
and emits `ACT(17)`:

- **Easiest (recommended):** use our `g1_fighter_env.py` — it
  already produces this exact obs/act. Train with SB3, MJX, or
  your own loop.
- **Custom sim:** replicate the obs/act layout. You must use the
  same ring + rope physics (`g1_arena.build_arena(ring="ropes")`)
  and the same `HOME` stand pose for fairness.

The frozen balance base means **your policy only needs to learn
arms + walk**; legs are handled. This is the whole point — a
miner can field a competitive striker without solving bipedal
locomotion from scratch.

---

## 5. Submission format

Submit **one** of:

1. An SB3 `.zip` with a `predict(obs) → act` (deterministic),
   or
2. A Python module exposing a `Challenger` subclass (`.predict`
   / `.reset`), or
3. A container/endpoint mapping `OBS(85) → ACT(17)`.

Run `fightlab_contract.validate_policy(your_policy)` before
submitting — it checks shape, determinism, and clipping.

---

## 6. Evaluation protocol (ground truth)

The core runs a **deterministic seeded bout**:

```
challenger.reset()
obs = env.reset(seed=ROUND_SEED)
for step in range(MAX_STEPS):
    act = challenger.predict(obs)
    obs, _, term, trunc, info = env.step(act)
    if term or trunc: break
score = BoxingJudge.score(bout_log)   # 10-point must, damage, fouls
```

- Bouts are **best-of-N rounds** vs the fixed KING.
- Ranking: **Elo** on win/loss + damage margin (see `league.py`).
- Anti-gaming: seeded env, held-out KING pool, physical metrics
  (orientation stability, torque smoothness) penalize exploits.

---

## 7. V1 limitations (be aware)

- **Footwork not yet active.** `walk_cmd` is accepted but the
  frozen balance base doesn't consume it → v1 challengers are
  *standing punches*. Real approach/footwork = **v2** (walkable
  balance base).
- **Balance is provided**, not trained by miners (fairness).
- **Single obs/step** (no LSTM). Use `residuals` as feedback.

---

## 8. Roadmap

- **v2:** walkable balance base → `walk_cmd` drives footwork;
  full 29-DoF challenger option.
- **v3:** vision/partial observability; multi-opponent;
  expanded skill library (kicks, per Track B evolution).
- **Scaling:** if you outgrow SB3 throughput, port to **MJX**
  (same MuJoCo engine, GPU-parallel) — keeps this contract.

---

## 9. Why this is the right core design

The Gittensor goal means **you don't train the tournament** —
miners do. Your pod is the *arena*, not the *gym*. A lean
MuJoCo/SB3 core is correct: it hosts bouts (1 env) cheaply, and
the expensive self-play training is distributed across miners.
The 4096-env problem is *their* incentive, not your infra bill.
