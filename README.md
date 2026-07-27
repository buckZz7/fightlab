# FightLab 🥊

Autonomous humanoid combat league — Unitree G1 robots learn to fight via RL in MuJoCo.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Motion Tracker | `train_motion_tracker.py` | Imitates combat mocap (punches, kicks, blocks) → balance + movement |
| 2. Combat Fine-Tune | `train_combat.py` | RL with combat rewards on top of tracker → learns to fight |
| 3. League | `league.py` | Round-robin bouts with ELO scoring → king of the hill |
| 4. Title Bout | `eval.py egl_bout` | Challenger vs king → rendered video |
| Evolution | `evolve.py` | Auto-trains new challengers against past kings → always improving |

## Combat Rules

- **Full combat:** punches, kicks, knees, elbows
- **Weapons:** wrists (punches) + ankles (kicks)
- **Damage:** Head = 2x, body = 1x, kicks = 1.5x, leg kicks = 0.5x
- **Scoring:** 10-point must per round (3 rounds × 20s)
- **Win:** KO (HP ≤ 0), fall (pelvis < 0.4m), or decision
- **Draw:** damage gap < 2 HP = draw round; scorecards tied + damage < 1 = draw

## How to Enter

1. Fork this repo (includes walker, env, eval pipeline, and all past kings' weights)
2. Train your fighter:
   - **From scratch**: train against a sandbag or scripted opponent
   - **Fine-tune a king**: load a past king's weights and improve on top
   - **Train against a king**: use the current king as your sparring partner
3. Test locally with the deterministic eval
4. Submit a PR adding your `models/*.zip`
5. CI auto-runs trustless eval (deterministic, multi-seed, auditable)
6. If you pass the gate (win on 2+ seeds, deal damage) → title bout vs king

## Open Weights Policy

All kings' weights are published in `models/kings/`. Anyone can:
- Download and study them
- Fine-tune them as a starting point
- Use them as training opponents
- The gate prevents copying: you must BEAT the king to take the crown, not match it.

## What's Open Source

- The G1 model + scene (MuJoCo physics)
- The walker (pretrained balance — shared infrastructure)
- The scoring engine (damage detection, 10-point must)
- The eval pipeline (deterministic, multi-seed, hashed, auditable)
- The league system (ELO, king of the hill, evolution loop)
- All past kings' weights

## What Miners Provide

Their trained model weights (`models/*.zip`). That's it. One file.

## Trustless Eval

- Deterministic physics (fixed seeds 42/123/777, no randomization)
- Model SHA256 hash verified
- Per-step bout logs published (HP, damage events, decisions)
- Runs in GitHub Actions (not our infrastructure)
- Anyone can clone + run = identical result

## Tech Stack

- **Physics:** MuJoCo (EGL GPU rendering, 225+ fps)
- **Training:** PPO via Stable Baselines3, 16 parallel envs
- **Mocap:** [exptech/g1-moves](https://huggingface.co/datasets/exptech/g1-moves) (retargeted to G1 29-DoF)
- **Platform:** Unitree G1 (29-DoF, bare-handed wrist-as-fist)
- **AMP:** Adversarial motion prior discriminator for natural movement

## Files

```
street_arena.py           Arena builder (2 G1s, navy checkerboard, tracking camera)
g1_fighter_env.py         Combat env + MoveCoach + loco_base29 (PD/HOME, weapons, damage, facing)
combat.py                 Rules engine (CombatJudge) + ShadowBoxer opponents + bout renderer
eval.py                   Deterministic eval + CI gate + bout overlay + test damage + eval tracker + egl bout
league.py                 Round-robin + ELO + page gen + auto-update + render
train_motion_tracker.py   Stage 1: mocap imitation training (+ AMP discriminator)
train_combat.py           Stage 2: combat fine-tuning (protected — training in progress)
evolve.py                 Evolution loop (auto-train challengers)
docs/                     GitHub Pages site (standings + bouts)
.github/workflows/        CI: trustless league gate
```
