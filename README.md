# FightLab 🥊

Autonomous humanoid combat league — Unitree G1 robots learn to fight via RL in MuJoCo.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Motion Tracker | `train_motion_tracker.py` | Imitates combat mocap (punches, kicks, blocks) → balance + movement |
| 2. Combat Fine-Tune | `train_combat.py` | RL with combat rewards on top of tracker → learns to fight |
| 3. League | `league.py` | Round-robin bouts with ELO scoring → king of the hill |
| 4. Title Bout | `egl_bout.py` | Challenger vs king → rendered video |
| Evolution | `evolve.py` | Auto-trains new challengers against past kings → always improving |

## Combat Rules

- **Full combat:** punches, kicks, knees, elbows
- **Weapons:** wrists (punches) + ankles (kicks)
- **Damage:** Head = 2x, body = 1x, kicks = 1.5x, leg kicks = 0.5x
- **Scoring:** 10-point must per round (3 rounds × 20s)
- **Win:** KO (HP ≤ 0), fall (pelvis < 0.4m), or decision
- **Draw:** damage gap < 2 HP = draw round; scorecards tied + damage < 1 = draw

## How to Enter

1. Fork this repo
2. Train your fighter:
   ```bash
   python3 train_motion_tracker.py --steps 2000000 --out models/my_tracker
   python3 train_combat.py --tracker models/my_tracker.zip --steps 1000000 --out models/my_fighter
   ```
3. Test locally:
   ```bash
   python3 deterministic_eval.py --fighter models/my_fighter.zip
   ```
4. Submit a PR adding `models/my_fighter.zip`
5. CI auto-runs trustless eval (deterministic, multi-seed, auditable)
6. If you pass the gate (≥1 win, ≥1 damage, won on ≥2 seeds) → title bout vs king

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
g1_fighter_env.py         Combat env (weapons, damage, facing, balance)
combat_rules.py           Rules engine (10-point must, KO, draw, fouls)
bout_fighter.py           ShadowBoxer opponents (punch, kick, dodge, guard)
egl_bout.py               EGL render pipeline (720p, tracking camera)
train_motion_tracker.py  Stage 1: mocap imitation training
train_combat.py           Stage 2: combat fine-tuning
league.py                 Stage 3: round-robin + ELO
evolve.py                 Evolution loop (auto-train challengers)
deterministic_eval.py     Trustless eval (multi-seed, hashed, logged)
ci_gate.py                CI merge gate
amp.py                    Adversarial motion prior discriminator
gen_league_page.py        League standings HTML page
bout_overlay.py           HP bars + timer overlay on videos
loco_base29.py            PD controller + HOME pose
g1_moves_reward.py        MoveCoach (motion reward)
docs/                     GitHub Pages site (standings + bouts)
.github/workflows/        CI: trustless league gate
```
