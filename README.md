# FightLab

Self-play boxing reinforcement learning for the Unitree G1 humanoid.

Two G1 robots in a MuJoCo arena. Each is driven by a frozen whole-body
balance policy (keeps them upright) plus a learned arm-residual fight
policy (throws punches, blocks, approaches). Damage comes from
fist-to-body contact force. The goal: learn to fight in sim, then
transfer to real hardware.

![Punch preview](punch_3d.mp4)

## Architecture

```
Frozen balance base (ONNX)    Learned fight layer (PPO)
─────────────────────────     ─────────────────────────
unitree_rl_mjlab              14-dim arm residuals
29-DoF velocity policy        added on top of base targets
keeps robot upright           trained via self-play
```

The balance policy (LocoBase29) is a pretrained whole-body locomotion
controller from unitree_rl_mjlab, exported as ONNX. It runs at 50 Hz
and produces PD torque targets for all 29 joints. The fight policy
outputs 14-dimensional arm residuals (7 per arm) that are added to
the base policy's arm targets. Legs and waist stay frozen — overriding
them breaks balance (validated empirically).

## Pipeline

| Stage | File | Status |
|---|---|---|
| Balance base | `loco_base29.py` | Done — stands 30s+, 30% arm sweep tolerance |
| Bag punch | `g1_punch_env.py` + `train_g1_punch.py` | Done — 0 falls, 0.67 m/s bag velocity |
| Mocap punch | `g1_mocap_punch_env.py` + `train_g1_mocap_punch.py` | Done — 0.72 m/s, 10/10 hits, 0 falls |
| Self-play boxing | `g1_arena.py` + `g1_selfplay_env.py` | Built — env runs, training next |
| Fight eval | `eval_fight_g1.py` | Built — 1v1 eval works |
| League | `league.py` | Needs porting to G1 env |

## Boxing rules (v1)

- **Win by KO:** opponent's pelvis drops below 0.4m
- **Otherwise:** higher HP when the round timer ends
- **Damage:** fist-to-torso contact force, capped per hit, cooldown to prevent hit-spam
- **Actions:** 14 arm joint residuals in [-1, 1]
- **Observations:** own joint state, torso orientation, HP, opponent pose + arm positions (58-dim)

## Usage

```bash
# Environment
pip install mujoco gymnasium stable-baselines3 onnxruntime imageio

# Train the mocap punch policy (bag striking)
python train_g1_mocap_punch.py --timesteps 800000 --out models/g1_mocap_punch

# Train self-play boxing (vs random opponent)
python train_g1_selfplay.py --timesteps 1000000 --out models/boxing_gen1

# Evaluate two policies in a 1v1 bout
python eval_fight_g1.py --red models/boxing_gen1.zip --blue random --matches 20

# Render a fight video
python eval_fight_g1.py --red models/boxing_gen1.zip --blue random --video fight.mp4
```

## Rendering

Local software rendering via OSMesa (no GPU needed):

```bash
export LD_LIBRARY_PATH=/opt/data/osmesa_lib/usr/lib/x86_64-linux-gnu
export MUJOCO_GL=osmesa
```

Dump a trajectory (CPU) then render (CPU or GPU):

```bash
python dump_traj.py --model models/g1_mocap_punch.zip --out traj.npz --seconds 12
python render_traj_3d.py --traj traj.npz --out punch_3d.mp4
```

## Repo layout

```
loco_base29.py            Frozen ONNX balance policy (29-DoF)
g1_punch_env.py           Bag striking env (frozen base + arm residuals)
g1_mocap_punch_env.py     Mocap-imitation punch env (DeepMimic-style)
g1_arena.py               Two-G1 boxing arena builder
g1_selfplay_env.py        Self-play boxing env (PPO training target)
eval_fight_g1.py          1v1 fight evaluation + video render
league.py                 King-of-the-hill league (ELO, lineage)
dump_traj.py              Trajectory dumper for rendering
render_traj_3d.py         Trajectory to 3D video renderer
train_g1_punch.py         Bag punch PPO trainer
train_g1_mocap_punch.py   Mocap punch PPO trainer
mocap/                    Retargeted boxing mocap clips
models/                   Trained PPO checkpoints
docs/                     GitHub Pages site
```

## Roadmap

- **Phase 1 — Sim (now):** balance → bag punch → mocap punch → self-play boxing with ELO-rated kings. Domain randomization from day one.
- **Phase 2 — One real robot:** deploy the king's balance policy on a single Unitree G1.
- **Phase 3 — Real bout:** two-robot autonomous boxing match.
