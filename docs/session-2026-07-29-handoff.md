# FightLab — Session Handoff (2026-07-29)

## TL;DR for next session
Say: **"Continue FightLab — port warmup env to mjlab and train."**
mjlab is installed and G1-verified on the 4090 pod. The path forward is to port our
two-robot combat warmup env to mjlab (MJWarp physics) so training and MuJoCo eval
share the same engine — eliminating the sim2sim spazzing. Old PhysX IdealPD run is
still training as a fallback/baseline.

## The two blockers this session solved (and how)

### Blocker 1: Isaac Sim rendering on RunPod = DEAD END (confirmed 3x)
- RunPod PyTorch image has a broken NVIDIA Vulkan ICD: `nvidia_icd.json` →
  `libGLX_nvidia.so.0` which exports `vk_icdGetInstanceProcAddr` but NOT
  `vkCreateInstance`. Isaac Sim's ENTIRE renderer is Omniverse RTX (Vulkan-based) —
  confirmed by the official rendering-modes doc; there is NO non-RTX/EGL path.
- Result: "GPU Foundation is not initialized", camera returns solid black frames.
- Host driver is mounted read-only in the container ("Invalid cross-device link"
  on dpkg) — cannot apt-fix the Vulkan ICD. Tested on 2 hosts, 2 driver versions.
- Training works (PhysX uses CUDA directly, not Vulkan); only RENDERING breaks.
- WORKAROUND (current): render in plain MuJoCo + OSMesa on CPU (no GPU, no Vulkan).
- REAL FIX: mjlab/MJWarp has its own GPU batch renderer (ray-traced) that does NOT
  use Isaac Sim's Vulkan path.

### Blocker 2: sim2sim spazzing (PhysX → MuJoCo)
- Symptom: Isaac-trained policy renders in MuJoCo as standing-then-violent-glitching.
- Root cause: PhysX ImplicitActuator runs PD inside the solver at 120Hz
  (solver-damped). MuJoCo explicit PD at 120Hz with kp~100 is under-damped → jitter.
- Diagnostic data: robot falls ~step 40, joint vel explodes to ~30 rad/s, policy
  saturates at max action. min_Az drops to 0.08.
- Anti-jitter stack (applied to render_bout_2bot.py, helps 4x but doesn't fix the fall):
  - `model.opt.integrator = mjINT_IMPLICITFAST` (implicit damping)
  - Move Kd into `model.dof_damping` (implicitly integrated); keep 0.3× explicit Kd
  - Action EMA low-pass alpha=0.6 on the PD target
  - Stiffen contacts: solref=[0.005,1], solimp=[0.9,0.95,0.001,0.5,2], Newton, iter=60
  - Clip torque to Isaac effort limits (88/139/50/25/5 Nm)
  - PD torque recomputed EVERY physics step against HELD target (unitree_rl_gym pattern)
  - Physics rate MUST match training (120Hz). 500Hz made it WORSE (robot fell).
- Research docs: /opt/data/sim2sim_g1_diagnostic_plan.md + spazzing fix research.
- HONEST LIMIT: render-side fixes reduce jitter 4x (jv 31→8) but the policy still
  falls — it's a genuine PhysX↔MuJoCo dynamics gap. The real fix is to train on
  MuJoCo physics (mjlab), not to keep patching the transfer.

## mjlab = the chosen path (solves BOTH blockers)
- mjlab = Isaac Lab API on top of MuJoCo Warp (MJWarp = GPU MuJoCo). By Kevin Zakka
  + Berkeley (Sreenath, Abbeel). 2.7k stars, actively maintained, arXiv 2601.22074.
- Training in MJWarp = MuJoCo physics → our CPU MuJoCo eval is the SAME engine →
  no sim2sim gap. G1 at 4096 envs GPU out of the box.
- Has Mjlab-Tracking-Flat-Unitree-G1 (motion imitation/DeepMimic = our Stage 1)
  and Mjlab-Velocity-Flat-Unitree-G1 (locomotion).
- Also fixes the rendering blocker (own GPU renderer, no Isaac Sim Vulkan).
- Isaac Lab API structure → our env config/reward/obs code ports almost verbatim.
- Caveat: MJWarp does NOT support IMPLICITFAST midpoint integrator (from compat list).

## mjlab install state on the 4090 pod (69.145.85.83:17342)
- Cloned: /workspace/mjlab (main branch)
- Venv: /workspace/mjlab/.venv (uv resolved Python 3.13 — works, >=3.10<3.14)
- uv installed at /root/.local/bin/uv (add to PATH)
- VERIFIED: `uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 512
  --agent.max-iterations 20` completed all 20 iters, checkpoints in
  /workspace/mjlab/logs/rsl_rl/g1_velocity/2026-07-29_03-42-47/ (model_0..19.pt)
- MUST set: MUJOCO_GL=disabled (EGL probe fails headless), WANDB_MODE=disabled
  (no API key, crashes on wandb login otherwise)

## Current PhysX IdealPD run (fallback, still training)
- Config swapped ImplicitActuatorCfg → IdealPDActuatorCfg (explicit PD = matches
  MuJoCo eval) + DR (friction 0.5-1.25, kp/kd ±20% scale via randomize_actuator_gains).
- Fixes applied: EventTermCfg import path (not EventTerm); render camera gated on
  ENABLE_CAMERAS=1 env var (training spawns no camera).
- At last check: iter ~550/5000, reward ~250, hit_rate ~0.67 (landing contact,
  but clinch/grapple not clean punches). Plateauing at contact-not-striking.
- Checkpoints: /workspace/logs/rsl_rl/fightlab_warmup/2026-07-29_02-37-37/model_*.pt
- Launch: cd /workspace && uv/conda python train_isaac_warmup.py --num_envs 4096
  --max_iterations 5000 --headless --latent_ckpt /workspace/latent_final.pt

## Renderers (local, /opt/data)
- /opt/data/render_bout_2bot.py — TWO-ROBOT MuJoCo bout renderer (policy A vs frozen
  sandbag B), real 12D goal obs (fists→torso in A body frame), anti-jitter stack.
  Scene: /opt/data/fightlab_render/scene_2bot.xml (built by build_2bot_scene.py).
- /opt/data/render_bout_mujoco.py — single-robot version (no goal obs).
- Run locally: LD_LIBRARY_PATH=/opt/data/osmesa_lib/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  MUJOCO_GL=osmesa /tmp/mrvenv/bin/python <renderer> --checkpoint <ckpt> --latent
  /opt/data/fightlab_render/latent_final.pt --output <out.mp4> --frames 300
- /tmp/mrvenv = local venv (torch 2.13 CPU, mujoco 3.11, imageio). EPHEMERAL — /tmp
  wipes on restart. Rebuild: uv venv /tmp/mrvenv --python 3.11; uv pip install mujoco
  imageio numpy "imageio[ffmpeg]"; torch --index-url .../whl/cpu
- G1 MuJoCo XML: /opt/data/repos/PBHC/description/robots/g1/g1_29dof_rev_1_0.xml
- latent_final.pt (decoder+prior) + checkpoints synced to /opt/data/fightlab_render/

## Policy architecture (both PhysX and mjlab runs)
rsl_rl PPO actor (61D obs -> 32D delta_z) + frozen DeepMimic decoder+prior:
z = normalize(delta_z + prior_mu(s_prop)); action_23 = decoder(s_prop, z);
target = default_pose + action*0.25 (PD position targets, 23-DoF, JOINT_NAMES_23 order).
Obs: ang_vel(3, body frame) + joint_pos_rel(23) + joint_vel(23) + goal(12) = 61,
normalized with rsl_rl RunningMeanStd (actor_obs_normalizer in checkpoint).

## NEXT STEPS (priority order)
1. Port fightlab warmup env to mjlab: two G1s, combat reward (facing/velocity/dist/
   hit), goal obs, latent decoder+prior residual policy. mjlab DirectRLEnv-style.
2. Train in mjlab at 4096 envs. Because physics = MuJoCo, the MuJoCo eval/renderer
   will be near-1:1 (no spazzing).
3. Render bout from mjlab checkpoint via render_bout_2bot.py — verify NO spazzing.
4. Self-play (two live policies) — sandbag warmup plateaus at contact-not-striking.
5. League eval already runs in MuJoCo (eval_harness.py) — mjlab policies drop in clean.

## Pod details
- 4090 pod: ssh -o StrictHostKeyChecking=no -p 17342 root@69.145.85.83 (secure, $0.69/hr)
- 3090 pod: DEAD (connection refused)
- RunPod API key: rpa_REDACTED-local-only
- RunPod REST: create=POST /v1/pods (fields: name,imageName,gpuTypeIds,gpuCount,
  cloudType,interruptible,supportPublicIp,containerDiskInGb,volumeInGb,volumeMountPath,
  ports,env). terminate=DELETE /v1/pods/{id}. gpuTypes via GraphQL api.runpod.io/graphql.
- 4090 community often FULL; secure works. PyTorch image:
  runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
