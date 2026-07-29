# FightLab v2 — Session Summary (2026-07-27, updated 2026-07-28)

## What was accomplished

### Platform built (local, /opt/data/fightlab/)
- League system (ELO, round-robin, king archive) — league.py
- Trustless eval harness (signed results, multi-seed, damage gate) — engine/eval_harness.py
- Miner SDK + baselines — miner_sdk.py, engine/baselines.py
- Website (black/red, real league standings) — web/index.html
- Gittensor SN74 scaffold (REVIEW.md, EVAL-TRUST.md, CONTRIBUTING.md, CI workflows, .gittensor/config.json)
- RULESET.md (kickboxing: punches + kicks)
- AMP discriminator (99% accuracy, separates boxing from flailing)

### Research completed
- RoboStriker (ICML 2026) — 3-stage pipeline validated (track -> distill -> self-play)
- KungfuBot/PBHC — motion processing, adaptive tracking, codebase cloned
- VLA models — wrong for combat (too slow, manipulation not striking)
- Bittensor subnets — Swarm, Ninja, Oro, Compelle patterns adopted
- Gittensor repos — vanguarstew, sparkinfer, kata patterns adopted
- Cloud providers — RunPod works with Isaac Sim 5.1.0 (not 5.0)

### v2 training pipeline (on 3090 pod)
- Stage 0: Motion library (68 clips, 62 combat, 23-DoF, 30fps) — DONE
- Stage 1: DeepMimic tracker (MJX, 0.92 reward, vision-verified punching/kicking) — DONE
- Stage 2: Latent distillation (decoder+prior, rec_mse=0.007, ONNX exported) — DONE
- Stage 3a: Warmup (MJX 128 envs, 4+ hours, reward 1.30, ZERO hits) — STUCK
- Stage 3b: Self-play — not started (needs warmup hits)
- Render: two G1s standing in arena (fight_real.mp4) — DONE

### Key breakthrough
- Isaac Sim 5.1.0 + Isaac Lab WORK on RunPod (fix: upgrade from 5.0 to 5.1.0)
- Isaac Lab tracker port IN PROGRESS (found humanoid_amp reference task)

## What's blocking us

The MJX warmup (128 envs, hand-rolled JAX PPO) is too slow for RL exploration. After 4+ hours and 150+ iterations, the policy hasn't learned to approach the sandbag. Zero hits.

Isaac Lab at 4096 envs with rsl_rl PPO would give 30x more data per iteration with proven RL infrastructure. This is the path to actual fighting.

## Isaac Lab on RunPod — CONFIRMED WORKING

The fix: Isaac Sim 5.0 fails on RunPod (Vulkan can't enumerate GPU). Isaac Sim 5.1.0 works.
Root cause was NOT a Vulkan driver issue — it was a missing `libvulkan1` library (128KB).
RunPod containers have the full NVIDIA driver stack including Vulkan ICD, just not the loader library.
Fix: `apt-get install libvulkan1` + `export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video`

Verified on 3090: vulkaninfo shows RTX 3090, SimulationApp launches, Isaac Lab CartPole test passes.

## Isaac Lab tracker env — CODE COMPLETE

Files on pod at /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/fightlab_tracker/:
- tracker_env.py — Full DirectRLEnv implementation (DeepMimic tracking)
- tracker_env_cfg.py — G1 config, PD gains, observation/action/reward
- motion_loader.py — Loads 68-clip motion_library.pkl with interpolation
- agents/rsl_rl_ppo_cfg.py — PPO config

Also at /workspace/train_isaac_tracker.py — standalone training script.

Verified: Isaac Sim launched, G1 spawned, motion library loaded.
Two fixes needed before training:
1. rsl_rl version: install rsl-rl-lib==3.0.1 (compatible with isaaclab_rl 0.4.7)
2. G1 USD has 43 joints (includes fingers) — need to add finger actuators or lock them

## 4090 pod setup (COMPLETE)

Pod 6lm1vgpl55e1ss, SSH: root@213.181.111.2 -p 35581
Installed: conda Python 3.11 + Isaac Sim 5.1.0 + Isaac Lab
This is the training machine (same GPU as RoboStriker/KungfuBot).

## Isaac Lab tracker training — WORKING (2026-07-28)

The tracker is now training on the 4090 via Isaac Lab + rsl_rl PPO.

Current status (as of session end):
- Iteration: 230
- Mean reward: 76 (and climbing)
- Throughput: 57,000 FPS (4096 parallel envs)
- Finger joints: locked (43-DoF G1 model, 23-DoF control space)
- rsl-rl-lib==3.0.1 installed and working

This confirms Isaac Lab as the primary training backend. The MJX warmup
was abandoned (no hits after 150+ iterations with hand-rolled JAX PPO).
Isaac Lab's rsl_rl PPO with 4096 envs provides 30x more data per iteration
with proven RL infrastructure.

Stage status:
- Stage 0 (motion library): DONE
- Stage 1 (tracker): IN PROGRESS — Isaac Lab training working, reward climbing
- Stage 2 (distillation): DONE (decoder + prior ONNX exported)
- Stage 3a (warmup): MJX attempt STUCK — needs Isaac Lab port
- Stage 3b (self-play): not started
- Render: fight_real.mp4 (two G1s standing in arena) — DONE

## What to do next session

1. Check tracker training progress (reward should be >90, approaching convergence)
2. Once tracker converges (reward >95, ELR >0.9): export policy
3. Port warmup to Isaac Lab (MJX warmup failed; Isaac Lab is the path)
4. Run warmup with 4096 envs, AMP discriminator, residual latent policy
5. Once warmup produces hits: transition to self-play (LS-NFSP or naive SP)
6. Export final fighting policy to ONNX
7. Register in league, run bouts, update standings
8. Update website with fight video

## Pod details
- 3090: ssh root@213.192.2.84 -p 40164 ($0.50/hr, running)
- 4090: pod 6lm1vgpl55e1ss (stopped, $0.69/hr, restart if needed)
- RunPod API key: rpa_REDACTED-local-only
- Vast.ai account ready as backup

## Key files on the 3090 pod
- /workspace/train_tracker_mjx.py — working MJX tracker
- /workspace/latent_distillation.py + run_latent_distill.py — distillation (done)
- /workspace/latent_ckpts/latent_final.pt — decoder + prior checkpoint
- /workspace/latent_ckpts/decoder_final.onnx — decoder for combat
- /workspace/combat_mjx_core.py — MJX combat env
- /workspace/train_warmup_mjx.py — MJX warmup trainer
- /workspace/render_fight_jax.py — JAX-compatible fight renderer
- /workspace/motion_library.pkl — 68 motion clips
- /workspace/g1_stripped.xml — simplified G1 for MJX
- /workspace/IsaacLab/ — Isaac Lab v2.3.2 source
- /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/ — reference task
