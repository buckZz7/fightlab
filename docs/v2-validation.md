# FightLab v2 Validation: Deep Research Report

**Date:** 2026-07-28
**Based on:** Full reading of RoboStriker (arXiv:2601.22517v1), RPG (arXiv:2604.21355v2), LATENT (arXiv:2603.12686), BumbleBee (arXiv:2506.12779), FastTD3 (arXiv:2505.22642), and simulator status research.
**Purpose:** Validate the v2 pipeline before committing. Honest assessment, no hype.

---

## Part 0: RunPod Vulkan Fix (2026-07-28)

### Problem

Isaac Sim 5.0 on RunPod fails to launch: Vulkan cannot enumerate the GPU.
The error appears as a Vulkan initialization failure in the SimulationApp
constructor.

### Root Cause

RunPod containers have the full NVIDIA driver stack including the Vulkan
ICD (Installable Client Driver), but are missing the `libvulkan1` loader
library (128KB). Without the loader, the ICD is never discovered even
though the driver is fully functional.

This is NOT a Vulkan driver issue. The driver works. The loader library
is simply not installed in the RunPod base image.

### Fix

```bash
apt-get install -y libvulkan1
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video
```

### Verification

On a RunPod 3090 pod:
1. `apt-get install libvulkan1` (installs the 128KB loader)
2. `export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video`
3. `vulkaninfo` now shows the RTX 3090
4. Isaac Sim SimulationApp launches successfully
5. Isaac Lab CartPole test passes

### Isaac Sim Version

Isaac Sim 5.1.0 is required (5.0 does not work on RunPod even with the
Vulkan fix). The fix above was verified with 5.1.0.

### Impact

This unblocks Isaac Lab as the primary training backend on RunPod. Both
3090 and 4090 pods are confirmed working. The tracker is now actively
training on the 4090 (iter 230, reward 76, 57K FPS).

---

## Part 1: Full RoboStriker Paper — Details We Hadn't Covered

### 1.1 Training Setup (Confirmed from Appendix F)

| Parameter | Value | Notes |
|---|---|---|
| Simulator | **Isaac Lab** (not IsaacGym) | Built on NVIDIA Omniverse. The v2-pipeline.md says "IsaacGym" — this is wrong. Isaac Lab is the successor. |
| Parallel envs | 4,096 | GPU-parallel |
| GPU | 1x RTX 4090 | Single GPU, not multi-GPU |
| Physics freq | 200 Hz | |
| Control freq | 50 Hz (20ms) | |
| Robot | Unitree G1, 29 DoF | |
| Latent dim | 32 | Hypersphere S^31 |
| NFSP eta | 0.1 | 10% average policy, 90% best response |
| Reservoir K | 10^6 per agent | |
| w_task / w_style | 0.8 / 0.2 | |
| lambda_prior | 0.001 | |
| Hit force threshold | 10 N | |
| Domain randomization | friction, mass, actuator gains | |

**Critical correction:** The paper uses Isaac Lab, not IsaacGym. IsaacGym is officially deprecated by NVIDIA. Isaac Lab is the replacement, built on Isaac Sim/Omniverse. The v2-pipeline.md must be corrected.

### 1.2 Motion Data (Confirmed from Appendix A.1)

- **46 original clips**, ~14 minutes total, captured at 50 Hz with Xsens inertial mocap
- Professional boxers
- Left-right mirrored to double to ~92 clips
- Categories: offensive strikes, defensive maneuvers, footwork, transitions
- Retargeted via GMR framework (arXiv:2510.02252)

**What they DON'T tell you about motion data:**
- No sensitivity analysis to motion quantity. They don't test 5 vs 10 vs 46 clips.
- No discussion of minimum viable motion set.
- No mention of what happens with lower-quality mocap (e.g., video-extracted vs professional inertial).
- No failure modes from data quality issues.

### 1.3 Wall-Clock Training Times — NOT REPORTED

**The paper does not report wall-clock training times for any stage.** This is a significant gap. We have to infer from:
- KungfuBot (comparable tracking task): ~2-3 days for 50K iterations on 1x RTX 4090 with 4096 envs
- Latent distillation: supervised learning, likely hours not days
- Warmup + self-play: unknown, but NFSP with reservoir buffer and dual-policy training is the most complex stage

**Implication:** The v2-pipeline.md estimates (2-3 days tracker, 1 day distillation, 3-5 days combat) are reasonable but unverified. Budget for 2x overage.

### 1.4 Architecture Details — What We Now Know

**32-D latent dimension:** The paper gives NO ablation on latent dimensionality. They chose 32 but don't test 16 or 64. From the t-SNE analysis (Figure 3), the 32-D space shows clear clustering (Move, Strike, Move+Strike). There's no evidence that 16-D would be insufficient or 64-D would be better. **This is an untested hyperparameter.** PULSE (the predecessor work) used 64-D, suggesting 32-D is a deliberate compression, not a ceiling.

**Hypersphere projection:** The paper argues this is critical for game-theoretic convergence (Glicksberg's theorem requires compact strategy sets). They contrast it with PULSE/CALM's unbounded Euclidean latent spaces. The theoretical argument is sound, but there's NO empirical ablation comparing hypersphere vs. bounded box vs. unbounded. The argument is: compact → Nash equilibrium exists → NFSP converges. Without the hypersphere, the strategy space is unbounded and convergence guarantees don't hold.

**Decoder as standalone tracker:** The paper doesn't explicitly discuss this. However, the architecture implies the decoder D_psi(s_prop, z) maps latent codes to actions. If you fix z to the encoder's output for a specific motion, the decoder IS a tracker. The v2-pipeline.md's question "can the decoder be used standalone" — architecturally yes, but it was never tested this way. The decoder is always used with the prior or a residual policy in the paper.

**Residual policy in Stage 3a:** The residual policy pi_theta outputs Delta_z_t over the frozen prior P_xi. The complete action is z_t = Normalize(Delta_z_t + z_p_t). This is NOT optional in their framework — it's how the agent explores combat behaviors while staying close to physically plausible motions. Without it, you'd be sampling directly from the prior, which gives you motion replay but no combat adaptation. **The residual policy is the mechanism that turns a motion library into a combat policy.**

### 1.5 Ablation Deep Dive (Table 3 — the most important data in the paper)

| Method | η_hit (Landing Rate) | ER (Engagement) | What it tests |
|---|---|---|---|
| SP w/o Warmup | **0.050** | **0.120** | No warmup, start self-play from scratch |
| Static-Target Specialist | 0.210 | 0.450 | Warmup only, no self-play |
| PPO-Only | 0.231 | 0.495 | No self-play, fixed opponent |
| Naive SP | 0.350 | 0.580 | Self-play, no opponent averaging |
| Fictitious SP | 0.420 | 0.650 | Self-play, population sampling, no learned average |
| LS-NFSP w/o AMP | 0.490 | 0.720 | Full pipeline, no style reward |
| **LS-NFSP (full)** | **0.685** | **0.824** | Full pipeline |
| 29DoF Action-Space SP | 0.142 | 0.315 | No latent space, raw joint control |

**Key gaps and what they tell us:**

1. **Warmup is the BIGGEST lever by far.** Going from no warmup (0.050) to with warmup (0.685) is a 13.7x improvement. Without warmup, self-play completely fails — η_hit of 0.050 means the robot almost never lands a clean punch. This is the single most important finding.

2. **Latent space is the second biggest lever.** 29DoF raw action space (0.142) vs latent space (0.685) is a 4.8x improvement. This directly explains v1's failure — v1 was essentially the "29DoF Action-Space SP" baseline.

3. **NFSP (opponent averaging) matters significantly.** Naive SP (0.350) → LS-NFSP (0.685) is 2x. The learned average policy prevents strategic cycling.

4. **AMP contributes meaningfully but is not make-or-break.** w/o AMP (0.490) → with AMP (0.685) is 1.4x. Without AMP the robot can approach and fight but strikes are "erratic and ineffective." AMP improves strike quality, not fundamental capability.

5. **Self-play is essential.** PPO-Only against fixed opponent (0.231) vs self-play (0.685) is 3x. "Competitive interaction is the indispensable driver of emergent combat behaviors."

**Can you do tracking + PPO (no latent, no NFSP) and still get fighting?**
The "Static-Target Specialist" baseline IS essentially this — warmup-trained policy against sandbag, no self-play. It achieves η_hit = 0.210. It can hit a static target but has no reactive combat capability. This is roughly where v1 was. **You cannot skip self-play.**

### 1.6 Sim-to-Real Details

The paper claims **zero-shot sim-to-real transfer**. Domain randomization covers:
- Contact friction
- Link masses
- Actuator gains

**What they DON'T tell you:**
- No specific parameter ranges for domain randomization (how much friction variation? mass variation?)
- No analysis of which randomization parameters matter most
- No robustness analysis to sim parameter variations
- No discussion of sim-to-real gap magnitude
- No mention of whether the policy needed any real-world fine-tuning (they claim zero-shot, but no details on how many attempts were needed)

### 1.7 What the Paper Doesn't Tell You (Summary)

1. **No wall-clock training times** for any stage
2. **No latent dimension ablation** (32 vs 16 vs 64)
3. **No motion data sensitivity analysis** (what's the minimum viable motion set?)
4. **No domain randomization parameter ranges** or importance ranking
5. **No failure mode documentation** — what didn't work during development?
6. **No comparison of hypersphere vs. alternative bounding** (box, tanh-squash)
7. **No decoder-as-standalone-tracker** analysis
8. **No minimum viable pipeline** discussion — they present the full pipeline as monolithic

---

## Part 2: Alternative Approaches We Might Have Missed

### 2.1 RPG: Robust Policy Gating (arXiv:2604.21355, June 2026)

**What it is:** Multi-skill fighting with smooth transitions. Trains separate expert policies per skill (punching, kicking, jumping, sword swing), then a gating network blends their outputs.

**Key differences from RoboStriker:**
- **No latent space** — works directly in joint action space (23-DoF)
- **No self-play** — no strategic opponent adaptation
- **No NFSP** — no game-theoretic framework
- Uses policy-transition randomization + temporal randomization for robustness
- Gating network fuses frozen expert outputs with smoothness regularization
- Integrates locomotion with fighting for continuous combat

**What we can learn from RPG:**
- The policy-transition randomization technique is valuable — training experts with random mid-motion truncations forces robustness to interruptions. This could improve our warmup stage.
- Their gating network approach is simpler than a latent space but loses strategic adaptability.
- They demonstrate that **multi-skill fighting without self-play is possible** — but it's scripted combat, not emergent. The user triggers skills; the robot doesn't decide strategy.
- **Not a replacement for RoboStriker** — no autonomous strategic decision-making.

**Relevance to FightLab:** Low for autonomous fighting (no self-play, no strategy), but the transition randomization technique could improve our warmup training robustness.

### 2.2 LATENT: Humanoid Tennis (arXiv:2603.12686, March 2026)

**What it is:** Humanoid tennis from imperfect motion data. Uses latent space + robot-robot self-play for tennis rallies.

**Key parallels to RoboStriker:**
- Same lab (Jingbo Wang is on both papers)
- Latent space approach for skill compression
- Self-play for competitive skill emergence
- Works from imperfect motion data (fragments, not full matches)

**What we can learn:**
- **Imperfect motion data can work.** LATENT explicitly handles fragmented, quasi-realistic motion clips. This suggests RoboStriker's 46 clips might be more than necessary — LATENT achieves results from "imperfect" data.
- **Self-play is validated beyond boxing.** The self-play + latent space pattern generalizes to tennis, suggesting it's a robust paradigm for competitive humanoid skills.
- **Robot-robot self-play works.** They demonstrate robot-robot tennis rallies in simulation.

**Relevance to FightLab:** High. Confirms the latent-space + self-play paradigm. Suggests motion data requirements may be flexible.

### 2.3 BumbleBee: Expert-to-Generalist (arXiv:2506.12779, June 2025)

**What it is:** Motion clustering → train clustered experts → distill into single generalist controller via Transformer.

**Key differences from RoboStriker:**
- No self-play, no competitive framework
- Focus is on general whole-body control, not combat
- Uses KL-divergence distillation (generalist || expert_k(s))
- Transformer-based generalist architecture

**Relevance to FightLab:** Low for combat (no strategic component), but the expert clustering + distillation pattern could inform our Stage 2 if we have many motion types.

### 2.4 Simpler Alternatives — Can We Skip Stages?

**Alternative 1: Tracking + PPO (no latent, no NFSP)**
- This is the "Static-Target Specialist" baseline: η_hit = 0.210
- Can hit a sandbag but cannot fight reactively
- **Verdict: Not viable for fighting. This is v1.**

**Alternative 2: Behavior cloning from mocap, then PPO**
- BC gives you a policy that can reproduce motions but has no combat reward signal
- PPO fine-tuning from BC init would need a combat reward — which requires the warmup stage anyway
- No evidence this is faster than DeepMimic tracking (which is essentially RL-based BC with a tracking reward)
- **Verdict: DeepMimic tracking IS the principled version of this approach. No shortcut.**

**Alternative 3: Tracking policy directly with combat reward (no latent space)**
- This is closest to the "29DoF Action-Space SP" baseline: η_hit = 0.142
- The paper is clear: "decoupling balance maintenance from tactical exploration is a fundamental prerequisite"
- **Verdict: Fails catastrophically. This is the core lesson of the paper.**

**Alternative 4: Tracking + latent + warmup + naive self-play (skip NFSP)**
- Naive SP achieves η_hit = 0.350 (vs 0.685 for full pipeline)
- Gets you ~50% of full performance with significantly less complexity
- No reservoir buffer, no average policy network, no dual-policy training
- **Verdict: Viable as a MINIMUM viable fighter. Will be worse but functional. This is the fastest path to a working fighter.**

### 2.5 Minimum Viable Pipeline

**Shortest path from "we have boxing mocap" to "G1 throws a punch in MuJoCo":**
1. Retarget mocap to G1 (1-2 days with PBHC pipeline)
2. Train DeepMimic tracker with PPO (2-3 days on GPU, 1-2 weeks on CPU)
3. Deploy tracker with a fixed motion command (jab) → G1 throws a punch

**This skips:** Latent distillation, warmup, self-play. You get motion replay, not combat.

**Shortest path from "G1 throws a punch" to "G1 fights another G1":**
1. Add latent distillation (1 day on GPU)
2. Warmup against sandbag (2-3 days on GPU)
3. Naive self-play (skip NFSP, just play against latest opponent snapshot) (3-5 days on GPU)
4. **Total: ~1-2 weeks of training on top of the tracker**

**This gets you:** A basic fighter at ~50% of full pipeline quality (η_hit ~0.35 vs 0.685).

**Full pipeline adds:** NFSP with reservoir + average policy → 2x better fighting quality.

### 2.6 Simulator Status (Updated)

| Simulator | Status | GPU Parallel | FightLab Fit |
|---|---|---|---|
| **IsaacGym** | **DEPRECATED** by NVIDIA | Yes (4096 envs) | Do not use. Replace with Isaac Lab. |
| **Isaac Lab** | Active, supported by NVIDIA | Yes (4096+ envs) | What RoboStriker actually uses. Requires NVIDIA GPU + Omniverse. |
| **MuJoCo (CPU)** | Production-ready, standard | No (CPU multiprocess only) | FightLab's current constraint. 128-256 envs. 3-5x slower. |
| **MuJoCo MJX** | Production-ready (JAX-based) | **Yes** (4096 envs on GPU) | **Game changer.** 2.7M steps/sec on TPU, fast on RTX 4090. Closes gap with Isaac Lab while keeping MuJoCo physics. |
| **Genesis** | v1.0 released May 2026 | Yes | Still maturing. KungfuBot has a backend but production-readiness unclear. |

**Critical finding: MuJoCo MJX enables GPU-parallel training in MuJoCo.** This eliminates the "MuJoCo is 3-5x slower" constraint. MJX runs on JAX and supports 4096+ parallel environments on a single GPU. This is the path for FightLab if we want MuJoCo physics without the CPU bottleneck.

**IsaacGym deprecation handling:** The robotics community has migrated to Isaac Lab. KungfuBot's PBHC repo supports Isaac Lab, IsaacGym, Genesis, and MuJoCo backends. The v2-pipeline.md should be updated to target Isaac Lab or MuJoCo MJX, not IsaacGym.

---

## Part 3: Honest Assessment

### 3.1 Is RoboStriker the Right Approach for FightLab?

**Yes, with caveats.**

**Why yes:**
- The ablation table is the strongest evidence. Every component contributes measurably. The full pipeline achieves η_hit = 0.685; the next best alternative (without any single component) drops to 0.49-0.59. The architecture is not over-engineered — each stage addresses a real failure mode.
- The paper directly explains v1's failure: 29DoF action-space self-play achieves η_hit = 0.142. V1 was essentially this baseline.
- The same lab validated the approach with LATENT (tennis), confirming the latent + self-play paradigm generalizes.
- Zero-shot sim-to-real transfer is demonstrated (though details are thin).

**Caveats:**
- The paper reports NO wall-clock training times. We're estimating 2-3 weeks total on 1x RTX 4090. This could be 2x longer.
- No sensitivity analysis to motion data quantity. We don't know if 10 clips would work or if 46 is the minimum.
- The simulator question is real: RoboStriker uses Isaac Lab (Omniverse), not MuJoCo. FightLab needs MuJoCo for the eval harness. MuJoCo MJX closes the gap but we'd be pioneers running two-agent combat in MJX.
- The paper is from the same lab as BFM and UniTracker. They have deep infrastructure and prior work. We're building from PBHC, which is a different codebase.

### 3.2 Is There a Simpler Path We're Missing?

**The simplest viable path is the "naive self-play" variant:**

```
Stage 1: DeepMimic tracking (required, no shortcut)
Stage 2: Latent distillation (required, the ablation shows raw action space fails)
Stage 3a: Warmup against sandbag (required, the biggest lever in the ablation)
Stage 3b: Naive self-play (simplified — skip NFSP, just play latest opponent)
```

This gives you η_hit ~0.350 instead of 0.685 — roughly half the quality but significantly less complexity:
- No reservoir buffer (K=10^6 samples)
- No average policy network
- No dual-policy training loop
- No anticipatory parameter tuning

**Trade-off:** You get a fighter that can hit and react but will have strategic cycling (rock-paper-scissors loops without convergence). For a first working version, this is acceptable. NFSP can be added later as an upgrade.

**There is no simpler path than this.** Every component below this level has been ablated and fails:
- No latent space → 0.142 (v1's failure)
- No warmup → 0.050 (complete failure)
- No self-play → 0.210 (sandbag-only)

### 3.3 Minimum Viable v2 (Confident In)

**MVP pipeline:**

1. **Motion data:** 20-30 boxing clips (CMU category 13 + augmentation). If 46 is the paper's number, aim for half — LATENT's results suggest imperfect/partial data works.
2. **Retargeting:** PBHC Mink pipeline (proven, has boxing examples)
3. **Tracker:** DeepMimic + KungfuBot adaptive sigma. Train on Isaac Lab or MuJoCo MJX with 4096 envs. Success criterion: ELR > 0.9 on all motions.
4. **Latent distillation:** 32-D hypersphere CVAE. Standard, ~1 day training.
5. **Warmup:** AMP + sandbag. The biggest lever — don't skip. Success criterion: η_hit > 0.3 on sandbag.
6. **Self-play:** Naive self-play first (play against latest opponent snapshot). Success criterion: emergent boxing tactics, η_hit > 0.2 in cross-play.
7. **Upgrade path:** Add NFSP (reservoir + average policy) for convergence. Target η_hit > 0.4.

**What to cut for speed:**
- Skip NFSP initially → saves ~50% of self-play complexity
- Use 20-30 motions instead of 46 → saves motion collection time
- Use MuJoCo MJX if Isaac Lab is unavailable → avoids Omniverse dependency

**What NOT to cut:**
- DeepMimic tracking (everything depends on it)
- Latent distillation (raw action space fails catastrophically)
- Warmup stage (the single biggest lever)
- AMP style reward (1.4x improvement, prevents erratic strikes)

### 3.4 Top 3 Risks and Mitigations

**Risk 1: MuJoCo two-agent combat simulation is too slow or unstable.**
- Running two G1 robots with contact physics in MuJoCo is computationally expensive
- The eval harness uses MuJoCo; if training uses Isaac Lab, there's a sim-to-sim gap
- **Mitigation:** Use MuJoCo MJX for GPU-parallel training. If MJX two-agent combat is unstable, train in Isaac Lab and validate in MuJoCo. The PBHC repo has sim-to-sim deployment code (`humanoidverse/deploy/mujoco.py`).

**Risk 2: Motion data quality/quantity is insufficient.**
- RoboStriker used 46 professional boxing clips with inertial mocap
- CMU category 13 has ~46 clips but they're optical mocap from the 2000s — lower quality
- No sensitivity analysis in the paper; we don't know the minimum
- **Mitigation:** Start with CMU + Mixamo boxing clips. If tracker training fails (ELR < 0.8), augment with custom mocap or more diverse sources. LATENT's success with "imperfect" data is encouraging. Budget for 2x the motion collection time.

**Risk 3: Training time blows past estimates.**
- No wall-clock times in the paper. Our estimates (2-3 weeks total) are based on KungfuBot, not RoboStriker
- NFSP adds significant complexity (dual policy, reservoir buffer, two agents)
- If using MuJoCo CPU (not MJX), multiply by 3-5x
- **Mitigation:** Start with the MVP (naive self-play, no NFSP). This halves the self-play complexity. Use MuJoCo MJX or Isaac Lab for GPU parallelism. Set a timebox: if tracker training takes > 5 days, reassess the approach.

---

## Summary Recommendations

1. **Proceed with the RoboStriker approach** — the ablation evidence is strong and directly explains v1's failure.
2. **Start with the MVP** (naive self-play, skip NFSP) to get a working fighter fast, then upgrade.
3. **Fix the simulator strategy** — target Isaac Lab or MuJoCo MJX, not deprecated IsaacGym or CPU-only MuJoCo.
4. **Don't skip any of: tracking, latent distillation, warmup, AMP.** Every one of these has been ablated and contributes meaningfully.
5. **Budget 2x the estimated time** — the paper doesn't report training times, so our estimates have high uncertainty.
6. **Watch for the three risks** and have mitigation plans ready before starting.

---

## References (New in This Report)

1. **RPG** — Xin et al., "Robust Policy Gating for Smooth Multi-Skill Transitions in Humanoid Fighting", arXiv:2604.21355v2, June 2026.
2. **LATENT** — Zhang et al., "Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data", arXiv:2603.12686, March 2026.
3. **BumbleBee** — Wang et al., "From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots", arXiv:2506.12779, June 2025.
4. **FastTD3** — et al., "FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control", arXiv:2505.22642, May 2025.
5. **MuJoCo MJX** — GPU-parallel MuJoCo via JAX. Documentation: mujoco.readthedocs.io/en/stable/mjx.html
6. **Isaac Lab** — NVIDIA et al., "Isaac Lab: a GPU-accelerated simulation framework for multi-modal robot learning", arXiv:2511.04831. (RoboStriker's actual simulator)
