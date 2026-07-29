# FightLab v2 Training Pipeline

**Version:** 2.1
**Date:** 2026-07-28
**Based on:** RoboStriker (arXiv:2601.22517, ICML 2026), KungfuBot/PBHC (arXiv:2506.12851, NeurIPS 2025), BFM (arXiv:2509.13780), ExBody2 (arXiv:2412.13196)
**Robot:** Unitree G1 (43-DoF with fingers, 23-DoF control space with fingers locked)
**Primary training backend:** Isaac Lab (NVIDIA Omniverse, RTX 4090)
**Backup backend:** MuJoCo MJX (GPU-parallel JAX, validated for tracker stage)

> **Update 2026-07-28:** Isaac Lab is confirmed as the primary training
> backend. The tracker is actively training on a 4090 via Isaac Lab +
> rsl_rl PPO (iter 230, reward 76, 57K FPS, 4096 envs). MJX was used for
> Stage 1 (tracker) initially but the hand-rolled JAX PPO was too slow
> for RL exploration in the warmup stage (Stage 3a). Isaac Lab's rsl_rl
> provides proven RL infrastructure with 30x more data per iteration.
> MJX remains a viable backup for the tracker stage. See
> `docs/v2-validation.md` for the RunPod Vulkan fix that enabled Isaac
> Lab.

---

## Why v1 Failed and What v2 Does Differently

FightLab v1 used PPO directly in a 17-D action space (3 velocity + 14 arm joints) against a static sandbag. The resulting policy stands but never strikes. This is not a hyperparameter problem — it is a structural one.

RoboStriker's ablation (Table 2 in their paper) quantifies this exact failure: 29-DoF action-space self-play achieves an offensive landing rate of 0.142 (vs. 0.685 for the latent-space approach) and a base stability of 0.418 (vs. 0.942). The paper states: "decoupling balance maintenance from tactical exploration is a fundamental prerequisite for competitive learning in high-dimensional humanoid combat."

**v2 adopts the RoboStriker architecture directly:** a three-stage pipeline that decouples low-level physical control from high-level strategy by operating in a learned latent action space. The key insight is that you cannot do RL in joint-space for combat — the action space is too high-dimensional and too unstable. You must first build a library of physically plausible boxing motions, compress them into a bounded latent manifold, and then learn strategy in that compact space.

---

## Architecture Overview

```
Stage 0: Motion Data Collection & Retargeting
    [Human boxing mocap] → SMPL → Filter → Retarget → G1 reference motions

Stage 1: DeepMimic Tracking (Skill Library)
    [G1 reference motions] → PPO tracking policy → pi_trk (29-DoF PD targets)

Stage 2: Latent Space Distillation
    [pi_trk + motion data] → CVAE encoder/decoder/prior → 32-D hypersphere latent

Stage 3: Combat RL (Warmup → Self-Play)
    Stage 3a: AMP warmup vs. sandbag (residual policy in latent space)
    Stage 3b: LS-NFSP self-play (two agents, latent action space)
```

---

## Stage 0: Motion Data Collection & Retargeting

### 0.1 Mocap Data Sources

**Primary: Custom boxing mocap (ideal, matches RoboStriker).**
RoboStriker used 46 clips (~14 min) of professional boxing captured at 50 Hz with an Xsens inertial mocap system, then left-right mirrored to double to ~92 clips. Categories: offensive strikes (jabs, hooks, uppercuts), defensive maneuvers (slips, blocks, weaves), footwork (lateral movement, pivots), and transitional movements.

**Fallback: Existing datasets if custom mocap is unavailable.**

| Source | Format | Boxing content | License | Notes |
|---|---|---|---|---|
| AMASS (amass.is.tue.mpg.de) | SMPL | Limited boxing (some martial arts in CMU, BMLrub subsets) | Research | Unified SMPL, the standard source. Used by KungfuBot. |
| CMU MoCap (mocap.cs.cmu.edu) | BVH | Categories 13 (boxing), 85 (martial arts) | Research-only | 46 boxing clips in category 13. Needs BVH→SMPL conversion. |
| LAFAN1 (github.com/ubisoft/animationeditor) | BVH | No boxing, but locomotion/footwork | Research | Used by KungfuBot for diverse locomotion. |
| Mixamo (mixamo.com) | FBX | Some boxing/martial arts animations | Adobe license | Easy FBX→SMPL via GVHMR or PHC. |
| MMD datasets | VMD | Some boxing motions | Various | Japanese community, quality varies. |

**Recommendation:** Start with CMU category 13 (boxing) converted to SMPL via AMASS pipeline, then supplement with custom mocap or Mixamo boxing clips. KungfuBot's pipeline at `repos/PBHC/motion_source/` handles the conversion.

### 0.2 Retargeting: Human → G1 (29-DoF)

RoboStriker used the Generalized Motion Retargeting (GMR) framework (arXiv:2510.02252, Araujo et al. 2025). KungfuBot offers two alternatives, both available in `repos/PBHC/smpl_retarget/`:

> **Note:** The G1 USD model shipped with Isaac Lab has 43 joints
> (including 14 finger joints across both hands). For kickboxing, the
> finger joints are locked, yielding the same 23-DoF control space used
> by RoboStriker/KungfuBot. The 43-joint model is retained for future
> MMA/grappling expansion. See `docs/combat-scope-analysis.md` for the
> full decision.

**Option A: Mink IK-based retargeting (recommended, what KungfuBot uses in production)**
- Repository: `repos/PBHC/smpl_retarget/mink_retarget/`
- Based on Mink (github.com/kevinzakka/mink), a differentiable IK solver
- Formulates a differentiable optimization matching end-effector trajectories while respecting G1 joint limits
- Command: `python mink_retarget/convert_fit_motion.py <motion_folder> --correct`
- Output: `.pkl` files with retargeted joint trajectories for G1

**Option B: PHC gradient-based optimization**
- Repository: `repos/PBHC/smpl_retarget/phc_retarget/`
- Requires SMPL model files (download from smpl.is.tue.mpg.de)
- Command: `python phc_retarget/fit_smpl_motion.py robot=unitree_g1_29dof_anneal_23dof +motion=<motion_folder>`

**Key retargeting considerations for G1:**
- G1 has 29 DoF (legs: 12, waist: 3, arms: 10, hands: 4 — but RoboStriker and KungfuBot typically lock the 6 wrist DoF, using 23 DoF for control with 29 DoF for the full model)
- Scale correction: human to G1 height/limb ratio must be adjusted
- Joint limit enforcement: G1's shoulder and hip ranges differ from human
- Contact mask computation (KungfuBot: `repos/PBHC/motion_source/count_pkl_contact_mask.py`)

### 0.3 Motion Filtering (KungfuBot's physics-based filter)

Before retargeting, filter motions for physical feasibility using KungfuBot's CoM-CoP stability metric:
- For each frame, compute projected distance between CoM and CoP on ground
- Frame is stable if `Delta_d_t < epsilon_stab`
- Motion sequence is accepted if: (1) first and last frames are stable, (2) max gap of consecutive unstable frames < `epsilon_N`
- This prevents training on motions the robot physically cannot track

### 0.4 Motion Correction

Apply contact-aware correction (KungfuBot Section 3.1):
- Estimate foot contact masks from ankle displacement and height thresholds
- Correct vertical position when foot is in contact (eliminate floating artifacts)
- Apply EMA smoothing to reduce frame-to-frame jitter

### Output Format

Retargeted motions are `.pkl` files containing:
```python
{
    'joint_pos': (T, 23),       # joint angles (rad), 23 DoF (wrists locked)
    'joint_vel': (T, 23),       # joint velocities
    'root_pos': (T, 3),         # global root position
    'root_rot': (T, 4),         # root orientation (quaternion)
    'root_vel': (T, 3),         # root linear velocity
    'root_ang_vel': (T, 3),     # root angular velocity
    'contact_mask': (T, 2),     # left/right foot contact booleans
    'fps': 50,                  # frame rate
}
```

---

## Stage 1: DeepMimic Tracking (Skill Library)

This stage trains a universal motion tracker `pi_trk` that can faithfully reproduce any retargeted boxing motion. This is the physical foundation — without it, nothing else works.

### 1.1 Method

Follow the DeepMimic paradigm (Peng et al. 2018), as implemented by both RoboStriker and KungfuBot. The tracker is trained via PPO on a reference motion tracking task.

**Policy:** `pi_trk(a_t | s_self, s_ref)` — takes proprioceptive state + reference motion, outputs 23-DoF PD target positions.

**Observation `o_trk`** (from RoboStriker Appendix A.2):
1. Proprioceptive state `s_prop`: base angular velocities (3D), relative joint positions (23D), joint velocities (23D) = 49D
2. Reference motion state `s_ref`: target full-body poses and base velocities for current + next K=3 timesteps (reference poses are 29D pose + 3D root velocity per timestep, so ~4 * (29+3) = ~128D)
3. Previous action `a_{t-1}`: 23D

**Action:** 23-DoF PD target joint positions for the Unitree G1 (wrists locked)

### 1.2 Reward Function (from RoboStriker Table 4)

The tracking reward is a weighted sum of exponential tracking terms:

```
r_trk = w_p * r_p + w_o * r_o + w_bp * r_bp + w_bo * r_bo + w_lv * r_lv + w_av * r_av - r_reg
```

where each `r_x = exp(-||err_x||^2 / sigma_x^2)`.

| Reward Term | Weight (w) | Scaling Factor (sigma) |
|---|---|---|
| Root Position `r_p` | 0.5 | 0.3 |
| Root Orientation `r_o` | 0.5 | 0.4 |
| Body Position `r_bp` | 1.0 | 0.3 |
| Body Orientation `r_bo` | 1.0 | 0.4 |
| Body Linear Velocity `r_lv` | 1.0 | 1.0 |
| Body Angular Velocity `r_av` | 1.0 | 3.14 |

Regularization `r_reg` penalizes: action rate, joint limit violations, self-collisions, undesired ground contacts.

**KungfuBot's adaptive tracking factor (important improvement):**
KungfuBot's key contribution is making the `sigma` values adaptive. Instead of fixed sigmas, it:
1. Initializes sigma to a large value (e.g., joint_pos sigma = 0.3)
2. Maintains an EMA of tracking error: `x_hat`
3. Updates: `sigma <- min(sigma, x_hat)` (non-increasing)
4. This creates a curriculum: easy tracking first, progressively tighten as policy improves

Implement this. The initial sigma values from KungfuBot's config (`repos/PBHC/humanoidverse/config/rewards/motion_tracking/main.yaml`):
```yaml
reward_tracking_sigma:
  teleop_joint_pos: 0.3         # Joint position
  teleop_joint_vel: 30.0        # Joint velocity
  teleop_upper_body_pos: 0.015  # Body position (upper)
  teleop_lower_body_pos: 0.015  # Body position (lower)
  teleop_vr_3point_pos: 0.015   # Head/hand positions
  teleop_feet_pos: 0.01         # Feet positions
  teleop_body_rot: 0.1          # Body rotation
  teleop_body_vel: 1.0          # Body velocity
  teleop_body_ang_vel: 15.0     # Body angular velocity
  teleop_max_joint_pos: 1.0     # Max joint position deviation

adaptive_tracking_sigma:
  enable: True
  alpha: 1e-3
```

### 1.3 Additional Reward Terms (from KungfuBot config)

Penalty terms (same file):
```yaml
penalty_action_rate: -0.5      # Action smoothness (critical for real robot)
penalty_torques: -0.000001     # Energy
limits_dof_pos: -10.0          # Joint limit violations
limits_dof_vel: -5.0
limits_torque: -5.0
termination: -200.0             # Fall penalty
collision: -30.0               # Self-collision
feet_air_time: 1.0             # Gait stability
penalty_feet_contact_forces: -0.01
penalty_stumble: -2.0
penalty_slippage: -1.0
```

### 1.4 Early Termination (from RoboStriker A.2)

Episode terminates if:
1. **Pose collapse:** vertical distance between robot base and reference height < 0.25m, or base orientation deviation > 0.8 rad
2. **End-effector violation:** ankle/wrist vertical position deviates from reference by > 0.25m
3. **Time limit:** 10 seconds

### 1.5 Training Setup

- **Reference State Initialization (RSI):** Initialize robot state from reference motion at random time phases — enables parallel learning of different motion phases
- **Time phase variable** `phi_t in [0,1]`: linear progress indicator for the reference motion
- **Architecture:** Asymmetric actor-critic (KungfuBot Section 3.3):
  - Actor: proprioception + time phase only (deployable)
  - Critic: augmented with reference motion positions, root linear velocity, randomized physical parameters (privileged)
- **Reward vectorization:** Each reward component has its own value head, improving multi-objective learning

### 1.6 Training Config

| Parameter | Value | Source |
|---|---|---|
| Simulator | Isaac Lab (primary) or MuJoCo MJX (backup) | RoboStriker uses Isaac Lab; MJX validated for tracker |
| Num environments | 4096 (GPU) / 128 (debug) | Both papers |
| Physics freq | 200 Hz | RoboStriker |
| Control freq | 50 Hz (20ms) | RoboStriker |
| Algorithm | PPO (via rsl_rl) | KungfuBot / RoboStriker |
| Training iterations | 50,000 | KungfuBot |
| GPU | 1x RTX 4090 | Both papers |
| Domain randomization | Friction, mass, actuator gains | Both papers |

**Current status (2026-07-28):** Tracker training on Isaac Lab 4090 is
working — iter 230, reward 76, 57K FPS. MJX tracker (Stage 1) was
completed earlier (0.92 reward) but MJX warmup (Stage 3a) failed (zero
hits after 150+ iterations with hand-rolled JAX PPO). The warmup will be
ported to Isaac Lab.

**For MuJoCo-only setup (FightLab constraint):**
- Use 128-256 parallel envs (MuJoCo is CPU-based, no GPU parallelism)
- Expect 3-5x longer training time vs. IsaacGym
- Alternative: use Genesis simulator (KungfuBot has a Genesis backend: `repos/PBHC/humanoidverse/simulator/genesis/`)

### 1.7 What to Reuse

**From PBHC repo (`/opt/data/repos/PBHC/`):**
- `humanoidverse/envs/motion_tracking/motion_tracking.py` — the tracking environment
- `humanoidverse/envs/motion_tracking/general_tracking.py` — general motion tracking (multi-motion)
- `humanoidverse/utils/motion_lib/motion_lib_robot.py` — motion loading and sampling
- `humanoidverse/config/rewards/motion_tracking/main.yaml` — reward configuration
- `humanoidverse/simulator/` — simulator abstractions (IsaacGym, IsaacSim, Genesis, MuJoCo)
- `humanoidverse/agents/` — PPO implementation (rsl_rl based)

**Key file to study:** `humanoidverse/envs/motion_tracking/motion_tracking.py` — this is the core tracking environment that handles reference motion loading, reward computation, early termination, and domain randomization.

---

## Stage 2: Latent Space Distillation

This stage compresses the 23-DoF tracking policy into a 32-dimensional latent space. The latent space is the strategic action space for combat — agents will select latent codes, not joint targets.

### 2.1 Architecture (from RoboStriker Section 3.3, Appendix B)

Three networks trained jointly via teacher-student distillation:

**Encoder `E_phi`:** Maps full observations to latent distribution
- Input: `o_t = (s_self, s_ref)` — proprioceptive state + reference motion
- Output: Gaussian `N(z | mu_e, sigma_e)` in R^32
- Architecture: MLP, 3-4 hidden layers, 1024 units

**Decoder `D_psi`:** Reconstructs teacher's action from latent + proprioception
- Input: `(s_prop, z)` — proprioceptive state + latent code
- Output: 23-DoF action (PD targets)
- Architecture: MLP, 3-4 hidden layers, 1024 units
- This is FROZEN after distillation and used as the low-level controller in Stage 3

**State-conditioned prior `P_xi`:** Generates valid latent codes from proprioception only
- Input: `s_prop` — proprioceptive state only (no reference motion)
- Output: Gaussian `N(z | mu_p, sigma_p)` in R^32
- Architecture: MLP, 3-4 hidden layers, 1024 units
- This provides the behavioral prior for the residual policy in Stage 3

### 2.2 Training Objective

```
L_distill = L_rec + lambda_prior * L_prior
```

where:
- `L_rec = E[||a_t - a_hat_t||^2]` — reconstruction loss (teacher action vs. decoder output)
- `L_prior = E[D_KL(E_phi(z|o_t) || P_xi(z|s_prop))]` — KL regularization between encoder and prior
- `lambda_prior = 0.001`

The teacher policy `pi_trk` is frozen during distillation.

### 2.3 Hypersphere Projection (Critical Design Choice)

The latent space is constrained to the unit hypersphere `S^31` (32-D, unit norm):

```
z_hat = Normalize(z) = E_phi(o) / ||E_phi(o)||_2
```

This is the key innovation that distinguishes RoboStriker from PULSE/CALM. Why it matters:
1. **Boundedness:** The latent space is compact, ensuring game-theoretic convergence (Glicksberg's theorem requires compact strategy sets)
2. **Physical safety:** All latent codes map to physically plausible motions — no OOD exploration
3. **Stable self-play:** Bounded action space prevents the non-stationarity spiral that plagues unbounded continuous-action MARL
4. **t-SNE analysis shows structured clustering:** Move, Strike, and Move+Strike form distinct but connected regions on the manifold

**Alternative: BFM's CVAE approach.** BFM (arXiv:2509.13780, same lab as RoboStriker) uses a CVAE with masked online distillation for a more general foundation model. If a pretrained BFM model becomes available, it could serve as a drop-in replacement for this stage. BFM's code is at `bfm4humanoid.github.io` but may not be open yet — check availability.

### 2.4 Latent Dimensionality

**32 dimensions** (from RoboStriker: "Strategic co-evolution is conducted over a 32-dimensional latent manifold"). This was chosen to be large enough to capture boxing motion diversity (strikes, defense, footwork, transitions) but small enough to make the MARL problem tractable.

### 2.5 Training Config

| Parameter | Value |
|---|---|
| Latent dimension | 32 |
| Encoder/Decoder hidden | [1024, 1024, 1024] |
| Prior hidden | [1024, 1024, 1024] |
| KL weight (lambda_prior) | 0.001 |
| Training data | Rollouts from pi_trk on all motion clips |
| Epochs | ~100-200K iterations |
| GPU | 1x RTX 4090 |

---

## Stage 3: Combat RL

### 3a: Behavioral Warmup with AMP (vs. Sandbag)

The agent learns basic striking against a stationary opponent (the sandbag). This solves the "competitive cold-start problem" — you cannot do self-play from scratch.

#### 3a.1 Policy Architecture

A **residual policy** `pi_theta` outputs residual latent commands on top of the frozen prior:

```
pi_z(. | s_prop, s_goal) = pi_theta(. | s_goal) (+) P_xi_perp(. | s_prop)
```

```
z_t = Normalize(Delta_z_t + z_p_t)
```

where:
- `Delta_z_t ~ pi_theta(. | s_goal)` — residual from the combat policy
- `z_p_t ~ P_xi(. | s_prop)` — sample from the frozen prior
- Normalize projects back to the hypersphere

This ensures the agent's actions stay close to physically plausible motions (the prior) while the residual policy adds combat intent.

#### 3a.2 Goal-Oriented Observation (from RoboStriker C.1)

**No explicit motion commands.** The observation is purely spatial:

`s_goal = (v_l^off, v_r^off, v_l^def, v_r^def)`

where:
- **Offensive target:** relative position of ego's left/right fists to opponent's torso, in ego-centric frame (2 x 3D = 6D)
- **Defensive target:** relative position of opponent's left/right fists to ego's torso, in ego-centric frame (2 x 3D = 6D)

Total goal observation: 12D. Combined with proprioceptive state (~49D), the full observation is ~61D.

This is rotation-invariant (all vectors in ego-centric frame), which is critical for generalization.

#### 3a.3 Reward Function (from RoboStriker C.3)

```
r_warmup = w_task * r_task + w_style * r_style
```

where `w_task = 0.8`, `w_style = 0.2`.

**Task reward:**
```
r_task = w_face * r_face + w_vel * r_vel + w_dist * r_dist + w_hit * r_hit
```

1. **Facing alignment** `r_face`: `exp(-(1-d_x)/sigma_face)` where `d_x` is the forward component of the direction to opponent in ego frame. Rewards facing the opponent.

2. **Velocity reward** `r_vel`: rewards locomotion toward opponent. `I[v_parallel > 0] * exp(-e_v^2 / sigma_vel)` where `v_parallel` is velocity projected onto line-of-sight to opponent.

3. **Distance reward** `r_dist`: velocity-gated closeness of wrists to opponent torso. `mean_i[exp(-d_i / sigma_dist) * g_i]` where `g_i = sigmoid(alpha * (s_i - v_th))` gates on punching speed toward opponent. This prevents passive exploitation (just standing close without striking).

4. **Hit reward** `r_hit`: sparse, binary. `I[H_l OR H_r]` where `H_i = C_ego_i AND C_opp_t AND s_i > tau_v`. Requires: (a) contact force on ego wrist > threshold, (b) contact force on opponent torso > threshold, (c) relative punching velocity > threshold. This eliminates degenerate behaviors like leaning or body pushing.

**Style reward (AMP discriminator):**
```
r_style(s_t, s_{t+1}) = max[0, 1 - 0.25 * (C(o_disc, o_{t+1}^disc) - 1)^2]
```

#### 3a.4 AMP Discriminator (from RoboStriker C.2)

**Purpose:** Prevent degeneration of motion quality during combat training. The discriminator ensures strikes look like real boxing, not flailing.

**Discriminator observation `o_disc`:** local rotation and velocity of each joint + angular velocities of the base. This captures motion style independent of global position.

**Training:** Standard AMP (Peng et al. 2021):
- Real data: transitions from mocap dataset `D_motion`
- Fake data: transitions from agent (policy + frozen decoder)
- Loss: `(C(real) - 1)^2 + (C(fake) + 1)^2 + w_gp * ||grad_C(real)||^2`
- Gradient penalty on real data prevents mode collapse

**What data:** The same boxing mocap used for Stage 1. The discriminator learns the boxing motion style distribution.

**Implementation:** The AMP discriminator code from nv-tlabs/ASE (github.com/nv-tlabs/ASE) or from the original AMP paper implementation. The discriminator is a simple MLP (3-4 layers, 512 units).

#### 3a.5 Reward Scheduling

Dynamic weight schedule during warmup:
- **Initial epochs:** primarily `r_style` (master stable bipedal stance and boxing postures)
- **Later epochs:** phase in `w_task` (optimize strike precision while maintaining style)
- This curriculum prevents the agent from rushing to exploit the hit reward before it can stand properly

#### 3a.6 Sandbag Setup

- Opponent: a G1 robot with a frozen standing-still policy (or the
  deprecated v1 walker.onnx, which is kept as a legacy baseline only)
- No contact response from the sandbag (it just stands there)
- Purpose: learn to approach, face, and land punches on a target

### 3b: Latent-Space Neural Fictitious Self-Play (LS-NFSP)

After warmup, transition to competitive self-play. Two agents fight each other in the latent action space.

#### 3b.1 Dual-Policy Architecture (from RoboStrizer D.1)

Each agent maintains:
1. **Best-response policy `pi_z^RL`:** on-policy PPO actor, exploits opponent's current average strategy
2. **Average policy `pi_bar_z`:** supervised learning network, approximates agent's historical best-response distribution. Trained by minimizing `||pi_bar_z(o) - z||^2` on reservoir buffer.

#### 3b.2 Action Selection

Mixed strategy with anticipatory parameter `eta = 0.1`:
```
sigma = eta * pi_bar_z + (1 - eta) * pi_z^RL
```

At each step, sample `m ~ Bernoulli(eta)`:
- `m=1` (best response): sample from `pi_z^RL`, insert into reservoir buffer
- `m=0` (average): sample from `pi_bar_z`

This mixing stabilizes training: 90% of the time the agent explores (best response), 10% it plays its stable average strategy.

#### 3b.3 Reservoir Buffer (from RoboStriker D.2)

- Capacity `K = 10^6` samples per agent
- Reservoir sampling: for every new experience `n > K`, replace random entry with probability `K/n`
- Prevents bias toward recent iterations, preserves tactical diversity

#### 3b.4 Expanded Reward for Competition (from RoboStriker D.3)

```
r_expand = w_str * r_str - w_def * r_def + w_term * r_term
```

1. **Strike force reward** `r_str = ||F_opp|| - ||F_ego||` — net contact force differential (hit opponent, don't get hit)

2. **Defensive penalty** `r_def = I[delta_l OR delta_r]` where `delta_i = I[f_i > tau_f AND s_i > tau_v]` — penalized when hit by high-velocity strike

3. **Terminal outcome** `r_term = I[h_opp < h_min] - I[h_ego < h_min]` — +1 if opponent falls, -1 if ego falls. Match is lost when any body part other than feet contacts the ground.

The full combat reward includes the warmup terms plus these competitive terms.

#### 3b.5 Training Config

| Parameter | Value | Source |
|---|---|---|
| Exploration parameter eta | 0.1 | RoboStriker |
| Reservoir capacity K | 10^6 per agent | RoboStriker |
| Task reward weight w_task | 0.8 | RoboStriker |
| Style reward weight w_style | 0.2 | RoboStriker |
| Prior loss coefficient lambda_prior | 0.001 | RoboStriker |
| Hit force threshold F_th | 10 N | RoboStriker |
| Num parallel envs | 4096 (Isaac Lab) | RoboStriker |
| GPU | 1x RTX 4090 | RoboStriker |
| Domain randomization | Friction, mass, actuator gains | Both |

---

## Sim-to-Real Transfer

RoboStriker demonstrates zero-shot sim-to-real transfer using:
1. **Domain randomization:** friction, link masses, actuator gains
2. **Sim-to-sim verification:** validate in MuJoCo before real deployment
3. **KungfuBot approach:** same — train in Isaac Lab, verify in MuJoCo, deploy on real G1 with zero-shot transfer

KungfuBot's sim-to-sim deployment code is at `repos/PBHC/humanoidverse/deploy/mujoco.py` and exports policies to ONNX format for deployment.

For FightLab (simulation-only league), sim-to-real is not required, but domain randomization should still be applied for robustness and to prevent overfitting to specific simulation parameters.

---

## RoboStriker Ablations — What Failed

These are critical findings from the paper that inform our design:

1. **29-DoF action-space self-play fails catastrophically.** Win rate: 0% vs. LS-NFSP. Offensive landing rate: 0.142 vs. 0.685. Base stability: 0.418 vs. 0.942. **This is exactly v1's failure mode.**

2. **Self-play without warmup fails.** `SP w/o Warmup` achieves `eta_hit = 0.050` (vs. 0.685). Cannot overcome reward sparsity from scratch. The sandbag warmup is essential.

3. **PPO-only (no self-play) fails.** Win rate: 15.5% vs. LS-NFSP. Competitive interaction is the indispensable driver of emergent combat behaviors.

4. **No AMP (LS-NFSP w/o AMP) degrades significantly.** `eta_hit` drops from 0.685 to 0.490. The agent approaches but executes erratic, ineffective strikes. Win rate: 17.6% vs. LS-NFSP. AMP is essential for both physical authenticity and strike effectiveness.

5. **Naive self-play (no opponent averaging) underperforms.** Win rate: 23.8% vs. LS-NFSP. Policy cycling without the NFSP average policy mechanism.

6. **Fictitious SP (uniform sampling from history, no learned average) underperforms.** Win rate: 31.5% vs. LS-NFSP. The explicit learned average policy network is necessary.

---

## Implementation Plan

### Phase 1: Motion Data Pipeline (1-2 weeks)

1. **Acquire boxing mocap:**
   - Download CMU category 13 boxing clips
   - Convert to SMPL via AMASS pipeline (`repos/PBHC/motion_source/`)
   - Optionally: record custom mocap or use Mixamo boxing clips
   - Target: 30-50 boxing motion clips

2. **Retarget to G1:**
   - Use Mink retargeting (`repos/PBHC/smpl_retarget/mink_retarget/`)
   - Apply motion correction (--correct flag)
   - Compute contact masks (`repos/PBHC/motion_source/count_pkl_contact_mask.py`)
   - Verify retargeted motions visually with `repos/PBHC/robot_motion_process/vis_q_mj.py`

3. **Filter motions:**
   - Apply physics-based stability filter (CoM-CoP distance)
   - Verify Episode Length Ratio (ELR) > 0.8 for each motion

### Phase 2: DeepMimic Tracker (2-3 weeks)

1. **Set up environment:**
   - Option A: Isaac Lab (primary, confirmed working on RunPod 4090). Requires NVIDIA GPU + libvulkan1.
   - Option B: MuJoCo MJX (backup, GPU-parallel via JAX). Validated for tracker stage.
   - Option C: Genesis (KungfuBot has backend at `repos/PBHC/humanoidverse/simulator/genesis/`)
   - Option D: MuJoCo CPU (slow, 128 envs, 3-5x slower — debug only)

2. **Implement tracking environment:**
   - Port `repos/PBHC/humanoidverse/envs/motion_tracking/motion_tracking.py` to FightLab
   - Implement reward function with adaptive tracking factors
   - Implement early termination, RSI, domain randomization
   - Use asymmetric actor-critic (privileged critic)

3. **Train:**
   - 50,000 iterations, 4096 envs (or 128 for MuJoCo)
   - Monitor: episode length, tracking error, sigma convergence
   - Success criterion: ELR > 0.9 on all boxing motions

### Phase 3: Latent Distillation (1-2 weeks)

1. **Collect teacher rollouts:**
   - Run `pi_trk` on all motion clips
   - Collect (s_self, s_ref, a_teacher) tuples
   - ~1M transitions

2. **Train CVAE:**
   - Encoder: (s_self, s_ref) -> N(mu, sigma) in R^32
   - Decoder: (s_prop, z) -> a
   - Prior: (s_prop) -> N(mu_p, sigma_p) in R^32
   - Hypersphere normalization
   - Lambda_prior = 0.001
   - ~100-200K iterations

3. **Verify latent space:**
   - t-SNE visualization should show structured clusters
   - Decoder should reconstruct teacher actions with low error
   - Prior should generate physically plausible motions

### Phase 4: Warmup + Self-Play (3-4 weeks)

1. **Implement combat environment:**
   - Two G1 robots in Isaac Lab (or MuJoCo for sim-to-sim validation)
   - Goal-oriented observation (offensive + defensive vectors)
   - Sandbag opponent (frozen standing policy)
   - Contact force sensing

2. **Train AMP discriminator:**
   - Collect mocap transitions for discriminator observation
   - Train discriminator on (real=mocap, fake=agent) classification
   - Apply gradient penalty

3. **Warmup training:**
   - Residual policy in latent space
   - Reward: facing + velocity + distance + hit + style
   - Dynamic weight scheduling (style first, then task)
   - Success criterion: consistent hits on sandbag, `eta_hit > 0.3`

4. **LS-NFSP self-play:**
   - Two agents, dual-policy (best response + average)
   - Reservoir buffer, K=10^6
   - eta=0.1
   - Expanded reward: strike force - defense + terminal
   - Domain randomization
   - Success criterion: emergent boxing tactics (slips, counters, footwork)

### Phase 5: League Integration (1 week)

1. **Export policy** to ONNX format
2. **Integrate with FightLab eval harness** (`engine/eval_harness.py`)
3. **Verify against v1 sandbag baseline**
4. **Set up as a new king** in the league

### Estimated Total Time and Compute

| Component | Time | Compute |
|---|---|---|
| Motion data pipeline | 1-2 weeks | CPU only |
| DeepMimic tracker | 2-3 weeks | 1x RTX 4090, ~2-3 days training |
| Latent distillation | 1-2 weeks | 1x RTX 4090, ~1 day training |
| Warmup + self-play | 3-4 weeks | 1x RTX 4090, ~3-5 days training |
| League integration | 1 week | CPU |
| **Total** | **8-12 weeks** | 1x RTX 4090 |

With MuJoCo CPU only (no Isaac Lab or MJX), multiply training times by 3-5x.

---

## Code to Write vs. Reuse

### Reuse from PBHC (`/opt/data/repos/PBHC/`)

| Component | File(s) | Notes |
|---|---|---|
| Motion processing pipeline | `motion_source/`, `smpl_retarget/` | Full pipeline, ready to use |
| Motion visualization | `robot_motion_process/`, `smpl_vis/` | Debugging and verification |
| Tracking environment | `humanoidverse/envs/motion_tracking/` | Core RL env, needs adaptation |
| Motion library | `humanoidverse/utils/motion_lib/` | Motion loading and sampling |
| PPO implementation | `humanoidverse/agents/` | rsl_rl based, proven |
| Simulator abstraction | `humanoidverse/simulator/` | Isaac Lab, IsaacGym (deprecated), Genesis, MuJoCo backends |
| Domain randomization | `humanoidverse/config/` | Config-based, easy to modify |
| MuJoCo deployment | `humanoidverse/deploy/mujoco.py` | Sim-to-sim verification |
| Example motions | `example/motion_data/` | Includes Hooks_punch, Side_kick, etc. |
| Pretrained checkpoint | `example/pretrained_horse_stance_punch/` | Can bootstrap from this |

### Write New

| Component | What it does | Based on |
|---|---|---|
| `combat_env.py` | Two-agent combat env with contact sensing | RoboStriker reward formulation |
| `latent_distillation.py` | CVAE encoder/decoder/prior training | RoboStriker Section 3.3 |
| `amp_discriminator.py` | Style discriminator for boxing | AMP (Peng et al. 2021), nv-tlabs/ASE |
| `ls_nfsp.py` | Latent-space NFSP training loop | RoboStriker Section 3.5, Algorithm 1 |
| `residual_policy.py` | Residual latent policy for warmup | RoboStriker Section 3.4 |
| `boxing_reward.py` | Combat reward functions | RoboStriker Appendix C |
| `motion_filter_boxing.py` | Boxing-specific motion filtering | KungfuBot Section 3.1 |

### Reuse from Other Repos

| Component | Repo | Notes |
|---|---|---|
| AMP/ASE discriminator | github.com/nv-tlabs/ASE | Style reward implementation |
| ExBody2 upper body | github.com/chengxuxin/expressive-humanoid | Upper body motion imitation patterns |
| BFM (if available) | bfm4humanoid.github.io | Potential base controller replacement |
| Mink retargeting | github.com/kevinzakka/mink | IK-based retargeting (in PBHC) |
| PHC retargeting | github.com/ZhengyiLuo/PHC | Alternative retargeting (in PBHC) |

---

## Key Differences from v1

| Aspect | v1 | v2 |
|---|---|---|
| Action space | 17-D raw (3 vel + 14 arm) | 32-D latent (hypersphere) |
| Observation | 41-D (proprioception only) | ~61-D (proprioception + combat goal) |
| Opponent | Static sandbag | Sandbag warmup → self-play |
| Reward | Sparse (HP, contact) | Dense (facing, velocity, distance, hit, style) |
| Motion prior | None (walker.onnx for balance) | DeepMimic tracking + CVAE latent |
| Style | None | AMP discriminator (boxing mocap) |
| Training | Single PPO | 3-stage: track → distill → NFSP |
| Result | Stands, doesn't fight | Emergent boxing tactics |

---

## Risk Assessment

### High Risk

1. **Isaac Lab availability:** Isaac Lab is confirmed working on RunPod
   4090 (requires libvulkan1, see v2-validation.md). If unavailable, use
   MuJoCo MJX (GPU-parallel, validated for tracker) or Genesis (KungfuBot
   has a backend). MuJoCo CPU is 3-5x slower and not recommended for
   production training.

2. **Boxing mocap data:** CMU category 13 may not have enough variety. May need to record custom mocap or augment with Mixamo. Budget for this.

3. **MuJoCo two-agent simulation:** Running two G1 robots in MuJoCo with contact is computationally expensive. May need to optimize or use a different simulator for the combat stage.

### Medium Risk

4. **Latent space quality:** If the CVAE doesn't capture enough motion diversity, the self-play won't produce diverse tactics. Mitigate by ensuring the motion library is comprehensive (strikes, defense, footwork, transitions).

5. **AMP discriminator stability:** AMP training can be unstable. Use gradient penalty and monitor for mode collapse. Follow nv-tlabs/ASE implementation.

6. **NFSP convergence:** Self-play can cycle without converging. The reservoir buffer and average policy are designed to mitigate this, but monitor win-rate stability.

### Low Risk

7. **Retargeting quality:** PBHC's Mink pipeline is proven and includes example boxing motions (Hooks_punch.pkl).

8. **Tracker training:** DeepMimic is a well-established method. The adaptive tracking factor from KungfuBot further improves convergence.

---

## References

1. **RoboStriker** — Yin et al., "RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing", ICML 2026, arXiv:2601.22517. Project: yinkangning0124.github.io/RoboStriker/

2. **KungfuBot/PBHC** — Xie et al., "KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills", NeurIPS 2025, arXiv:2506.12851. Code: github.com/TeleHuman/PBHC. Project: kungfubot.github.io

3. **KungfuBot2** — Han et al., "KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control", arXiv:2509.16638. General motion tracking extension.

4. **BFM** — Zeng et al., "Behavior Foundation Model for Humanoid Robots", arXiv:2509.13780. Project: bfm4humanoid.github.io. Same lab as RoboStriker.

5. **ExBody2** — Ji et al., "ExBody2: Advanced Expressive Humanoid Whole-Body Control", arXiv:2412.13196. Code: github.com/chengxuxin/expressive-humanoid. Project: exbody2.github.io

6. **DeepMimic** — Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills", ACM TOG 2018.

7. **AMP** — Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control", ACM TOG 2021.

8. **ASE** — Peng et al., "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters", ACM TOG 2022. Code: github.com/nv-tlabs/ASE

9. **NFSP** — Heinrich and Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games", arXiv:1603.01121.

10. **UniTracker** — Yin et al., "UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots", arXiv:2507.07356. Same first author as RoboStriker.

11. **GMR** — Araujo et al., "Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking", arXiv:2510.02252.

12. **PULSE** — Luo et al., "Universal Humanoid Motion Representations for Physics-Based Control", arXiv:2310.04582.

13. **AMASS** — Mahmood et al., "AMASS: Archive of Motion Capture as Surface Shapes", ICCV 2019. Data: amass.is.tue.mpg.de

14. **Mink** — Zakka, "Mink: Python Inverse Kinematics based on MuJoCo", github.com/kevinzakka/mink

15. **PHC** — Luo et al., github.com/ZhengyiLuo/PHC

16. **ASAP** — He et al., "ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills", arXiv:2502.01143. Code: github.com/LeCAR-Lab/ASAP (base for PBHC)
