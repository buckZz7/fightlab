# ROADMAP.md — FightLab Roadmap

**Version:** 1.1
**Last updated:** 2026-07-28T00:00:00Z

This roadmap is honest about what works today and what does not. It is
not a hype document. Each phase has a clear done condition. Items move
left only when the done condition is met.

---

## Current Status (2026-07-28)

| Component | Status | Details |
|---|---|---|
| League system (ELO, round-robin, king) | Working | league.py, league_state.json |
| Trustless eval harness | Working | engine/eval_harness.py, MockBoutRunner |
| Miner SDK | Working | miner_sdk.py |
| Website | Live | Real standings, tracker_eval.mp4 in hero |
| Gittensor SN74 scaffold | Complete | REVIEW.md, EVAL-TRUST.md, CI, config.json |
| Stage 0: Motion library | Done | 68 clips, 62 combat, 23-DoF, 30fps |
| Stage 1: DeepMimic tracker | In progress | Isaac Lab 4090, iter 230, reward 76, 57K FPS |
| Stage 2: Latent distillation | Done | decoder + prior ONNX exported |
| Stage 3a: Warmup | Blocked | MJX attempt failed (zero hits); needs Isaac Lab port |
| Stage 3b: Self-play | Not started | Depends on warmup |
| Render pipeline | Partial | fight_real.mp4 (standing G1s), tracker_eval.mp4 |
| AMP discriminator | Done | 99% accuracy, separates boxing from flailing |
| Primary training backend | Isaac Lab | Confirmed working on RunPod 4090 |
| Backup training backend | MuJoCo MJX | Validated for tracker stage only |

---

## Phase 1 — First king (current)

**Status:** In progress.
**Done condition:** A trained policy wins a multi-seed bout against the
scripted baseline, is crowned king, and the crown is reproducible from
the signed result.

- Train a fighting policy for the Unitree G1 using the v2 pipeline
  (track → distill → warmup → self-play). The v1 17-D action interface
  is deprecated; v2 uses a 32-D latent space over 23-DoF PD targets.
- Run the multi-seed eval harness (`engine/eval_harness.py`) against the
  scripted baseline.
- Crown the first king via `league.py crown`; archive open weights and
  metadata to `kings/`.
- Render the crowning bout for review (on the GPU pod, not bundled here).
- Publish the signed bout result so any third party can replay it.

This phase proves the pipeline end to end: train, submit, eval, crown,
render. It does not prove trustlessness at scale.

### Training pipeline progress

The v2 pipeline follows the RoboStriker architecture (3-stage: track →
distill → self-play). Current state:

1. **Tracker (Stage 1):** Training on Isaac Lab 4090 with rsl_rl PPO.
   4096 parallel envs, 57K FPS, reward climbing (iter 230, reward 76).
   Target: reward >95, episode length ratio >0.9. MJX tracker was
   completed first (0.92 reward) but Isaac Lab provides better RL
   infrastructure for the warmup and self-play stages.

2. **Distillation (Stage 2):** Complete. Decoder and prior exported to
   ONNX. Latent space is 32-D hypersphere. Reconstruction MSE = 0.007.

3. **Warmup (Stage 3a):** Blocked. MJX warmup with hand-rolled JAX PPO
   failed (zero hits after 150+ iterations, 4+ hours). Will be ported to
   Isaac Lab with rsl_rl PPO and 4096 envs.

4. **Self-play (Stage 3b):** Not started. Depends on warmup producing
   consistent hits (target: eta_hit > 0.3 on sandbag).

---

## Phase 2 — League live

**Status:** Not started.
**Done condition:** Multiple miners submit policies, bouts run on a
schedule, ELO rankings update after every bout, and the king changes
hands at least once.

- Multi-fighter round-robin scheduling (`league.py schedule`).
- Cumulative ELO (K=32, start 1000). Per-tournament reset causes churn;
  cumulative ELO stabilizes and surfaces durable skill (lesson from
  Compelle, SN82).
- Weekly bout schedule: every fighter challenges the king and one peer
  per cycle.
- Public leaderboard (the `web/` landing page, fed by `league_state.json`).
- King archive grows: each reign is a directory under `kings/` with
  weights, metadata, and the signed crowning result.

This phase proves the league is live and the king is not permanent. It
does not prove the eval is tamper-proof.

---

## Phase 3 — Trustless hardening

**Status:** Not started.
**Done condition:** A bout can be run in a Docker sandbox with network
lockdown, multiple validators independently agree on the outcome, and a
TEE attestation proves the correct env and policy were loaded.

- **Docker isolation.** Run each bout in a sandboxed container with
  network lockdown applied before policy import. Prevents phone-home
  and file-system tampering. Pattern from Swarm (SN124) and Oro (SN15).
- **Multi-validator consensus.** Multiple validators independently run
  the same bout with the same seed and env pin. Results must agree
  (same outcome, same hash). Disagreement flags the bout for manual
  review. Pattern from Oro (SN15).
- **TEE attested eval.** Run the bout inside a Trusted Execution
  Environment (e.g., Intel TDX, AMD SEV-SNP). The TEE produces an
  attestation that the correct env and policy were loaded. Pattern from
  Poker44 (SN126).
- **Dual-target eval.** Score PRs against a visible bot roster and a
  held-out private roster; take the worse result. Prevents overfitting
  to known opponents. Pattern from vanguarstew (SN74).
- **Rolling seed leases.** Seeds are leased per-evaluation, not reused,
  to prevent seed memorization. Pattern from Swarm (SN124).

This phase closes the honest-boundary gaps documented in
[EVAL-TRUST.md](EVAL-TRUST.md). Until it lands, trust is by replay and
signature, not by hardware guarantee.

---

## Phase 4 — Sim-to-real

**Status:** Not started. This is the hardest phase and the furthest out.
**Done condition:** A sim-trained king policy is deployed on a real
Unitree G1 and completes a structured contact bout without falling.

- Deploy a sim-trained king policy to a real G1.
- Structured contact protocol: controlled strikes against a padded
  target, then a compliant opponent, then a resisting opponent.
- Measure sim-to-real transfer gap: does the sim winner stay a real
  winner?
- If transfer fails, iterate on domain randomization and observation
  richness in the sim until transfer improves.

This phase is honest about the sim-to-real gap. A policy that wins in
sim may not win on a real G1. The goal of Phase 4 is to measure the
gap, not to claim it is closed.

---

## What this roadmap is not

- It is not a promise of a real fighting robot league. Real humanoid
  combat is dangerous and legally complex. Phase 4 is structured contact,
  not a free fight.
- It is not a timeline. Phases land when the done condition is met, not
  on a date.
- It is not a tokenomics document. Emission config lives in
  `.gittensor/config.json` and [REVIEW.md](REVIEW.md).
