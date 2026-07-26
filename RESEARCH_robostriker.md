# Track B research: RoboStriker author ecosystem (2026-07-26)

## RoboStriker (arXiv 2601.22517, ICML 2026) — Kangning Yin et al.
Autonomous humanoid boxing, 29-DoF Unitree G1. 3-stage hierarchy:
  1. Motion tracker (mocap imitation) -> skill repertoire
  2. Distill to bounded latent manifold (hypersphere)
  3. Latent-Space Neural Fictitious Self-Play (LS-NFSP)
**Code NOT released** (paper + website only). No weights to bootstrap.

## Same-author / cited open resources (for our Stage-1 motion library)
- **UniTracker** (arXiv 2507.07356, Yin et al. 2025): motion tracker
  trained on 8k G1 motions. Same lead author as RoboStriker. Stage-1
  foundation. Code status unclear.
- **BeyondMimic** (arXiv 2508.08241, HybridRobotics): OPEN CODE
  (github.com/HybridRobotics/whole_body_tracking, MIT). G1 motion
  tracking, IsaacSim/Lab. State-of-the-art tracker RoboStriker builds on.
- **GMR / Retargeting Matters** (arXiv 2510.02252, ICRA 2026):
  OPEN CODE (github.com/YanjieZe/GMR). Retargets human mocap -> G1.
  RoboStriker cites this for its mocap pipeline.
- **openhe/g1-retargeted-motions** (HuggingFace): 174 retargeted G1
  motions (23-DOF, 30fps, .pkl) incl kungfu (Hooks_punch,
  Horse-stance_punch, Roundhouse_kick, Side_kick) + LAFAN1 fight.
  **CORRUPT on HF** (pickle fails at \x0b across wget + snapshot_download
  + multiple clips). LFS/truncation issue. RETRY LATER via a working
  mirror or contact author. If we get clean copies, convert 23->29 DOF
  (like build_g1moves_npz.py) and use as motion-match references /
  Stage-1 imitation targets — directly upgrades strike quality.

## Conclusion
Our lean reimplementation mirrors RoboStriker's architecture (G1 Moves =
Stage 1, motion-match coach = Stage 2 latent, league = Stage 3 self-play).
No shortcut weights available, but the *method* is validated and the
open tools (BeyondMimic code, GMR) confirm the approach. The
retargeted-motion dataset would help but is currently unobtainable
intact — defer.
