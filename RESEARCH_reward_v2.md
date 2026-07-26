# Reward design v2 — balance + fighter (research-backed)

Synthesized from:
- **HoST** (Huang et al, RSS 2025, Unitree G1 standing, MIT): foot-contact
  reward, action-smoothness (L2 delta-action), height/upright Gaussians,
  multi-critic. PD gains they use: hip 150, knee 300, ankle 40, upper 100.
  OUR KP MATCHES THIS (hip150/knee300/ankle40/upper100). Validated.
- **RoboStriker** (Yin et al, ICML 2026, G1 boxing): w_hit=50 (speed>1.0),
  w_def=8, w_delta=0.3, w_face=1.2, w_dist=1.5 (velocity-gated approach),
  terminal win +25. Motion-prior (AMP-equivalent) critical: dropping it
  drops hit-rate 0.685 -> 0.49. Warmup vs heuristic opponent.
- **Our own findings** (2026-07-26): foot-ground contact was DISABLED
  (mesh-only feet) -> robot could not stand. Fixed. PD alone cannot hold a
  G1 (>~1.4s sag) -- the policy MUST learn active balance.

## Balance env reward (g1_balance_env.py) -- CURRENT
- height Gaussian exp(-(z-0.793)^2 / 0.02)
- upright 0.5*max(0, up)
- -0.02 * torso lin-vel (drift)
- -0.005 * ||act||^2 (action cost)
- -0.02 * max(0, 0.4-z) (near-fall)
- +0.1 * min(foot_contacts, 2)   (PLANT FEET)
- -0.01 * ||act - prev_act||^2     (SMOOTHNESS, HoST)

## Fighter env reward (g1_fighter_env.py) -- CURRENT (RoboStriker-aligned)
- strike: +50 * dmg/8 (gated rel_vel>1.0 forceful, >0.5 glancing)
- defensive: -8 * dmg_taken/8
- delta force: +0.3 * (dealt - taken)
- motion-match (coach): +2.0 * coach_bonus
- facing: +1.2 * exp(-max(0,1-facing)/0.5)
- approach: +1.5 * (toward & within range) * exp(-|dist-0.5|)
- balance: -0.05 * max(0, 0.4 - z)

## Proposed v2 upgrades (when we iterate)
1. Fighter: foot-contact reward for BOTH bots (encourage planting, not
   hopping) -- HoST shows it stabilizes.
2. Fighter: action-smoothness on the 17-dim action (prevents jitter punches).
3. Fighter: add a small "distance maintenance" reward so bots don't clinch
   or run away (RoboStriker keeps them in striking range).
4. Balance: if policy still sinks, add ankle/foot torque penalty (energy)
   and a stronger height weight; consider HoST's multi-critic (separate
   value nets per reward group) -- but SB3 single-critic is fine for v1.
5. Curriculum: warmup r2 as a static sandbag (current), then promote the
   previous best challenger as r2 (self-play league) -- already in league.py.

## Sim2real DR (already in g1_fighter_env, randomize=True)
mass +-10%, friction +-15%, PD-gain jitter, per-step torque noise ~5%,
1-step actuator delay. Eval/bout use deterministic (randomize=False).
