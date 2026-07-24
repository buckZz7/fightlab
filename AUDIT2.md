# FightLab Audit #2 — wrong-direction risks
Run after G1 env is verified. Goal: catch our next bad assumption early.

## Open assumptions to attack
1. **CPU-only PPO will get us to watchable boxing.** Check: sample-complexity of comparable work (RoboStriker compute, Bansal sumo ~100M+ steps?). At what point do we rent GPU + MJX? What's the cheapest path (rental vs Modal vs local 4090)?
2. **Reward-shaped self-play can produce strikes, not shoving.** Bansal sumo converged to shoving. RoboStriker used mocap skills first. Our plan adds AMP motion prior — validate that AMP on retargeted boxing mocap actually transfers to G1 morphology (any failures reported?), and find the cheapest mocap source + retarget pipeline (LAFAN1? AMASS? GMT?).
3. **SB3 PPO is enough.** Should we be on a league-capable MARL stack (e.g. Ray RLlib, mjx envs) from the start? PFSP/exploiters on SB3 = DIY; is there prior art?
4. **HP/force-damage rules produce good fights.** Any game-design analysis of damage models in robot/physics fighting (hitstun, knockback scaling)? Watch URKL footage: what makes those fights good/bad — can sim rules steal the good parts?
5. **Sim2real gap items nobody warns about.** Contact-rich punching = hardest sim2real regime (impacts, gear backlash, foot slip). What failed in OP3 soccer / humanoid-gym transfers? What DR ranges matter most? What does Unitree G1 hardware actually tolerate (impact warranty? falls)?
6. **The G1 itself is the right first robot.** vs Booster T1 (cheaper, fight-oriented), vs H1, vs used/cheaper platforms. Re-price as of audit date.
7. **League incentive design.** If others ever submit fighters: what stops overfit-to-king exploits, collusion, eval gaming? Gittensor lesson applied.
8. **Legal/safety surface for real bouts.** Venue/insurance precedent from URKL/BattleBots — what does running a real autonomous fight event actually require?

## Method
For each: search for disconfirming evidence, not confirmation. Output: keep/change/drop verdict per assumption + concrete action.
