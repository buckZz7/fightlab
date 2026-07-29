# EVAL-TRUST.md — How FightLab Evaluation Is Trustworthy

**Version:** 1.0
**Last updated:** 2026-07-27T00:00:00Z

FightLab evaluates fighting policies by running bouts in a MuJoCo
physics simulation. This document states plainly what is deterministic,
what is reproducible, where the honest boundary is, and what is on the
roadmap to tighten trust further. It is the companion to
[REVIEW.md](REVIEW.md).

The goal: any third party can take a signed bout result, replay the bout
with the same seed and env pin, and get the same outcome. No party can
fake a result without breaking the hash.

---

## What is deterministic

- **Bout outcome.** A bout is a MuJoCo physics simulation step loop. Given
  the same seed, the same env config, and the same policy weights, the
  simulation produces the same trajectory and the same outcome.
- **Damage tracking.** Damage is computed from wrist-to-torso contact
  gated by relative velocity (see `G1FighterEnv._update_damage`). HP
  starts at 100 and decrements deterministically per contact event.
- **HP system.** HP is a scalar per fighter, updated only by the damage
  function. There is no subjective "how hurt is the fighter" call.
- **KO detection.** A KO is HP reaching 0, or pelvis-z falling below
  0.4 m (the robot fell). Both are numeric thresholds, not judgments.
- **Decision scoring.** If no KO occurs by the step cap, the winner is
  the fighter with the higher cumulative reward. Reward is computed by
  `G1FighterEnv._compute_reward` per step. This is a pure function of
  the state, not an LLM judge.

The bout outcome is not a human or LLM opinion. It is a function of the
physics engine output, which is deterministic given the seed and config.

---

## What is reproducible

- **Sim seed.** Every bout pins a list of seeds (default 5). `np.random`,
  Python `random`, and `env.reset(seed=...)` are all seeded before each
  bout. The seed list is part of the signed result.
- **Env pin.** The env config (sim dt, max steps, damage threshold,
  observation/action dims, domain randomization flag) is hashed into the
  result as `env_config_hash`. Any override changes the hash and is
  auditable.
- **Policy hash.** Both policy files are SHA-256 hashed into the result.
  A swapped policy produces a different hash.
- **Replay.** Anyone with the result JSON, the two policy files, and the
  env pin can re-run the bout and get the same outcome. The
  `engine/eval_harness.py verify` subcommand checks the payload hash
  locally.

---

## Multi-seed

A bout runs across 5 seeds by default. The aggregate is majority wins
across seeds. Per-seed results (winner, score, KO flag, step count) are
all in the signed result.

The **damage gate** prevents passive wins: a fighter must deal at least
`damage_threshold` (default 50.0) over the bout to win by decision. A
fighter that avoids contact and waits out the clock does not win. This
prevents the "do nothing and draw" strategy from being rewarded.

---

## SHA-256 signed results

Every bout result is a JSON payload with a `payload_sha256` field equal
to the SHA-256 of the canonical (sorted-key, no-space) serialization of
the rest of the payload. The payload includes:

- seed list
- both policy file hashes
- env config hash
- per-seed results
- aggregate result

Any edit to any field invalidates the hash. An optional HMAC-SHA256
signature (computed when a signing key is supplied) adds authentication
on top of integrity. Verification is a local operation; no network call
is required.

---

## Honest boundary

- **Sim stochasticity.** MuJoCo is deterministic given the seed, but
  real hardware is not. A policy that wins in sim may not win on a real
  G1. This is the sim-to-real gap and it is real. FightLab does not
  claim sim winners are real winners; the roadmap tracks sim-to-real
  transfer as Phase 4.
- **No TEE yet.** Evaluation runs on a validator's machine, not in a
  Trusted Execution Environment. A hostile validator with kernel access
  could in principle tamper with the result before signing. The
  multi-seed and signed-result design makes tampering detectable by
  replay, but not impossible. TEE attested eval is on the roadmap.
- **Single validator.** Today a bout is run by one validator. A
  dishonest validator could report a false result. Multi-validator
  consensus (multiple validators run the same bout and must agree) is
  on the roadmap.

This section is the honest boundary. It is not a marketing claim. If
you find a gap between this document and the implementation, open an
issue.

---

## Roadmap (eval trust)

These items are tracked in [ROADMAP.md](ROADMAP.md) Phase 3 and are
repeated here for eval-trust context.

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

Until these land, trust is established by replay and signature, not by
hardware guarantee.
