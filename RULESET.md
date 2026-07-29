# FightLab Ruleset Specification

**Version 1.0**  
**Status: Official**  
**Last updated: 2026-07-27 (UTC)**  
**Governs: All sanctioned FightLab bouts**

---

## 1. Scope and Authority

This document is the official ruleset for FightLab, the autonomous humanoid
combat platform. FightLab evaluates fighting policies for the Unitree G1
humanoid inside the MuJoCo physics simulator. The platform is neutral: it
provides the arena, the rules, and the referee (the `eval_harness`). Miners
submit any policy that conforms to the submission protocol.

All sanctioned bouts are governed by this ruleset. Any bout executed by the
`EvalHarness` with a conforming `BoutConfig` is, by definition, a sanctioned
bout, and its signed result is binding on league standings and ELO.

Where this ruleset and the implementation disagree, this ruleset is
authoritative; the implementation is to be corrected.

---

## 2. Definitions

- **Fighter.** A submitted policy and its associated weights. Each fighter has
  a miner-chosen name, a policy file, and an ELO rating.
- **Bout.** A single competitive engagement between two fighters, consisting
  of one or more seeds.
- **Seed.** A single deterministic simulation run of a bout, identified by an
  integer seed that initializes all stochastic elements (initial pose noise,
  contact perturbations, referee coin-flips).
- **Policy.** A file containing a fighter's weights, submitted for
  evaluation. Must be a single file not exceeding the size cap.
- **Referee.** The `EvalHarness` — the trustless component that runs seeds,
  aggregates results, applies the damage gate, and signs bout results.
- **King.** The highest-rated fighter in the league at the time of a crown.
- **UTC.** All timestamps are Coordinated Universal Time, ISO-8601 with a `Z`
  suffix (e.g. `2026-07-27T14:33:01.123456Z`).

---

## 3. Bout Format

### 3.1 Weight Class

All fighters use the identical Unitree G1 body and the identical MuJoCo
model. There is exactly one weight class: **Open G1**. There are no weight
classes, height classes, or reach classes because there is no physical
variation between fighters. Every fighter brings the same chassis; the only
difference is the policy.

### 3.2 Bout Length

A single seed runs for at most `max_steps` simulation steps. Defaults:

| Parameter | Default | Notes |
|---|---|---|
| `max_steps` | 1000 | Per-seed step cap. |
| `sim_dt` | 0.0025 s | MuJoCo timestep. |
| Wall-clock sim time per seed | 2.5 s | `max_steps * sim_dt`. |

A seed ends when either:
1. A fighter is knocked out (Section 5.1), or
2. The step cap is reached without a knockout (the seed goes the distance).

### 3.3 Rounds

A bout is a single continuous round per seed. There are no multi-round
structures within a seed, and no rest periods within a seed.

### 3.4 Seeds per Bout

Every bout is contested across `N` seeds, `N >= 1`. The default is
**5 seeds** (`DEFAULT_SEEDS = 5`). Seeds are drawn from a fixed list
(`[1, 2, 3, 4, 5]` by default) so that any third party can reproduce the
exact same bout by replaying the same seeds.

The seed list is part of the signed bout result. Changing even one seed
invalidates the `payload_sha256` and is detectable as tampering.

### 3.5 Policy Submission Limits

| Limit | Default | Enforced |
|---|---|---|
| Policy file size cap | 50 MB (`50 * 1024 * 1024` bytes) | Before any seed runs; oversized submissions are rejected. |
| Policy file format | Single file, any format the runner accepts | The runner must be able to load and execute it. |

A policy that fails to load or that throws an uncaught exception during a
seed forfeits that seed.

---

## 4. Damage Model

Damage is the fundamental currency of a FightLab bout. It determines
knockouts, technical knockouts, and the damage gate.

### 4.1 How Damage Is Computed

Damage is computed from MuJoCo contact events. For each contact between a
fighter's striking body part and the opponent, the instantaneous damage
contribution is a function of the **relative velocity at the contact point**
and the **normal force** at that contact:

- **Relative velocity.** The relative linear velocity of the two contacting
  bodies at the contact point. Glancing, slow, or static contact produces
  negligible damage; high-speed, square-on contact produces high damage.
- **Force threshold.** Contacts below a minimum force threshold produce zero
  damage. This filters out incidental brush contacts (limbs grazing during
  stance transitions) that should not count as strikes.
- **Damage accumulation.** Per-contact damage is integrated over the
  timestep and added to the striker's running `damage_dealt` total for the
  seed.

The exact mapping (velocity, force) -> damage points is fixed by the
environment configuration and recorded in the `env_config` hash of every
signed bout result, so it is reproducible and auditable.

### 4.2 Hit Points (HP)

Each fighter has a **hit-point (HP) pool**, initialized to a fixed value at
the start of each seed. Damage received is subtracted from HP. A fighter's
HP is not visible to the opponent's policy; it is a referee-side bookkeeping
quantity.

A fighter is not directly knocked out by reaching zero HP (see Section 5 for
knockout criteria). HP reaching zero triggers a **technical knockout**
evaluation (Section 5.2).

### 4.3 The Damage Gate

The damage gate is an anti-avoidance rule. A fighter who wins the seed
majority by surviving without engaging does **not** win the bout.

**Rule.** The overall winner of a bout must have dealt at least
`damage_threshold` total damage across the entire seed set. The default
threshold is **50.0** (`DEFAULT_DAMAGE_THRESHOLD = 50.0`).

If a fighter wins the strict seed majority but fails the damage gate, the
result is **downgraded to a draw** with `damage_gate_passed = false` and a
`gate_note` explaining the downgrade. The gate is applied *after* aggregation
and *before* the result is signed.

The damage gate is per-bout, not per-seed. It is evaluated on the sum of the
winner's `damage_dealt` across all seeds in the bout.

---

## 5. Win Conditions

A bout can end in one of four outcomes, determined per seed and then
aggregated (Section 7).

### 5.1 Knockout (KO)

A knockout occurs when a fighter's body falls (the torso or pelvis drops
below a defined height threshold, or the fighter is otherwise unable to
maintain an upright posture) **and** the fighter fails to recover within a
referee count.

- **Fall.** Detected by the environment when the fighter's upright condition
  is violated (specific body-segment heights or orientation thresholds, fixed
  in `env_config`).
- **Count.** A recovery window of a fixed number of steps begins at the
  moment of the fall. If the fighter regains an upright posture within the
  window, the bout continues. If not, the fall is ruled a knockout and the
  seed ends with `terminated = true`.
- **Award.** The standing fighter is awarded the seed as the winner.

### 5.2 Technical Knockout (TKO)

A technical knockout occurs when a fighter's accumulated damage received
exceeds the TKO threshold (HP reaches zero) and the referee judges the
fighter unable to continue, or when accumulated damage so exceeds the
threshold that continued participation is pointless.

- **Trigger.** A fighter's `damage_received` crosses the TKO threshold
  (equivalently, HP <= 0).
- **Award.** The fighter who dealt the damage is awarded the seed. The
  damaged fighter is ruled unable to continue (`terminated = true`).

### 5.3 Decision

If a seed reaches the step cap without a KO or TKO, the seed goes the
distance and is decided by the scoring criteria in Section 6. The seed
winner is the fighter with the higher score; a tie in score is a draw for
that seed (`terminated = false`).

### 5.4 Disqualification (DQ)

A fighter is disqualified for a foul (Section 8) severe enough to warrant
it, or for any policy behavior that circumvents the simulation (Section 8.4).
A DQ ends the seed immediately and awards the seed to the opponent.

### 5.5 No Contest

If a seed cannot be completed due to a simulation failure, a crash in the
referee, or both fighters being rendered unable to continue simultaneously,
the seed is ruled a no contest (`winner = None`) and does not count toward
either fighter's seed total.

---

## 6. Scoring (Decision Criteria)

When a seed goes the distance, the winner is determined by a composite
score. The score is a scalar; the fighter with the higher score wins the
seed. Ties are draws.

The composite score is a weighted sum of four criteria. The weights are
fixed in `env_config` and recorded in every signed bout result.

| Criterion | What it measures |
|---|---|
| **Effective damage** | Total `damage_dealt` by the fighter this seed. This is the dominant component. |
| **Aggression** | Forward movement toward the opponent and frequency of strike attempts (contact-initiating actions), as opposed to circling away or turtling. |
| **Control** | Time spent in a dominant positional relationship (e.g. opponent retreating, opponent below the fighter in a grapple, opponent off-balance). |
| **Effective striking** | Fraction of attempted strikes that landed above the force threshold (Section 4.1). Rewards accuracy and clean contact over wild swings. |

**Interpretation.** Damage is the primary determinant. Aggression, control,
and effective striking act as tiebreakers and as guardrails against purely
defensive strategies — but the damage gate (Section 4.3) is the hard
guardrail. A fighter who lands more damage almost always wins the decision;
the other criteria resolve genuinely close seeds.

The `score_a` / `score_b` fields in a `SeedResult` carry this composite
score. The damage gate operates on `damage_dealt`, not on the composite
score.

---

## 7. Legal and Illegal Techniques

### 7.1 Legal Techniques

The G1 humanoid is permitted to use any technique achievable within the
MuJoCo simulation that does not violate Section 7.2. This includes, where
the body's kinematics permit:

- **Strikes.** Punches, kicks, elbows, knees, and any other limb-driven
  impact. Open-hand and closed-hand strikes are equally legal.
- **Grappling.** Clinching, holding, and applying joint forces through
  contact.
- **Takedowns.** Forcing the opponent to the ground by legal contact,
  including trips, throws, and sweeps.
- **Pushing and positional control.** Forcing the opponent backward or into
  the arena boundary.

The arena is bounded. Forcing an opponent out of the ring (if the
environment defines a ring-out) is legal and counts as a KO equivalent.

### 7.2 Illegal Techniques

Because FightLab is a simulation, some real-world fouls (eye gouging,
groin strikes, biting, fish-hooking) have no physical analogue in MuJoCo
and are not enumerated. The illegal-technique list is defined entirely in
terms of what the simulation *can* express but the ruleset *prohibits*:

- **Targeting the back of the head / spine.** Contact whose primary point
  of impact is the rear skull or spinal column, above a force threshold, is
  illegal. (In the current G1 contact model this is enforced as a contact-
  site rule; see `env_config`.)
- **Striking a downed opponent.** Once an opponent has fallen and the
  recovery count (Section 5.1) is in progress, additional strikes to the
  downed opponent are illegal. The count and the fall detection are
  referee-side; the standing fighter must disengage.
- **Leaving the arena.** A fighter that exits the legal arena volume by
  its own locomotion (as opposed to being forced out) forfeits the seed.
- **Stalling / refusal to engage.** Sustained avoidance of contact with no
  strike attempts and no forward movement is a foul. This overlaps with the
  damage gate but is enforced per-seed by the referee, not only at
  aggregation.
- **Simulation circumvention.** Any policy behavior that exploits a
  simulation artifact to produce an effect impossible in a physical G1
  (e.g. self-interpenetration to gain an impossible lever, NaN injection,
  action clipping exploits, zero-energy teleportation through contact
  resolution). This is the catch-all and carries the harshest penalty
  (Section 8.4).

### 7.3 What Is Not a Foul

- Accidental low strikes (the G1 has no anatomically protected region in
  sim) are not fouls.
- Knocking an opponent down with a legal strike is not a foul; the
  downed-opponent rule applies only to *additional* strikes during the count.
- Ground-and-pound on an opponent that is still within the recovery window
  but has not yet been ruled down is legal.

---

## 8. Fouls and Penalties

### 8.1 Foul Detection

Fouls are detected by the referee during seed execution. A foul is recorded
with the offending fighter, the step at which it occurred, and a severity.

### 8.2 Penalty Ladder

| Severity | Example | Penalty |
|---|---|---|
| **Minor** | Brief stalling; incidental illegal contact with negligible force | Formal warning; no score change. Repeated minors escalate. |
| **Major** | Sustained stalling; striking a downed opponent during the count; repeated minors | Point deduction (subtract from the offender's composite score for the seed) and/or a forced position reset. |
| **Flagrant** | Simulation circumvention; intentional and repeated illegal contact; refusal to engage to the point of making the seed a non-event | Seed forfeiture (opponent awarded the seed) or full disqualification. |

### 8.3 Point Deductions

A point deduction subtracts a fixed amount from the offender's composite
score for that seed. Because the composite score determines the decision
(Section 6), a deduction can flip a close seed. Point deductions are
recorded in the per-seed result and are auditable.

### 8.4 Disqualification

A fighter is disqualified from a seed (or, for flagrant simulation
circumvention, from the bout) when a flagrant foul is committed. DQ awards
the affected seed to the opponent. For a bout-level DQ, all remaining seeds
are awarded to the opponent and the bout result is recorded with the DQ
note.

---

## 9. Multi-Seed Aggregation

### 9.1 Per-Seed Winner

Each seed produces one of: **A wins**, **B wins**, or **draw**. The per-seed
winner, damage dealt by each fighter, steps survived, and composite score
are recorded in the `SeedResult` and included in the signed bout payload.

### 9.2 Aggregate Winner

The overall bout winner is determined by **strict majority of seeds**:

- A fighter must win strictly more than half of the seeds to be the overall
  winner.
- If neither fighter achieves a strict majority (including the case where
  both win equal seeds, or the damage gate downgrades a majority winner),
  the bout is a **draw** (`overall_winner = None`).

Formally, with `n` seeds:
- A is the overall winner iff `seeds_won_a > seeds_won_b` **and**
  `seeds_won_a > n / 2`.
- B is the overall winner iff `seeds_won_b > seeds_won_a` **and**
  `seeds_won_b > n / 2`.
- Otherwise, draw.

### 9.3 Damage Gate at Aggregation

After the raw majority winner is computed, the damage gate (Section 4.3)
is applied. If the raw winner dealt less than `damage_threshold` total
damage across the seed set, the result is downgraded to a draw and
`damage_gate_passed` is set to `false` with an explanatory `gate_note`. The
gate is mandatory; there is no override.

### 9.4 Confidence

The `confidence` field reports the fraction of seeds won by the overall
winner, in `[0, 1]`. For a draw, confidence is the maximum seed-share held
by any fighter (so a near-draw reads as low confidence). Confidence is an
advisory field; it does not change the verdict.

### 9.5 Integrity

The full per-seed results, the aggregate, the seed list, both policy hashes,
and the `env_config` hash are all part of the signed payload. The
`payload_sha256` is the SHA-256 of the canonical (sorted-key, no-space)
serialization of the payload. Any edit to any field invalidates the hash.
An optional `hmac_sha256` (computed when a signing key is supplied) adds
authentication on top of integrity.

Any party can independently recompute a bout by replaying the same seeds
against the same policy hashes with the same `env_config` and verifying
that the resulting `payload_sha256` matches.

---

## 10. League and ELO

### 10.1 ELO System

Fighters are ranked by ELO. The default starting rating is **1000**
(`DEFAULT_ELO`). The K-factor is **32** (`K_FACTOR`). The expected-score
formula is standard:

```
E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
```

After a bout, ratings are updated by:

```
R_A' = R_A + K * (S_A - E_A)
R_B' = R_B + K * (S_B - E_B)
```

where `S_A` is 1.0 for a win, 0.0 for a loss, 0.5 for a draw, and `S_B = 1 - S_A`.

### 10.2 Bout Effect on Standings

- A win/loss updates both fighters' ELO and their W-L records.
- A draw updates both fighters' ELO (toward each other) and their draw
  counts.
- The damage-gate downgrade to a draw is recorded as a draw for ELO
  purposes.

### 10.3 Scheduling

The league supports round-robin scheduling. A single round-robin pairs every
fighter once; a double round-robin pairs every pair twice (home and away).
The schedule is generated by the circle method and is deterministic given
the fighter list.

---

## 11. King Challenges

### 11.1 The King

The **King** is the highest-ELO fighter in the league at the time a crown
is issued. The King is the reigning champion.

### 11.2 Earning a Title Shot

A challenger earns a title shot by reaching the top of the contender pool.
The specific protocol:

1. **Rating threshold.** A challenger must be within a configured ELO gap of
   the King (default: within 0 ELO, i.e. must be the top contender, or must
   have beaten the King in a prior sanctioned bout).
2. **Minimum bouts.** A challenger must have a minimum number of completed
   sanctioned bouts (default: 5) to ensure the ELO is statistically
   meaningful and not the product of a single upset.
3. **Challenge bout.** The title bout is a standard sanctioned bout
   (Sections 3-9) between the King and the challenger.

### 11.3 Crown

When a challenger defeats the King in a sanctioned bout, the challenger
becomes the new King. The crown is issued by archiving the new King's
weights and metadata (Section 12).

When the King retains the title (wins or draws the challenge bout), the
King remains King and no new crown is issued.

### 11.4 Vacancy

If the King is removed from the league (e.g. policy withdrawn), the title
is vacant until the next top contender is crowned by the league operator.

---

## 12. Open Weights Policy

### 12.1 Principle

When a fighter is crowned King, their policy weights are published in full.
This is the **open weights** commitment: a King's power is verifiable and
reproducible by any third party, and the weights become a public baseline
that future challengers must surpass.

### 12.2 What Is Published

Upon crowning, the following are archived to `kings/<fighter-slug>/`:

1. **Policy weights file.** The exact policy file that was hashed and
   executed in the bouts that earned the crown. This is the binary that
   produces the King's behavior. Its SHA-256 matches the `policy_a` /
   `policy_b` hash recorded in the signed bout results.
2. **`king.json` metadata sidecar.** A JSON file containing:
   - `fighter_name` — the King's name.
   - `elo` — the ELO rating at the time of crowning.
   - `record` — the W-L-D record at the time of crowning.
   - `wins`, `losses`, `draws`, `bouts` — the numeric breakdown.
   - `policy_path` — the original submitted path.
   - `archived_path` — the path to the archived weights copy.
   - `crowned_at` — the UTC timestamp of crowning.

### 12.3 What May Optionally Be Published

Miners are encouraged but not required to also publish, alongside the
weights:

- **Training configuration.** The hyperparameters, reward shaping, and
  curriculum used to train the King. This is the recipe that produced the
  weights and helps the community reproduce and improve on the result.
- **Evaluation results.** The signed bout result JSON for the bout(s) that
  earned the crown, so any party can verify the crown without re-running.

### 12.4 Integrity Guarantee

The archived weights file is byte-identical to the file that was hashed
for the signed bout results. Any party can hash the published weights and
confirm it matches the `sha256` in the bout payload, then re-run the bout
under the same seeds and `env_config` to confirm the King's behavior
reproduces. This is the trustless guarantee: the crown is not a claim, it
is a verifiable artifact.

### 12.5 Licensing

Published King weights are released under an open license that permits
reproduction, study, and use as a training baseline. The specific license
is set by the league operator and recorded with the archive. Miners who
submit a policy that becomes King consent, by submission, to this
publication.

---

## 13. Configuration Defaults

The following defaults are enforced by the `eval_harness` and may be
overridden per-bout by the league operator. Any override changes the
`env_config_hash` and is therefore auditable.

| Parameter | Default | Reference |
|---|---|---|
| `DEFAULT_SEEDS` | 5 | Section 3.4 |
| `DEFAULT_MAX_STEPS` | 1000 | Section 3.2 |
| `sim_dt` | 0.0025 s | Section 3.2 |
| `DEFAULT_DAMAGE_THRESHOLD` | 50.0 | Section 4.3 |
| `DEFAULT_SIZE_CAP_BYTES` | 50 MB | Section 3.5 |
| `DEFAULT_ELO` | 1000 | Section 10.1 |
| `K_FACTOR` | 32 | Section 10.1 |
| `ELO_DIVISOR` | 400 | Section 10.1 |
| `obs_dim` | 41 | `env_config` |
| `act_dim` | 17 | `env_config` |

---

## 14. Versioning

This is **Version 1.0** of the FightLab ruleset. Changes to the ruleset
that alter the meaning of a signed bout result (e.g. changing the damage
threshold, the KO criteria, or the scoring weights) constitute a new
major version and require re-crowning the King under the new ruleset.
Cosmetic and clarifying changes do not require re-crowning.

All changes are timestamped in UTC and recorded in the ruleset history.

---

*End of Ruleset.*
