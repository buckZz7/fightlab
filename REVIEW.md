# REVIEW.md — FightLab Contribution Contract

**Version:** 1.0
**Last updated:** 2026-07-27T00:00:00Z
**Authority:** This document mirrors the scoring contract in
`.gittensor/config.json`. If this page and the registry JSON disagree, the
registry JSON wins and this page is the bug.

FightLab is a Gittensor (SN74) repository. Miners contribute fighting
policies via pull requests. Every PR passes through a 3-gate pipeline
before it earns an emission label. This document is the contract: it
defines the gates, the rubric, the label multipliers, and the
anti-cheating rules. No gate may be skipped by human override.

---

## Gate 1 — Automated

A machine checks the PR before a human looks at it. All checks run in
`.github/workflows/pr-integrity.yml`. A single red check closes the PR
automatically.

| Check                          | Rule                                                   |
|--------------------------------|--------------------------------------------------------|
| Lint                           | `ruff check .` clean                                   |
| Tests                          | `pytest -q` passes                                      |
| Coverage floor                 | `pytest --cov-fail-under=75`                           |
| Linked issue                   | PR description contains `Fixes #N` or `Refs #N`        |
| No AI-attribution trailers     | No `Co-authored-by:` line that names an AI tool        |
| Max 2 open PRs per contributor | Over-limit PRs auto-closed by `pr-limit.yml`           |
| Sensitive paths guard          | Non-maintainer PR touching `engine/`, `eval/`, `bench/`, `.gittensor/`, `.github/` fails |

A PR that fails Gate 1 is closed with a pointer to the failing check.
There is no human override. Fix the check and reopen.

---

## Gate 2 — Scope

A maintainer reviews the PR for scope fit **before** any bout is run.

- Is the PR a fighting policy submission, a ruleset fix, a harness
  improvement, or documentation? Out-of-scope PRs are closed with a
  pointer to an issue that would make them in scope.
- Max 2 open PRs per contributor at a time. This is enforced by CI in
  Gate 1 and re-checked here.
- A policy submission must land under `submissions/<miner>/<bundle>/`.
  A harness change must land under `engine/` or `eval/` and ship a test
  change under `tests/`.

A PR that fails Gate 2 gets the `out-of-scope` label and is closed with
a reason on the record.

---

## Gate 3 — Human rubric

A maintainer scores the PR against the rubric below. The label is
deterministic for policy PRs (it is a function of the bout outcome,
not a human read). For tooling/docs PRs the maintainer assigns the
`fightlab:tooling` label by inspection.

| Criterion            | Weight | What it means                                                         |
|----------------------|--------|-----------------------------------------------------------------------|
| Correctness          | High   | The code does what the PR claims. Tests cover the new behavior.       |
| Scope fit            | High   | The PR belongs in this repo and this lane.                            |
| Non-redundancy       | High   | The PR does not duplicate an existing policy or harness path.         |
| Quality              | Medium | Code is readable, idiomatic, and follows the repo conventions.       |
| Real-behavior-proof | Medium | For policy PRs: a signed bout result is attached. For harness PRs: the change is exercised by a test. |

A PR that fails Gate 3 gets a rejection reason on the record. Rejection
reasons are not subjective ("I don't like it"); they cite a rubric
criterion and the evidence.

---

## Label to multiplier table

The label is the scoring contract. Unlabeled PRs score zero
(`default_label_multiplier: 0.0`). Only labels prefixed `fightlab:` are
trusted by the SN74 validator bot (`trusted_label_pipeline: true`).

| Label                | Multiplier | Assigned when                                                            |
|----------------------|------------|--------------------------------------------------------------------------|
| `fightlab:winner`    | x4.0       | Submitted policy beats the current king by majority across 5 seeds.     |
| `fightlab:ko`        | x2.5       | Submitted policy wins by KO (opponent HP reaches 0) in the deciding seed.|
| `fightlab:decision`  | x1.5       | Submitted policy wins by decision (score margin, no KO).                |
| `fightlab:loss`      | x0.0       | Submitted policy loses the bout.                                         |
| `fightlab:draw`      | x0.0       | Bout ends in a draw across seeds.                                        |
| `fightlab:invalid`   | x0.0       | Submission fails validation (size, format, import, runtime error).       |
| `fightlab:tooling`   | x0.05      | Harness, eval, docs, tests, or infra PR (not a policy submission).      |

The label for a policy PR is deterministic: it is a function of the
signed bout result JSON produced by `engine/eval_harness.py`. A human
does not pick the label by reading the diff. The maintainer's job is to
run the bout, read the result, and apply the label the result dictates.

---

## Anti-cheating pass

After Gate 3 and before a label is applied, a maintainer runs an
anti-cheating pass over the diff. Any of the following closes the PR
regardless of the measured outcome:

- **Reward-hacking:** the policy manipulates the reward signal directly
  (e.g., patches the reward function, patches the HP variable, or
  returns hardcoded high-reward actions).
- **Env-manipulation:** the policy or the submission bundle modifies the
  environment, the bout runner, the seed, or the scoring path.
- **Disabled assertions:** the diff turns off an assertion, a guard, a
  test, or the damage gate.
- **Benchmark-detection branching:** the policy branches on the seed
  value or the opponent identity to behave differently during eval than
  in general.
- **Hardcoded outputs:** the policy returns a hardcoded action sequence
  keyed to a known opponent.

A PR flagged by the anti-cheating pass is closed with the `fightlab:invalid`
label and a note on the record naming the violation.

---

## Authority clause

The registry JSON (`.gittensor/config.json` in this repo, mirrored in
`master_repositories.json` on the SN74 validator) is the source of truth
for scoring. If this page and the registry disagree, the registry wins.
This page is maintained to match the registry; if you find a mismatch,
open an issue.

---

## Deterministic label from fight outcome

For policy PRs the label is not a human opinion. The bout is run by
`engine/eval_harness.py` across 5 seeds. The harness produces a signed
result JSON. The label is a pure function of that result:

- If the submission wins the majority of seeds and the deciding seed was
  a KO, the label is `fightlab:ko`.
- If the submission wins the majority of seeds and the deciding seed was
  a decision, the label is `fightlab:decision`.
- If the submission wins the majority of seeds and also beats the
  current king (a king-change bout), the label is `fightlab:winner`.
- If the submission loses the majority of seeds, the label is
  `fightlab:loss`.
- If the bout is drawn across seeds, the label is `fightlab:draw`.
- If the submission fails validation or the bout errors out, the label
  is `fightlab:invalid`.

The maintainer reads the result JSON, applies the label, and posts the
result hash in the PR thread. Any third party can recompute the hash and
replay the bout with the same seed and env pin to verify.
