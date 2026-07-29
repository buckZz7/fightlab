# CONTRIBUTING.md — How to Contribute to FightLab

**Version:** 1.0
**Last updated:** 2026-07-27T00:00:00Z

FightLab is a Gittensor (SN74) repository. Miners contribute fighting
policies and harness improvements via pull requests. This document is
the how-to. The contribution contract (gates, labels, anti-cheating)
lives in [REVIEW.md](REVIEW.md). The eval trust model lives in
[EVAL-TRUST.md](EVAL-TRUST.md). Read both before opening a PR.

---

## Branch model

- PRs target the `test` branch.
- A maintainer promotes `test` to `main` after a PR lands and passes the
  3-gate pipeline.
- Do not open PRs against `main`. They will be closed and you will be
  asked to retarget `test`.

---

## How to submit a fighting policy

1. **Train.** Train a policy against the 41D observation / 17D action
   interface documented in `engine/baselines.py` and `miner_sdk.py`.
   Any framework is fine; the submission is a policy file on disk.
2. **Validate locally.** Run `python miner_sdk.py submit <policy> --name
   <name> --miner <miner>` to validate the file (exists, under size cap,
   hashes cleanly) and register it with the local league.
3. **Run a local challenge.** Run `python miner_sdk.py challenge king
   --name <name>` to run a multi-seed bout against the current king and
   produce a signed result JSON. Check that you win the majority of
   seeds.
4. **Open a PR.** Place the policy file under
   `submissions/<miner>/<bundle>/` in the PR. Attach the signed result
   JSON from your local challenge. Reference an issue
   (`Fixes #N` or `Refs #N`).
5. **Wait for eval.** A maintainer runs the bout with the validator's
   env pin and applies the deterministic label. See [REVIEW.md](REVIEW.md)
   for the label table.

---

## Local checks

Run these before opening a PR. They are the same checks CI runs.

```bash
# Lint
ruff check .

# Tests
pytest -q

# Coverage floor (must be >= 75)
pytest --cov=. --cov-fail-under=75

# Eval harness self-test (no MuJoCo required)
python engine/eval_harness.py selftest
```

---

## What belongs where

| Path              | What goes here                                                | Who can touch it          |
|-------------------|--------------------------------------------------------------|--------------------------|
| `engine/`         | Core sim + eval: `eval_harness.py`, `real_bout_runner.py`, `baselines.py` | Maintainer review required |
| `eval/`           | Reserved for future eval scripts                              | Maintainer review required |
| `bench/`          | Reserved for future benchmark scripts                         | Maintainer review required |
| `lanes/`          | Competition config: weight classes, bot rosters              | Maintainer review required |
| `kings/`          | Published current king per lane (open weights + metadata)   | Maintainer (promotion only) |
| `submissions/`    | PR-submitted policy bundles                                   | Contributors              |
| `runs/`           | Fight artifacts with provenance (gitignored, not committed)  | Local only                |
| `tests/`          | Test files                                                    | Contributors (with code change) |
| `web/`            | Landing page (static HTML)                                    | Contributors              |
| `docs/`           | Architecture and workflow docs                               | Contributors              |
| `.gittensor/`     | Intra-repo emission config                                    | Maintainer review required |
| `.github/`        | CI workflows, CODEOWNERS                                      | Maintainer review required |
| `league.py`       | ELO league CLI (root entry point)                             | Contributors              |
| `miner_sdk.py`    | Miner SDK CLI (root entry point)                              | Contributors              |
| `RULESET.md`      | Official bout ruleset (authoritative for semantics)           | Maintainer review required |

A code change under `engine/` must ship a test change under `tests/`.
CI enforces this.

---

## Out of scope

- Anything that is not a fighting policy, a harness/eval improvement, a
  ruleset fix, or documentation. Open an issue first if you are unsure.
- Changes to the scoring config (`.gittensor/config.json`) or CI
  workflows (`.github/workflows/`) from non-maintainers. These are
  maintainer-owned paths; a non-maintainer PR touching them fails the
  sensitive-paths guard.
- LLM-judge style scoring. FightLab's judge is the physics sim. Do not
  add subjective scoring layers.
- Marketing copy, sponsor logos, or anything that is not engineering or
  ruleset content.

---

## Rules

- **Max 2 open PRs per contributor.** Over-limit PRs are auto-closed by
  `pr-limit.yml`. Finish one before opening another.
- **No AI-attribution trailers.** Do not add `Co-authored-by: Copilot`
  or similar lines to commits or PR bodies. PRs with AI-attribution
  trailers fail CI.
- **PR must reference an issue.** The PR description must contain
  `Fixes #N` or `Refs #N`. PRs without a linked issue fail CI.
- **No human override skips a red gate.** If CI is red, fix the check.
  Do not ask a maintainer to merge a red PR.

---

## Conventions

- All timestamps are UTC, ISO-8601 with a `Z` suffix.
- JSON files are the single source of truth (`league_state.json` for
  league state; signed bout result JSON for bout results).
- No emojis, no external dependencies in the core modules, no network
  calls from the eval path.
- Where `RULESET.md` and the implementation disagree, `RULESET.md` is
  authoritative; the implementation is to be corrected.

---

## Questions

Open an issue. Do not DM a maintainer. Decisions are on the record.
