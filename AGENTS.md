# AGENTS.md — Instructions for AI Contributors

**Version:** 1.0
**Last updated:** 2026-07-27T00:00:00Z

This document is for contributors using AI coding tools (Claude Code,
Copilot, Cursor, Codex, etc.) to work on FightLab. It is the
metagraphed-style pattern: tell the agent the rules, the layout, and
what not to do.

If you are an AI agent reading this, follow it. If you are a human
using an AI tool, paste this into the tool's context before starting.

---

## What FightLab is

FightLab is a Gittensor (SN74) repository for autonomous humanoid
combat. Miners train fighting policies for the Unitree G1 (training in
Isaac Lab, bouts evaluated in MuJoCo) and
submit them via pull requests. Bouts are refereed by a trustless eval
harness (`engine/eval_harness.py`). The league tracks ELO and archives
kings.

Read these first:
- [REVIEW.md](REVIEW.md) — the contribution contract (gates, labels,
  anti-cheating).
- [EVAL-TRUST.md](EVAL-TRUST.md) — the eval trust model.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute.
- [RULESET.md](RULESET.md) — the bout ruleset (authoritative for
  semantics).
- [docs/architecture.md](docs/architecture.md) — the system
  architecture.

---

## Repo layout

```
fightlab/
  league.py          ELO league CLI (root entry point)
  miner_sdk.py       Miner SDK CLI (root entry point)
  engine/            Core sim + eval (maintainer-owned)
    eval_harness.py      Trustless referee: multi-seed, signed results
    real_bout_runner.py  Bridges harness to MuJoCo
    baselines.py         Starter policy templates
  lanes/             Competition config (weight classes, bot rosters)
  kings/             Published current king per lane (open weights)
  submissions/       PR-submitted policy bundles
  runs/              Fight artifacts with provenance (gitignored)
  tests/             Test files
  web/               Landing page (static HTML)
  docs/              Architecture and workflow docs
  .gittensor/        Intra-repo emission config
  .github/           CI workflows, CODEOWNERS
  RULESET.md         Official bout ruleset
  REVIEW.md          Contribution contract
  EVAL-TRUST.md      Eval trust model
  CONTRIBUTING.md    How to contribute
  ROADMAP.md         Roadmap
  SECURITY.md        Security policy
  AGENTS.md          This file
```

---

## Rules for AI agents

- **Do not touch maintainer-owned paths** unless the task explicitly
  asks for it and you are operating as the maintainer. The paths are
  `engine/`, `eval/`, `bench/`, `.gittensor/`, `.github/`, `kings/`,
  `RULESET.md`, `REVIEW.md`, `EVAL-TRUST.md`. A non-maintainer PR
  touching these fails CI.
- **Do not add AI-attribution trailers.** Do not add
  `Co-authored-by: Copilot` or similar to commits or PR bodies. CI
  rejects them.
- **Reference an issue.** Every PR must reference an issue
  (`Fixes #N` or `Refs #N`). If no issue exists, open one first.
- **Max 2 open PRs per contributor.** Do not open a third if two are
  already open.
- **Ship tests with code changes.** A change under `engine/` must ship
  a test change under `tests/`.
- **No emojis.** All docs and code comments are emoji-free. UTC
  timestamps only, ISO-8601 with a `Z` suffix.
- **No external dependencies in core modules.** `engine/`, `league.py`,
  and `miner_sdk.py` are standard-library-only. Do not add imports of
  third-party packages to these modules. (`real_bout_runner.py` may
  import `stable_baselines3` and `gymnasium` because it bridges to the
  training pod; that is the exception.)
- **Do not fake results.** Do not fabricate bout outcomes, coverage
  numbers, or test results. If a check fails, report it; do not invent
  a pass.

---

## What to work on

Good first tasks for an AI agent:
- Add a test under `tests/` for an existing `engine/` function that is
  not yet covered. This is a `fightlab:tooling` PR.
- Fix a lint warning in `league.py` or `miner_sdk.py`.
- Improve a doc under `docs/`.
- Add a baseline policy to `engine/baselines.py` (maintainer-owned; open
  an issue first).

Do not work on:
- Changing the scoring config (`.gittensor/config.json`).
- Changing the CI workflows (`.github/workflows/`).
- Adding subjective scoring layers (LLM judges). FightLab's judge is the
  physics sim.

---

## Local checks before opening a PR

```bash
ruff check .
pytest -q --cov=. --cov-fail-under=75
python engine/eval_harness.py selftest
```

All three must pass. CI runs the same checks.

---

## If you are an AI agent and unsure

Open an issue describing what you want to do and why. Do not open a PR
that is out of scope. Decisions are on the record.
