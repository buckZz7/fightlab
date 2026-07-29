# docs/architecture.md — FightLab System Architecture

**Version:** 1.1
**Last updated:** 2026-07-28T00:00:00Z

This document describes the FightLab system architecture: the pipeline,
the component map, and the data flow. It is the reference for how the
pieces fit together. For bout rules, see [RULESET.md](../RULESET.md).
For the contribution contract, see [REVIEW.md](../REVIEW.md). For eval
trust, see [EVAL-TRUST.md](../EVAL-TRUST.md). For the policy/bout
interface miners conform to, see
[docs/policy-contract.md](policy-contract.md).

---

## Pipeline

```
train                submit         eval              record          crown          render
  |                    |              |                 |               |              |
  v                    v              v                 v               v              v
miner trains a        miner_sdk.py   engine/           league.py       league.py      render (pod)
policy (any stack)    submit         eval_harness.py   records the     archives       (not bundled)
                      validates +    runs the series,  title fight,    king weights
                      registers      signs the         updates the     + metadata
                                     result           reign ledger
```

1. **Train.** A miner trains a policy using any stack (see
   [docs/v2-pipeline.md](v2-pipeline.md) for the reference recipe). The
   contract is docs/policy-contract.md (`fightlab-contract=2`, 73D obs /
   29D joint targets).
2. **Submit.** `miner_sdk.py submit` validates the policy bundle (exists,
   under size cap, hashes cleanly) and registers the fighter in the
   league.
3. **Eval.** `engine/eval_harness.py` runs a multi-seed series (default 5
   seeds), applies the damage gate, aggregates results, and produces a
   signed result JSON (SHA-256 payload hash + optional HMAC). Any third
   party can recompute the hash (`miner_sdk.py verify result.json`) and
   replay the bout with the same seed and env pin.
4. **Record.** `league.py` records the outcome in the reign ledger
   (`league_state.json`, format `fightlab-league=2`): reigns, title
   fights, submission decisions. There is no ELO — the belt is the only
   ranking.
5. **Crown.** `league.py crown` archives the reigning king's open
   weights and metadata to `kings/<fighter-slug>/`.
6. **Render.** Bout videos are rendered from signed results for
   promotion and review (on the GPU pod, not bundled here).

---

## Component map

| File                          | Role                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| `league.py`                   | King-of-the-hill ledger: reigns, title fights, submissions, king archiving (CLI + library). Root entry point. |
| `miner_sdk.py`                | Miner entry point: submit, evaluate, check status (CLI + library). Root entry point. |
| `engine/eval_harness.py`      | Trustless referee: multi-seed bouts, signed results, damage gate. Maintainer-owned. |
| `engine/real_bout_runner.py`  | Bridges the eval harness to MuJoCo (`G1FighterEnv`). Maintainer-owned. |
| `engine/baselines.py`         | Starter policy templates: `RandomPolicy`, `ScriptedPolicy`. Maintainer-owned. |
| `lanes/`                      | Competition config: weight classes, bot rosters. Placeholder.       |
| `kings/`                      | Published current king per lane: open weights + metadata sidecar.   |
| `submissions/`                | PR-submitted policy bundles. Winner's bundle is cleared on promotion. |
| `runs/`                       | Fight artifacts with provenance (gitignored, not committed).         |
| `tests/`                      | Test files. A code change under `engine/` must ship a test change here. |
| `web/index.html`              | Static HTML landing page. No build step.                             |
| `.gittensor/config.json`      | Intra-repo emission config: label multipliers, eligibility, scoring. Mirrors the SN74 registry. |
| `.github/CODEOWNERS`          | Maintainer-owned paths. PRs touching these require maintainer review. |
| `.github/workflows/pr-integrity.yml` | CI: lint, tests, coverage floor, linked issue, no AI trailers, max 2 PRs, sensitive paths guard. |
| `RULESET.md`                  | Official bout ruleset. Authoritative for bout semantics.             |
| `REVIEW.md`                   | Contribution contract: 3-gate pipeline, label table, anti-cheating. |
| `EVAL-TRUST.md`               | Eval trust model: what is deterministic, reproducible, and the honest boundary. |
| `CONTRIBUTING.md`             | How to contribute: branch model, local checks, what belongs where.  |
| `ROADMAP.md`                  | Roadmap: Phase 1 (first king) through Phase 4 (sim-to-real).         |
| `SECURITY.md`                 | Security policy: how to report, what is in scope.                   |
| `AGENTS.md`                   | Instructions for AI contributors.                                    |

---

## Data flow

JSON state files are the single source of truth. Each is written
atomically (temp file + `os.replace`) to prevent corruption on crash.

```
league_state.json        Reign ledger (fightlab-league=2): reigns, title fights, submission decisions.
  ^                      Written by league.py. Read by miner_sdk.py and web/.
  |
  +-- league.py (record_bout, crown)
  |
  +-- miner_sdk.py (submit, challenge -> records bout)
  |
  +-- web/index.html (reads for leaderboard)

bout_result.json          Signed bout result: seeds, policy hashes, env config hash,
  ^                      per-seed results, aggregate, payload_sha256, optional HMAC.
  |                      Written by engine/eval_harness.py. Verified by anyone.
  |
  +-- engine/eval_harness.py (run_bout -> signs result)
  |
  +-- engine/eval_harness.py verify (checks payload hash)
  |
  +-- league.py (reads aggregate to record the bout)

kings/<slug>/king.json    King metadata sidecar: fighter name, ELO, crown timestamp,
  ^                      policy path, policy hash, crowning result hash.
  |
  +-- league.py crown (writes on promotion)
  |
  +-- web/index.html (reads for current king display)

.gittensor/config.json    Emission config: label multipliers, eligibility, scoring.
  ^                      Mirrors the SN74 registry JSON. Maintainer-owned.
  |
  +-- SN74 validator bot (reads to score PRs)
  |
  +-- REVIEW.md (mirrors the label table)
```

### Provenance

- Every signed bout result pins the seed list, both policy file hashes,
  and the env config hash. A change to any field invalidates the
  payload hash.
- Every king archive entry (`kings/<slug>/king.json`) records the
  crowning bout result hash, so a king's reign is traceable to a
  reproducible bout.
- `runs/` holds local fight artifacts (trajectories, logs) with
  provenance refs, but is gitignored and not committed. The committed
  truth sources are the JSON state files above.

---

## Trust anchors

- `RULESET.md` is authoritative for bout semantics. Where the ruleset
  and the implementation disagree, the ruleset wins; the implementation
  is corrected.
- `.gittensor/config.json` is authoritative for scoring. Where this
  page and the registry disagree, the registry wins
  (see [REVIEW.md](../REVIEW.md) authority clause).
- Signed bout result JSON is authoritative for bout outcomes. The
  label on a policy PR is a deterministic function of the result, not a
  human read (see [REVIEW.md](../REVIEW.md) label table).

These three anchors and their precedence are the core of the trust
model. Everything else is implementation.
