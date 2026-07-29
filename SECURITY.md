# SECURITY.md — FightLab Security Policy

**Version:** 1.0
**Last updated:** 2026-07-27T00:00:00Z

This document describes how to report a security vulnerability in
FightLab and what is in scope. It is not a general support channel.

---

## Reporting a vulnerability

- Email the maintainers at `security@fightlab.dev` (replace with the
  real address once the domain is live).
- Do not open a public issue for a security vulnerability.
- Include a description of the issue, the steps to reproduce, the
  impact, and any proof-of-concept.
- You will receive an acknowledgement within 72 hours (UTC).
- Coordinated disclosure: we will work with you on a fix and a release
  timeline, and credit you in the advisory unless you prefer to remain
  anonymous.

---

## What is in scope

- **Eval tampering.** Any way to produce a signed bout result JSON that
  does not match what the harness would produce with the same seed and
  env pin. This includes hash collisions, HMAC bypass, or result
  forgery.
- **Anti-cheating bypass.** Any way a submitted policy can manipulate
  the environment, the reward, the HP variable, the seed, or the
  scoring path without being caught by the anti-cheating pass in
  [REVIEW.md](REVIEW.md).
- **CI bypass.** Any way to make `pr-integrity.yml` report green when a
  check actually failed, or to skip a gate.
- **Sensitive paths guard bypass.** Any way a non-maintainer PR can
  touch `engine/`, `eval/`, `bench/`, `.gittensor/`, or `.github/`
  without failing the sensitive-paths guard.
- **Supply chain.** A malicious dependency or a typosquat in the import
  path that could run code during eval or CI.

---

## What is out of scope

- General questions about how to use FightLab (open an issue, not a
  security report).
- Theoretical attacks that require kernel-level access to the validator
  machine. The honest boundary in [EVAL-TRUST.md](EVAL-TRUST.md) already
  documents that a hostile validator with kernel access can tamper
  before signing; TEE attested eval (Phase 3) is the fix, not a security
  report.
- Social engineering of maintainers.
- DoS against the GitHub repo or the CI runners.

---

## Threat model (summary)

- The bout result is the trust anchor. If the result can be forged, the
  league is broken. The defense is the SHA-256 payload hash plus
  optional HMAC, plus the multi-seed design that makes a single-seed
  fluke visible.
- The eval path (`engine/`) is maintainer-owned. A contributor who can
  change the harness that scores them can cheat. The defense is the
  sensitive-paths guard and CODEOWNERS.
- The policy file is untrusted code. In Phase 3 (Docker isolation) it
  runs in a sandbox. Until then, treat any submitted policy as
  untrusted and do not run it outside the eval harness.

This summary is expanded in [EVAL-TRUST.md](EVAL-TRUST.md) and
[REVIEW.md](REVIEW.md).
