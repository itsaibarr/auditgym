# AuditGym: build plan

The order follows the case's advice (§08): contract first, then all four error types plus clean books, then interacting errors, isolation and attacks.

| # | Task | Acceptance |
|---|---|---|
| T1 | Scaffold: pyproject (uv, pins), package layout, MIT license, `.gitignore` | `uv run python -c "import auditgym"` |
| T2 | Core utils: `money` (ISO exponents, `convert`), `rng` (stage streams on `.random()`), `canon` (canonical JSON), `calc` (Decimal ast calculator), `timeutil` (tz, period) | unit tests |
| T3 | World + posting engine: customers, invoices, payments, CNs, refunds, fees, opening AR, expenses, fx random walk, the D-set distractors | clean books pass I1–I8 for 200 seeds |
| T4 | Injectors E1–E4, X1, X2 → recorded books + provenance + derived truth; post-injection gate; independent `recheck` | every issue rechecks exactly; only the declared invariant breaks |
| T5 | IDs assigned after injection; canonical ordering; `perturb` bijection plus shuffle; episode writer/loader; configs; splits | permuted episode gives the same oracle reward |
| T6 | Env: reset/step, 5 actions, pages/cursors, errors, budget, modes, schemas (msgspec), JSON Schema export | contract tests |
| T7 | Matcher + reward + grade (+ `reward_experimental`, match trace) | adversarial matrix |
| T8 | Runner: child-process policy host, rlimits/timeouts, trajectories/results/summary; baselines random/reconcile/flag_all/oracle | baseline ordering asserted |
| T9 | CLI: check, generate, run, compare, grade, replay, leakscan, perturb, play, schema | `auditgym check` exits 0 |
| T10 | Dockerfile + offline `docker run --network none auditgym check` | passes in Docker |
| T11 | Docs (RU): README, DESIGN.md, failure analysis, limitations; data/train, data/dev; trajectories | jury skill checklist |
| T12 | Council loop (3× Opus) until all ≥9 | scores logged in `docs/COUNCIL.md` |
| T13 | Demo video (VHS) + presentation | files in `submission/` |
| T14 | Optional: Kaggle Qwen3-14B base vs RFT eval | honest report in `training/REPORT.md` |
