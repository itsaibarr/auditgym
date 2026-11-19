# AuditGym: spec (v1)

The source of truth for requirements is `research/case.pdf` (Genesis FinTech 2026). Domain rules are in `.claude/skills/audit-domain-expert`, contract and grader rules in `.claude/skills/rl-env-grader-expert` (+ `references/contract-spec.md`, `references/adversarial-test-matrix.md`), and the scoring rubric in `.claude/skills/genesis-jury`. This file records **decisions** and the places where we deviate from or narrow those skills. When the skills and this file disagree, this file wins.

## Goal
A reproducible, offline RL environment plus grader for AI financial auditors. It must score ≥90/100 on the jury rubric with no blockers.

## Settled decisions
- **Language and runtime:** Python 3.11; runtime deps `msgspec`, `tzdata`, `rich`; test deps `pytest`, `hypothesis`, `pytest-timeout`; optional `gymnasium`; locked with uv. MIT license. Docs in Russian, code in English.
- **CLI:** `auditgym` (argparse + Rich). Commands: `check`, `generate`, `run`, `compare`, `grade`, `replay`, `leakscan`, `perturb`, `play`, `schema`. When piped, output is plain markdown/JSON.
- **Isolation:** in eval mode the policy is a child process (JSON lines) running with a scrubbed environment in an empty temp cwd, under rlimits and timeouts. The oracle is runner-internal.
- **Seed salt:** generation uses `hmac(salt, seed)`. The open splits use the published default salt. For closed runs the jury sets `AUDITGYM_SALT` in the runner, and the child never sees it. A policy that imports our generator therefore cannot brute-force the seed and regenerate the truth.
- **Data model:**
  - Ledger amounts are in base currency; the transaction currency lives on the documents.
  - Rates are expressed as base per 1 source unit. Convert once per document total, ROUND_HALF_EVEN. The rate date is the company-local date of the doc; if it is missing, use the last rate on or before it.
- **Type slugs:** `duplicate_invoice`, `payment_allocation`, `fx_conversion`, `period_cutoff`.
- **Interacting scenarios:** X1 (partial foreign-currency payment at the period boundary, 3 linked issues) and X2 (foreign duplicate plus credit note with wrong FX, 2 linked issues). Each linked issue is matched separately.
- **IDs:**
  - `affected_ids` and evidence use primary keys: ledger `row_id` = `L-<je>-<line>`, docs `doc_id` = `D-<n>`, fx `FX-<date>-<SRC>-<TGT>`, entities `C-<n>`. Business keys `invoice_id`/`payment_id` are referenced by the ledger.
  - IDs are assigned after injection in canonical (time) order. The paired check uses a seeded bijection.
- **Tolerance:** `amount_tolerance_minor = 1` (uniform, published). Every injected delta is ≥100.
- **Reward:**
  - The shared reward is `clip(2*F1-1-0.002*inspection_steps,-1,1)`, where `inspection_steps` = all consumed steps before submit, including errors.
  - Malformed findings give exactly −1.
  - Truncation (100 steps or 60 s) grades empty findings.
  - `reward_experimental` (a per-type-weighted variant) is reported separately and never mixed into the shared reward.
- **Modes:** `train` returns the terminal reward. `eval` always returns 0 with minimal `info`; grading happens after the episode in the runner.
- **Matcher:** deterministic greedy over unique `(type, sorted affected_ids)` keys. IDs are compared exactly, without normalization. Evidence must be a superset of one allowed set, and every evidence ID must exist.
- **Splits:**
  - `data/train` holds 10 episodes and `data/dev` 10, generated from versioned configs `configs/{train,dev}.json`. They include clean, single-error, X1 and X2 episodes.
  - Closed evaluation is jury-side; `configs/closed_example.json` shows its shape.
- **Baselines:**
  - `random`: seeded.
  - `reconcile`: truth-free rule engine; must fit in 100 steps at maximum size.
  - `flag_all`: 30 shotgun findings.
  - `oracle`: submits at step 1, so its reward is exactly 1.0.

## Scope narrowing (YAGNI)
- **Distractors (required set):**
  - D1 recurring equal amounts
  - D2 split payment
  - D3 bulk payment
  - D4 genuine open balance
  - D5 credit note + net payment
  - D7 overpayment + refund
  - D8 realized FX gain/loss
  - D9 weekend rate
  - D10 tz-boundary clean twin
  - D12 bank fee
  - D14 same amount, different customer
  - D16 near-identical invoice numbers
  - D17 on-account receipt

  Everything else is optional.
- **Out of scope:** no AP side, no revaluation, no web UI.
- **Training:** an optional, separate `training/` folder. It runs on Kaggle and never runs in `check`.

## Acceptance (what `auditgym check` asserts; exit 0/1/2)
1. Contract: reset/step shapes, the 5 actions, 50-row pages, stable cursors, bounded errors that consume a step, 100-step limit, terminated/truncated semantics, the episode_over behavior, eval reward of 0.
2. Data: I1–I8 hold before injection on every train/dev episode and on N fresh seeds. After injection only the declared invariant breaks, by exactly the expected amount. An independent re-checker reproduces every truth amount from the visible records in its evidence set.
3. Reward: the full adversarial matrix, including the case's 5 orderings, guessed IDs, unsupported evidence, wrong currency, resubmission, malformed input and timeout.
4. Reproducibility: same seed/config/actions give identical bytes, also across `PYTHONHASHSEED`. Permutation plus ID rename preserves the oracle reward and the mapped-submission TP/FP/FN.
5. Isolation: leakscan on all trajectories, truth-swap invariance, the child process can't see the salt or truth.
6. Baseline ordering: oracle = 1 > reconcile > random/flag_all on dirty books; flag_all = −1 on clean.
