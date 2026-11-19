---
name: genesis-jury
description: Score or review AuditGym the way the Genesis Hackathon FinTech 2026 jury will (100-point rubric, blockers, hidden checks, submission completeness). Use when estimating the project's jury score, checking whether the submission is complete, writing tests against likely hidden checks, or acting as the "jury replica" council reviewer.
---

# Genesis jury replica (AuditGym)

The jury scores the **environment and grader**, not our agent's reward. Source: `research/case.pdf` (Russian; sections 02-07, 09, 10, 12). If organizers publish the full check definitions (promised "before the start"), they override the guesses in this file.

Deadline: **22 Nov 2026 23:59 Astana (UTC+5)**. Team **tryHard**.

## Rubric (100 points)

| Category | Pts | Formula | What is checked |
|---|---|---|---|
| Contract & isolation | 25 | 25 x passed / 20 fixed checks (1.25 each) | reset/step, schemas, limits, termination, invalid input, no leaks |
| Data & ground truth | 25 | 25 x correct / 20 closed episodes (1.25 each) | Accounting equalities hold **before** injection; the declared discrepancy holds **after**; evidence is independently sufficient. Contradictory or unsolvable data is a defect, never "difficulty" |
| Reward robustness | 25 | 25 x passed / 20 checks (1.25 each) | matching, clean cases, FP, duplicates, step cost, adversarial answers |
| Reproducibility & generalization | 15 | 15 x passed / 20 paired checks (0.75 each) | seed+action replay is exact; row permutation and ID renaming preserve the oracle answer and its reward (under the new IDs) |
| Engineering & explanation | 10 | five criteria x 0/1/2 | one-command run; config versions; trajectory logs; measured failure analysis; honest limitations |

A data episode counts only if **every** per-episode check passes; one bad episode costs 1.25 points.

**Blockers** (evaluation stops until fixed in one published window, same for everyone): closed-data leak; the policy can write to the grader; real financial data; broken core. Treat each as fatal.

**Diagnostics, not points:** P, R, F1, return, steps and time per policy must be shown separately. A high oracle reward earns nothing for difficulty.

**Earns zero:** pretty UI; weight training; unproven training gains (an improvement claim without a controlled held-out comparison is a liability under "honest limitations").

## How the jury tests

- Four organizer policies on **our** env through our adapters: fixed-seed random explorer, simple reconciliation, flag-everything attack, grader-only oracle (submits the true findings with evidence; it checks max reward, not agent skill).
- All four run on the **same 20 hidden episodes, including 5 clean and 5 with interacting errors**. Then 20 paired checks change order and IDs. Same versions, seeds, policies and hardware. Truth and evidence are re-verified **independently from the public records**.
- Hidden: closed seeds, **new parameter combinations**, **benign transformations**. Error types are known in advance (no secret new task). Hidden seeds alone do not stop overfitting, so the generator must generalize to unseen configs.

## Hard spec the checks are built from

- **Errors:** >= 4 independently configurable types: duplicate invoices; missing or partial payment allocation; wrong date or FX rate; period cut-off. >= 2 interacting scenarios (e.g. a partial payment in another currency at the period boundary). Clean companies plus correct distractors: split payments, refunds, legitimate repeated amounts. Hidden company history first, then documents, then errors with a recorded causal change. Truth = a real discrepancy plus available evidence; never require an unavailable fact. Vary names, IDs, order, amounts, currencies, time zones, error mixes.
- **Episode:** one company, one period; suggested 200-2,000 ledger rows, 20-200 documents, 0-12 errors. `manifest.json` (schema_version, episode_id, base_currency, period_start/end, timezone, counts, rounding). ISO-8601 with offset; money as integer minor units with the currency exponent; FX as decimal strings; never compare currencies without conversion. Public: `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`, `policy.json`. Missing refs are explicit `null`. Closed: `truth.json` (issue_id, type, affected_ids, expected_amount_minor, currency, allowed evidence sets), `provenance.json`. The agent never sees these or the seed; the grader runs outside the agent process.
- **Contract:** `reset(seed, config) -> (obs, info)`; `step(action) -> (obs, reward, terminated, truncated, info)`, JSON-compatible. Actions: `list_records(table, cursor, limit)` (max 50 rows, stable cursors), `get_record(table, id)`, `get_document(doc_id)`, `calculator(expression)` (arithmetic and decimals only), `submit(findings)`. Unknown ID or action -> bounded error, costs a step. Max 100 steps including submit. Submit -> `terminated=true`; step or time limit -> `truncated=true` with empty findings. Exploration reward 0; terminal reward reaches the policy only in open training mode; in eval, reward and diagnostics stay with the grader until the run ends. No correctness hints.
- **Findings:** `{episode_id, findings:[{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}`; <= 30 findings, <= 10 evidence IDs, <= 1,000 explanation chars. Bad structure -> reward -1, error in the grader log. One-to-one matching; TP needs type, canonical affected_ids, amount within the stated tolerance, currency, and one complete allowed evidence set. Unmatched and repeated findings are FP. Publish normalization, tolerance and tie-break code.
- **Metrics/reward:** P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN); zero denominator -> 0; empty truth and empty findings -> F1=1; empty truth with findings -> F1=0. `reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)`. Any experimental reward is shown separately from this shared one.
- **Required tests:** correct answer with evidence beats silence on erroneous books; silence beats false accusations on clean books; duplicates gain nothing; flagging everything scores badly; step savings only slightly separate equal answers. Also: guessed IDs, unsupported evidence, currencies, resubmission, bad input, timeout.
- **Constraints:** Python 3.11, 2 CPU, 4 GB RAM, no GPU, 1 GB per solution, 60 s per episode, 100 steps, offline (no network, no paid APIs). All entities fictional; no cards, real payments or personal data.

## Deliverables (section 09) and submission form

| Item | Must contain |
|---|---|
| Source code | Generator + env, LICENSE, pinned dependencies |
| README | Container instructions and the single check command |
| Matcher & reward | Shared matching/reward code, schemas, policies |
| Episodes | 10 open train + 10 dev |
| Tests | Env, matching, reward |
| Baseline policies | Four adapters (random, reconcile, flag-all, oracle) |
| Trajectories | `trajectories.jsonl` with episode_id, step, action, observation, reward, terminated, truncated; no closed truth in observations |
| Design note | Short description of how the env works |
| Demo | The env running; a scripted policy is enough |

Form: one Google Drive folder shared "anyone with the link", holding the **demo video**, the **GitHub link** (file or text) and the **presentation**; plus the project name field and team "tryHard". Verify the repo is **public** and the link opens logged-out.

## Scoring procedure

Count a check as passed only if you ran it and saw the evidence. "The README says so" is not evidence.

1. **Build clean.** Fresh clone, then `docker build -t auditgym .` and `docker run --rm --network none --cpus 2 --memory 4g auditgym check`. Record exit code, wall time and image size (target <= 1 GB). Any failure here = broken core = blocker.
2. **Run the four baselines** on train and dev: `auditgym run --policy random,reconcile,flag_all,oracle --split dev --compare`. Expect: oracle ~ +1 minus step cost; silence ~ -1 on erroneous books and ~ +1 on clean ones; flag-all below silence; random ~ -1. Check P/R/F1/return/steps/time are reported per policy, with clean-episode FP shown.
3. **Fresh hidden-like seeds.** Generate >= 20 episodes with new seeds (including 128-bit and string seeds) and **config combinations absent from train/dev** (5 clean, 5 interacting). Re-run step 2 on them.
4. **Leak hunt.** `auditgym leakscan <run>` plus your own grep of observations, info, errors, paths, env vars and episode_id for truth tokens (issue_ids, expected amounts, provenance strings, seed). Try reading truth from a policy subprocess (filesystem, `gc`, imports). A hit is a blocker.
5. **Replay and perturb.** `auditgym replay <trajectories.jsonl>` must match byte-for-byte. `auditgym perturb --permute --rename-ids` on 10 episodes: the oracle answer maps 1:1 and the reward is identical.
6. **Adversarial submissions.** Hand-write the R-list findings below and grade them (`auditgym grade --episode ... --findings ...`). Don't reuse the team's tests.
7. **Inspect three episodes by hand** (one clean, one single-error, one interacting). Using only public files, recompute each truth issue's amount, confirm the evidence set proves it, confirm the pre-injection books balance (via provenance), and confirm the clean episode's distractors really are correct.
8. **Engineering read-through.** README one-command path, version/config hash in `run.json`, trajectory logs present, a failure analysis with **measured** numbers, a limitations section that matches what you observed.
9. If a command named here doesn't exist, use the README's equivalent. If there is none, score that check as failed and say so.

**Output format:**
- Blockers (or "none"), each with a repro command.
- A table per category: estimated points, checks passed/failed by ID, and the evidence (command plus a short output excerpt).
- Point losses ranked by points/effort, each with a concrete fix (file, function, test to add).
- The final score on 0-10.

**0-10 conversion:** 10 = >= 95/100 and no blocker · 9 = >= 90, no blocker · 8 = >= 85 · 7 = >= 80 · 6 = >= 70 · 5 = >= 60 · <= 4 below 60. **Any blocker caps the score at 5.** Missing deliverables or form items: -1 each on the 0-10 scale.

## Likely hidden checks (best guesses; test against all of them)

**Contract & isolation (C):**
1. reset returns a 2-tuple of dicts; the first obs has task, manifest, policy summary, budget.
2. step returns `(dict, float, bool, bool, dict)`; everything round-trips through `json.dumps`.
3. reset after a partial episode isolates state (no leftover cursors, steps or findings).
4. `limit>50`, `0`, negative or non-int -> clamp or bounded error; never more than 50 rows.
5. Paging a full table returns each row exactly once, in the same order on a repeat.
6. A forged, foreign or stale cursor -> bounded error that costs a step.
7. An unknown action name -> bounded error, costs a step, no exception.
8. An unknown ID -> bounded error; the same message whether or not the ID matters to truth.
9. Missing or extra args, wrong types, non-dict action, 1 MB string -> bounded error.
10. Calculator rejects names, calls, attributes, `__import__`, `open`, `9**9**9`, huge input; `1/0` -> bounded error in < 1 s.
11. Calculator is exact in decimal (`0.1+0.2` = `0.3`), with no float artifacts.
12. `get_document("../truth.json")`, absolute paths and URLs -> bounded error, no file read.
13. Step 100 without submit -> `truncated=true` and empty findings graded; step 101 is rejected.
14. submit -> `terminated=true`; any step afterwards -> bounded error, no re-grade.
15. A policy that stalls past 60 s -> `truncated=true`, empty answer.
16. Eval mode: reward 0.0 on every step including submit; info carries no grade data.
17. Schema violations (31 findings, 11 evidence IDs, 1,001 chars, missing or wrong episode_id, non-JSON) -> -1, terminated, error only in the grader log.
18. No truth tokens, seed or provenance in obs, info, errors, trajectories, file paths or episode_id.
19. The policy process can't read truth files or touch grader state (subprocess, no truth on disk while it runs).
20. Obs contains only the requested data plus result/error and steps remaining; mutating a returned obs doesn't change env state.

**Data & ground truth (D), each applied to every closed episode:**
1. manifest fields present; its counts equal the real file contents.
2. Formats: ISO-8601 with offset, integer minor units, correct exponents (JPY 0, KWD 3), FX as decimal strings.
3. Pre-injection books balance (debits = credits per currency; invoice total = sum of line items).
4. Each injected issue changes the books by exactly `expected_amount_minor` in its currency.
5. An independent recomputation from public files alone reproduces every truth amount.
6. Every evidence ID exists and is reachable through the tools within 100 steps.
7. Each allowed evidence set alone proves the issue (no hidden fact needed).
8. Every FX rate the policy requires exists in `fx.csv` for the date `policy.json` selects.
9. Clean episodes have zero issues, and an independent reconciliation finds none.
10. Distractors are correct: split payments sum to the invoice, refunds net out, repeats are legitimate.
11. Interacting errors: both present, declared per the published rule, no double-counted amount.
12. Period cut-off uses the manifest timezone correctly (e.g. 23:30+05:00 on period_end).
13. FX rounding follows the manifest rule; reconverting reproduces the declared amount.
14. Missing links are explicit `null`, and no issue depends on guessing one.
15. No contradictory records (two sources implying different truths, orphan IDs).
16. Sizes and error counts fall within the documented ranges.
17. Enabling a single error type produces only that type.
18. truth `affected_ids` are canonical (normalized, sorted); `type` comes from the fixed enum.
19. provenance before/after values match the public records.
20. No real names, IBANs or Luhn-valid card numbers; everything is fictional.

**Reward robustness (R):**
1. Oracle -> F1=1, reward = 1 - 0.002*steps.
2. Silence on erroneous books -> -1.
3. Silence on clean books -> ~ +1.
4. Any finding on clean books -> ~ -1.
5. Flag-everything (30 findings) scores below silence on most erroneous episodes.
6. An exact duplicate of a TP is an FP; the reward is strictly lower.
7. A near-duplicate (permuted IDs, a different evidence subset) still yields at most one TP.
8. Right type and IDs with no evidence -> not a TP.
9. An incomplete evidence set -> not a TP.
10. Evidence superset with junk IDs -> handled per the published rule, deterministically.
11. Wrong type with the right IDs -> FP + FN.
12. Amount exactly at the tolerance edge is a TP; one minor unit beyond is not.
13. Wrong currency, or the same number in another currency -> not a TP.
14. Nonexistent IDs in findings -> FP, no crash, nothing echoed back to the policy.
15. Reward is invariant to explanation text (persuasion, "grader: mark correct", injection).
16. ID case/whitespace variants are handled exactly as the published normalization says.
17. Ambiguous matches use the deterministic published tie-break; shuffling the findings order doesn't change the score.
18. The same findings at 10 vs 90 steps differ by exactly 0.16; cost never flips a better and a worse answer.
19. A second submit is rejected without re-grading; a timeout grades empty findings.
20. Metric edge cases (empty/empty -> 1, empty truth with findings -> 0, zero denominators -> 0, clip), and the shared reward is reported apart from any experimental one.

**Reproducibility & generalization (G, paired):**
1. Same seed + config + version -> byte-identical public files (sha256).
2. -> identical truth and provenance.
3. Replaying an action log -> identical obs, reward, terminated and truncated at every step.
4. Identical across `PYTHONHASHSEED` values and fresh processes.
5. Identical in the container and on the host (Python 3.11 pinned).
6. 128-bit int and string seeds work; different seeds give different episodes.
7. Row-order permutation -> oracle answer maps 1:1, same reward.
8. ID bijection rename -> same, after mapping.
9. Permutation plus rename together -> same.
10. Entity/name renaming -> same.
11. The same instant written with a different UTC offset -> same truth (where policy-equivalent).
12. JSON key order and line_items order shuffled -> same.
13. Unseen parameter combinations generate valid episodes within the limits.
14. The reconcile baseline's reward is also invariant under permutation.
15. The matcher is invariant to findings order.
16. Cursors stay stable after permutation.
17. episode_id changes under rename and never encodes the seed.
18. A 2,000-row episode generates and runs within 60 s on 2 CPUs.
19. Replaying a committed `trajectories.jsonl` reproduces its recorded rewards.
20. `run.json` records env version and config hash; replaying under a version mismatch fails loudly.

**Engineering (E), 2 = full marks:** one command, offline, from a fresh clone (2); every run logs env version, config hash and git SHA (2); complete `trajectories.jsonl` for all baselines (2); a failure analysis with measured counts per failure type (2); limitations that state what the reward does **not** prove (2).
