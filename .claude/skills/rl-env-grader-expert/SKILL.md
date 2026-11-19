---
name: rl-env-grader-expert
description: RL environment engineering and reward/grader robustness for AuditGym, the Genesis Hackathon FinTech 2026 RL environment for AI financial auditors. Use when building or reviewing the env contract (reset/step, terminated/truncated), action or observation schemas, the policy runner and child-process isolation, the matcher, reward or F1 edge cases, trajectories.jsonl/results.jsonl, determinism and reproducibility, paired permutation/ID-rename checks, truth-leak prevention, the four baselines (random, reconcile, flag-all, oracle), adversarial grader tests, or the `auditgym check` command. Also use when acting as the "RL/ML engineer" council reviewer scoring the project 0-10.
---

# RL env + grader expert (AuditGym)

The jury scores the **environment and grader**, not our agent (case p.5). 75 of 100 points come from fixed checks: contract and isolation (25), data and reference (25), reward robustness (25), and paired repro checks (15). Every rule below comes from the case (`research/case.pdf` §04–§07, §10) or closes a known reward-hacking vector. Wire formats are in [references/contract-spec.md](references/contract-spec.md). The test table is in [references/adversarial-test-matrix.md](references/adversarial-test-matrix.md).

## 1. The contract (case §04/§05)

- `reset(seed, config) -> (observation, info)`. The first observation holds the task, the manifest, a policy summary, the remaining budget, the action list and table descriptors. It must not hold seeds, error counts or config error knobs.
- `step(action) -> (observation, reward, terminated, truncated, info)`. Everything is JSON-compatible dicts, built fresh on every call (Gymnasium: return "new observation and info objects rather than modifying and returning the same buffer"). Inheriting `gymnasium.Env` is optional.
- There are exactly five actions: `list_records(table, cursor, limit)`, `get_record(table, id)`, `get_document(doc_id)`, `calculator(expression)` and `submit(findings)`. There's no shell, SQL, filesystem or network.
- **Pages:** at most 50 rows. Treat `limit` as optional (default 50). Any value outside 1..50 is a bounded `invalid_arguments` error. Make cursors opaque but offset-based (row position in the table's fixed published order), so they're stable for the whole episode and invariant under ID renaming. `next_cursor` is null at the end. A forged or out-of-range cursor gives `invalid_cursor`.
- **Calculator:** an `ast` walker over `Decimal` that whitelists `Expression`, `BinOp(+ - * /)`, `UnaryOp(+ -)` and numeric `Constant`. Take literal text through `get_source_segment` so values never become floats. Cap length at 256 and use a local decimal context. Never call `eval` or `simpleeval`/`asteval`, which have had 2025–26 sandbox-escape CVEs (05 §4).
- **Errors:** an unknown action, table, ID, bad arguments or a bad expression returns a bounded error `{code, message}`, **consumes a step**, and does not end the episode. Messages are fixed templates that only echo the policy's own input.
- **Step budget:** 100 steps **including** `submit`. Count every action, including failed ones. `submit` gives `terminated=true` (it's valid as the 100th action). If the 100th action isn't `submit`, return `truncated=true` on that step. Exceeding 60 s of episode wall clock also gives `truncated=true`. Truncation grades an **empty** findings list.
- **After the end:** any further `step` returns error `episode_over`, reward 0, the same flags, no step consumed and **no re-grade**. That covers resubmission.
- **Intermediate reward is exactly 0.0.** In **train** mode, only the terminal step returns the real reward. In **eval** mode, every step including the terminal one returns `0.0`, `info` is limited to `{steps_remaining, error_code}`, and the reward plus diagnostics are released only after the run.
- **Findings** are `{episode_id, findings:[{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}`. The limits are ≤30 findings, ≤10 evidence IDs per finding, and ≤1000 characters of explanation. Any structural violation (bad JSON, missing or extra fields, wrong `episode_id`, a type outside the enum, bool/float/string amounts, NaN, duplicate keys, ints with more than 4300 digits, over-limit lists) terminates the episode with reward **exactly −1**, no step cost, and the detail recorded only in the grader log.
- **Matching** is one-to-one with truth. A TP needs the right type, canonical `affected_ids` (an exact set match), an amount within the **published** tolerance, the exact currency, and `evidence_ids ⊇` at least one allowed evidence set. Unmatched findings and repeats are FP. Unmatched truth issues are FN.
- **Metrics:** `P=TP/(TP+FP)` and `R=TP/(TP+FN)`, with 0 when the denominator is 0. `F1 = 2TP/(2TP+FP+FN)`, except that empty truth with empty findings gives F1=1, and empty truth with any finding gives F1=0. For both-empty, report P=R=0 and F1=1, as the case states.
- **Reward:** `clip(2*F1 − 1 − 0.002*inspection_steps, −1, 1)`, and the episode return equals it. **`inspection_steps` = every step consumed before `submit`**, errors and calculator calls included. It excludes the submit itself. On truncation it equals all steps used. Publish this definition. If errors were free, invalid probes would become free information. An experimental reward may exist, but log it as `reward_experimental` and never let it replace or alter `reward`.

Reward facts to reason with:
- An oracle that submits at step 1 gets exactly **1.0**. Silence on books with errors gets **−1**, and silence on clean books gets **1.0**. Any finding on clean books gives **−1**.
- The maximum step cost is 0.198 when submitting, or 0.2 on truncation. Truncation on clean books therefore gives **0.8**. That's mandated by the case: report it honestly and don't "fix" the shared reward.
- The region at F1=0 is flat at −1 (clipped), so zero-F1 policies get no gradient from step savings.
- Step cost can outweigh one TP. With 12 issues, missing one costs ≈0.087 in reward, while 44 saved steps gain 0.088. List this under limitations.
- Reward depends only on findings and step count, never on the text. `explanation` must be invariant below the cap.

## 2. Isolation architecture

- The **runner owns the truth**. `generate(seed, config) -> (PublicEpisode, PrivateTruth)` is pure. The env is built from `PublicEpisode` **only**, so its code can't read truth. `grade(findings, truth, inspection_steps)` is pure and runs in the runner after `submit` or truncation.
- In eval mode the **policy always runs in a child process**. Spawn it with `Popen([sys.executable, "-I", "-m", "auditgym.policy_host", spec], env={"PATH":…, "PYTHONHASHSEED":"0"}, cwd=<empty tmpdir>, start_new_session=True)` and talk over JSON lines on stdin/stdout. `policy_host` calls `resource.setrlimit` (AS, CPU, NOFILE, FSIZE=0) before it imports the policy. Don't use `preexec_fn` or in-process eval policies, because an in-process policy can reach truth through `gc.get_objects()` or monkey-patch the grader (METR 2025).
- **Timeouts:** enforce a per-step read deadline (`selectors`) and a 60 s episode clock. On expiry, `os.killpg(SIGKILL)`. Note that `communicate(timeout)` does not kill the child. Then grade empty findings with `truncated=true`. A policy crash or EOF does the same, with termination `policy_crash`. A non-JSON or oversized stdout line is an `invalid_action` and consumes a step.
- **No truth on disk while the policy is alive.** Write `truth.json`, `provenance.json`, the grader log and `results.jsonl` only after the child exits. The Docker image contains no closed seeds. The seed never appears in `episode_id`: use `hmac(secret, seed)[:16]`.
- The **oracle is privileged**. It's built into the runner, never reachable through `cmd:`, and still acts only through `env.step()`, so its trajectory goes through the same leak scan.
- The policy cannot write to grader state. The reward is computed from runner state plus the submitted `findings` and nothing else.

## 3. Determinism (15 pts plus every reward check)

- Use one `random.Random(sha256(f"{seed}|{stage}").digest())` per generator stage. Never use global `random`, `hash()`, `set` iteration, `dict`-from-`set`, `os.listdir` order, `datetime.now()`, or locale-dependent formatting. Only `.random()` is guaranteed stable across Python versions, so build helpers on it and pin Python 3.11.
- Money is `int` minor units. FX is `Decimal` using string rates inside `localcontext(prec=50, traps=…)`, with a published rounding mode. Never put a float in emitted data (assert this).
- Serialize with canonical JSON everywhere (trajectories, fingerprints, observations): `json.dumps(o, sort_keys=True, separators=(",",":"), ensure_ascii=False)`. Fingerprint an episode as the sha256 of its canonical public files.
- Use the pinned `tzdata` wheel with `PYTHONTZPATH=""`, so period cutoffs don't depend on host tz files. Also set `PYTHONHASHSEED=0` in Docker, but prove you don't rely on it: regenerate in a subprocess with `PYTHONHASHSEED=random` and compare fingerprints.
- **IDs:** assign final IDs **after** error injection, in a canonical order (timestamp, then a seeded tie-break). Then sort or shuffle rows so injected rows aren't appended. The paired check applies a **seeded bijection** to every opaque ID field consistently: records, refs, `invoice_refs`, line items, truth `affected_ids` and evidence sets. Content-derived keys such as `fx_id` stay fixed.
- `reset` isolates state. reset(A) after an episode on B must equal a fresh reset(A). Two env instances must share no mutable module state.

## 4. Leak vectors: check all of them

| Vector | Leak example | Rule |
|---|---|---|
| IDs | `INV-0042-DUP`, numbering gaps, injected IDs in a separate range | Assign IDs after injection with a uniform format, so prefixes encode the table only |
| Row order | Injected rows last; a duplicate adjacent to its original | Canonical post-injection order; test that position doesn't predict truth |
| Record shape | An injected row is missing a key, has a different key order, a `"100"` string, a different TZ offset format, or a memo like "dup" | One serializer for every record; schema-validate all rows |
| Error text | "already flagged", "not a known issue", different text for `../truth.json` | Fixed codes; `not_found` is identical for every unknown ID or path |
| `info` / observation | Grade, TP count, match trace or error count in eval mode | Eval `info` = `{steps_remaining, error_code}` only |
| Manifest / config | `n_errors`, error knobs, seed, split name that encodes the seed | The manifest holds only the case §03 fields and record counts |
| Sizes | Record counts correlate with `n_errors` | Draw sizes independently; clean episodes keep every distractor generator on |
| Paths / disk | Truth file readable during the run, tracebacks with paths, `/app/data/closed` | No truth on disk during the run; catch every exception into a code |
| Timing | Lookups for affected rows are slower; grading runs on submit in-band | O(1) lookups by key; grade after the episode in eval mode |
| Trajectories | Train trajectories carry truth in observations | `leakscan` checks issue_ids, provenance strings, seed, `truth`, `Traceback` and absolute paths |

The decisive test is **truth-swap invariance**: replay the same action sequence with `PrivateTruth` replaced by a dummy. Eval-mode observations and `info` must be byte-identical.

## 5. Matcher design (publish the code, case §05)

- **Eligibility** of finding f for issue i: `f.type == i.type`, `sorted(f.affected_ids) == sorted(i.affected_ids)` (exact strings, no case, whitespace or Unicode normalization, so homoglyphs and zero-width characters never match), `|f.amount − i.expected| ≤ tol(i)` (`policy.amount_tolerance_minor`), `f.currency == i.currency`, every evidence ID exists in the episode, and some allowed set is a subset of `f.evidence_ids`.
- **Assignment:** the generator asserts that `(type, sorted(affected_ids))` is **unique across truth issues** (merge any collisions). Each finding is then eligible for at most one issue, the graph is a union of stars, and **deterministic greedy is exactly maximal**. Iterate issues by `issue_id`, and for each one take the eligible finding with the lowest submission index. Every other finding is FP with reason `duplicate | no_issue | type | amount | currency | evidence | unknown_id`. Do **not** use `scipy.linear_sum_assignment`, whose tie choice depends on input order (05 §6).
- **Proof tests:** a Hypothesis comparison against brute-force max matching (n ≤ 6), invariance of TP/FP/FN under shuffled findings, and invariance under record permutation plus ID bijection for any submission mapped through the bijection.
- Size every injected error at ≥100× tolerance, so rounding noise can't produce or hide a TP.
- Evidence sets list only primary keys that `get_record` or `get_document` can fetch. Never require a fact that isn't public (case §02). An `evidence-walk` check fetches each set through the env and re-derives the discrepancy from the fetched records alone.

## 6. Baselines and the jury run (case §06)

The jury runs the same four policies on 20 closed episodes (5 clean, 5 with interacting errors), plus 20 paired permutation and ID-rename checks. Versions, seeds, policies and hardware are identical across teams.

| Policy | Implementation | Must hold |
|---|---|---|
| `random` | Seeded `random.Random`, valid random actions, almost never submits | Deterministic per seed; ≈ −1 on books with errors, 0.8 on clean books (truncated) |
| `reconcile` | Full scan (ledger 40 + invoices 4 + payments 4 + fx 2 pages ≈ 51 steps at maximum size) plus rule checks | Fits the budget at maximum config; between flag-all and oracle; its clean-FP rate is reported |
| `flag_all` | 30 findings spread over types and candidate records | Mean return < reconcile and < 0; exactly −1 on clean books |
| `oracle` | Runner-internal; reads truth; submits at step 1 with one allowed evidence set per issue | Reward **== 1.0 on every episode** (clean → `[]`); unchanged under permutation and rename |

The oracle submits without browsing, so its step count (and reward) can't depend on row order. Browse-then-submit oracles that scan pages break the paired checks.

**Metrics to report** per policy × split: micro and macro P/R/F1, mean return, mean steps, wall time per episode, per-type TP/FP/FN, **clean-FP rate** (the share of clean episodes with at least one finding), a termination breakdown (`submitted | max_steps | timeout | schema_error | policy_crash`), and failure categories from the match trace. Never claim model improvement without a controlled comparison on held-out data (case §01).

## 7. `auditgym check`

`check` is one offline command (`docker run --rm --network none auditgym check`) with no GPU or paid API. Exit codes are 0 for pass, 1 for any check failing, and 2 for a usage or internal error. It prints one `PASS|FAIL id seconds detail` line per check to stdout. `--json` emits a single document, and human logs go to stderr. It always writes `runs/check/summary.{json,md}`. Hypothesis runs with `derandomize=True, database=None`, and pytest-timeout keeps the suite from hanging. The suite runs the contract checks, Gymnasium-style seed and step determinism, leakscan plus truth-swap, every row of the adversarial matrix, the four baselines with **asserted orderings** (not just printed numbers), the evidence walk, and N paired permutation and rename checks.

## 8. REVIEW CHECKLIST (council "RL/ML engineer" role)

Score from 10. Take the **lowest cap** triggered, then subtract the minor deductions. A ≥9 requires zero BLOCKER, CRITICAL and MAJOR findings. Cite file:line and give a reproducing command for each finding.

**BLOCKER (cap 3)**, because the jury blocks grading for these:
- [ ] Any truth leak: observation, `info`, error text, paths, trajectories, truth on disk during the run, seed derivable from `episode_id`, or failing truth-swap invariance
- [ ] The policy can write grader state, or eval policies run in-process
- [ ] `check` isn't a single offline command, or it needs network, a GPU or a paid API
- [ ] The core is broken: reset/step crashes, or `submit` doesn't grade

**CRITICAL (cap 5):**
- [ ] Nondeterminism: same seed, config, version and actions give different records, transitions or rewards (including across `PYTHONHASHSEED`)
- [ ] Oracle < 1.0 on any episode, or permutation/ID rename changes the oracle's reward or the TP/FP/FN of a mapped submission
- [ ] flag-all, duplicates, extra findings or explanation text can raise reward; silence on clean ≠ 1.0; a false finding on clean ≠ −1
- [ ] Malformed findings ≠ exactly −1, or a malformed submit crashes the runner
- [ ] Matcher isn't one-to-one, or its tie-break depends on input order

**MAJOR (cap 7):**
- [ ] Pages > 50, unstable cursors, errors that don't consume a step, wrong 100-step or 60 s semantics, wrong `terminated`/`truncated` flags, or truncation that doesn't grade empty findings
- [ ] Nonzero intermediate reward, or reward/diagnostics visible during an eval run
- [ ] `inspection_steps`, tolerance, ID normalization or tie-break unpublished, or code that disagrees with the docs
- [ ] Calculator can execute code, hang or overflow; no per-step or episode timeout; no process-group kill
- [ ] `trajectories.jsonl` is missing the required keys (`episode_id, step, action, observation, reward, terminated, truncated`)
- [ ] An evidence set requires a fact that isn't public, or the evidence walk fails
- [ ] No adversarial tests for any case-required item (see the matrix)

**MINOR (−0.5 each):** per-type metrics missing; clean-FP rate missing; no match trace or failure analysis; JSON Schemas not exported; the experimental reward mixed into the shared column; no honest-limitations section (the truncation 0.8 and step-cost vs TP trade-off); unpinned dependencies; `check` exits with something other than 0/1/2.

## Sources

- Case: `research/case.pdf` §03–§07 and §10. Design notes: `research/01-oss-rl-env-references.md` §3 and §6, `research/05-component-libraries.md` §4–§10, `research/04-cli-framework-eval.md` §9.
- Gymnasium Env API (terminated vs truncated, seeding, no reused buffers): https://gymnasium.farama.org/api/env/
- DeepMind, specification gaming ("satisfies the literal specification of an objective without achieving the intended outcome"): https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ ; examples list: http://tinyurl.com/specification-gaming
- Skalse et al. 2022, *Defining and Characterizing Reward Hacking*: https://arxiv.org/abs/2209.13085
- Agentic Benchmark Checklist (oracle solver, do-nothing agents, list-all answers, ground-truth isolation): https://arxiv.org/abs/2507.02825
- SWE-bench #465 (side-channel leak through git history): https://github.com/SWE-bench/SWE-bench/issues/465 ; METR on grader tampering: https://metr.org/blog/2025-06-05-recent-reward-hacking
- Python `random` reproducibility notes: https://docs.python.org/3/library/random.html#notes-on-reproducibility
