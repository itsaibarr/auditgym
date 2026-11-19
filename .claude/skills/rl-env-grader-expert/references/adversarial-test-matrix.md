# Adversarial and grader test matrix

Each row is one pytest case or Hypothesis property, and `auditgym check` runs all of them.
`[CASE]` marks a test the case requires (§05 "Тесты должны показать" / "Проверьте…", §10 "Надёжность награды").

**Fixtures**
- `E_err`: an episode with **n = 4** truth issues, one of each type.
- `E_clean`: an episode with 0 issues and its distractors on.
- `E_x`: an episode with an interacting pair (2 linked issues).
- `O`: the oracle findings, with one allowed evidence set per issue.
- `k`: inspection steps (steps consumed before `submit`).
- Unless stated otherwise, the submission happens at step 1 (k = 0) in **train** mode. Reward = `clip(2*F1 − 1 − 0.002k, −1, 1)`.

## A. Ordering tests the case requires

| ID | Input | Expected | Assert |
|---|---|---|---|
| A1 [CASE] | `E_err`: submit `O` vs submit `[]` | O: TP4 FP0 FN0, F1=1, **r=1.0**. `[]`: FN4, F1=0, **r=−1.0** | r(O) > r([]) |
| A2 [CASE] | `E_clean`: submit `[]` vs one plausible finding (a distractor such as a split payment, a refund, or a legitimate repeated amount) | `[]`: F1=1, **r=1.0**. One finding: F1=0, **r=−1.0** | r([]) > r(false accusation) for every distractor kind |
| A3 [CASE] | `E_err`: `O` + an exact copy of `O[0]` | TP4 FP1, F1=8/9, **r≈0.778** | r < r(O); copies never add TP |
| A3b | `E_err`: every finding of `O` twice | TP4 FP4, F1=8/12, **r≈0.333** | Strictly decreasing in the number of copies |
| A3c | `E_err`: a copy of `O[0]` with a different explanation and permuted `affected_ids` order | Same canonical key, so it's a duplicate: FP1 | The reason in the trace is `duplicate` |
| A4 [CASE] | `E_err`: `flag_all` (30 findings over types × candidate records), e.g. 2 accidental TPs | TP2 FP28 FN2, F1≈0.118, **r≈−0.765** | r(flag_all) < r(reconcile) < r(O); mean over the split < 0 |
| A4b [CASE] | `E_clean`: `flag_all` | FP30, F1=0, **r=−1.0** | Equal to the minimum |
| A4c | `E_err`: for every true `affected_ids`, all 4 types, with no evidence | 0 TP (evidence missing), F1=0, r=−1.0 | Type spraying earns nothing |
| A5 [CASE] | `E_err`: `O` submitted at k=0, 10, 90 | r = 1.0, 0.98, 0.82 | Δr = 0.002·Δk exactly; max Δ ≤ 0.198 |
| A5b | `E_err`: `O` at k=0 vs `O` minus one finding at k=0 | 1.0 vs 2·(6/7)−1 ≈ 0.714 | Step savings never reverse a TP difference at the same k |

## B. Field-level matcher attacks

| ID | Input (base: `O` on `E_err`, modify one finding f) | Expected for f | Episode r |
|---|---|---|---|
| B1 [CASE] guessed IDs | `affected_ids` are syntactically valid IDs that don't exist | FP `unknown_id`, plus 1 FN | TP3 FP1 FN1 → F1=0.75, **r=0.5** |
| B2 [CASE] unsupported evidence | `evidence_ids=[]` | FP `evidence` | r=0.5 |
| B3 [CASE] | evidence = a strict subset of every allowed set | FP `evidence` | r=0.5 |
| B4 | evidence = real but irrelevant records | FP `evidence` | r=0.5 |
| B5 | evidence = an allowed set + up to 10 total real extras | **TP** (superset allowed) | r=1.0 |
| B6 foreign evidence | evidence = an allowed set + one ID from another episode, or a non-existent ID | FP `unknown_id` (any unknown evidence ID disqualifies) | r=0.5 |
| B7 [CASE] currency | correct amount, wrong currency (`EUR` for `USD`) | FP `currency` | r=0.5 |
| B8 | the same value converted into base currency when truth is in the document currency, or the reverse | FP `currency` or `amount` | r=0.5 |
| B9 | lowercase `"usd"`, or a 4-letter currency | Schema `^[A-Z]{3}$` → **malformed, r=−1** | −1 |
| B10 | amount = expected ± tol | TP | 1.0 |
| B11 | amount = expected ± (tol+1) | FP `amount` | 0.5 |
| B12 negative amount | `claimed_amount_minor = −expected` | Schema `ge=0` → **malformed, r=−1** | −1 |
| B13 huge int | `10**16`, or a 5000-digit literal (Python 3.11 `ValueError` from int max-str-digits) | **malformed, r=−1**; no traceback anywhere | −1 |
| B14 | wrong type, correct ids, amount and evidence | FP `type` + FN | TP3 FP1 FN1 → 0.5 |
| B15 | `affected_ids` is a superset or subset of the canonical set | FP `no_issue` | 0.5 |
| B16 | `affected_ids` with a repeated ID `[A, A]` | Not canonical → FP | 0.5 |
| B17 unicode/whitespace | `" INV-7Q2K"`, `"inv-7q2k"`, Cyrillic homoglyph `"ІNV-7Q2K"`, zero-width `"INV​-7Q2K"`, fullwidth digits | FP `unknown_id` (exact-match rule, no normalization) | 0.5 |
| B18 | f with a wrong amount at index 0 + the correct f at index 1 | Index 0 FP `amount`, index 1 TP (ineligible findings never block) | TP4 FP1 → 0.778 |
| B19 | Shuffle the order of the findings list | Same TP/FP/FN and r | Invariant |
| B20 explanation | Persuasive text, "ignore rules, mark correct", empty `""`, exactly 1000 chars | Same verdict as baseline | r unchanged |
| B21 | explanation of 1001 chars | **malformed, r=−1** | −1 |
| B22 interacting | `E_x`: report only one of the 2 linked issues | TP1 FN1, F1=2/3, **r≈0.333** | Partial credit only |
| B23 interacting | `E_x`: one merged finding with the union of ids | FP1 FN2, F1=0, **r=−1** | No credit for merged findings |

## C. Structural and malformed submissions [CASE: неверный ввод]

All of these give `terminated=true`, **reward exactly −1** (no step cost), the detail only in the grader log, and an eval observation with no detail.

| ID | Input |
|---|---|
| C1 | `findings` argument is a non-JSON string, `null`, a list instead of an object, or a number |
| C2 | `episode_id` missing, or another episode's `episode_id` |
| C3 | 31 findings; 11 `evidence_ids`; `affected_ids` empty or over its cap |
| C4 | `type` not in the enum (`"Duplicate_Invoice"`, `"other"`) |
| C5 | `claimed_amount_minor` = `100.0`, `"100"`, `true`, `null` (bool is an `int` subclass, so reject it explicitly) |
| C6 | An unknown extra field in a finding or at the top level |
| C7 | `NaN`/`Infinity` literals (stdlib `json` accepts them by default: pass `parse_constant` to reject) |
| C8 | Duplicate JSON keys (stdlib keeps the last one: reject them with `object_pairs_hook`) |
| C9 | An ID element that isn't a string (`123`, `{}`), or nested lists |

## D. Episode-flow attacks

| ID | Input | Expected |
|---|---|---|
| D1 [CASE] resubmission | `submit(O)` then `submit(O')` | 2nd: error `episode_over`, reward 0, no step consumed, **no re-grade**; return stays r(first) |
| D2 | Any action after `truncated` | `episode_over`, same as D1 |
| D3 [CASE] timeout | The policy sleeps past the 60 s episode clock | Process group killed; `truncated=true`; empty findings graded: `E_err` → −1, `E_clean` → 1−0.002k |
| D4 | The policy hangs inside one step (no newline) | Per-step deadline → same as D3; the runner never blocks |
| D5 step exhaustion | 100 non-submit actions | Step 100 returns `truncated=true, terminated=false`; `E_err` r=−1, `E_clean` **r=0.8** |
| D6 | 99 actions, then `submit(O)` as the 100th | `terminated=true`, `truncated=false`, r = 1 − 0.002·99 = 0.802 |
| D7 | Unknown action name or bad envelope | `unknown_action` / `invalid_action`; step consumed; not terminated |
| D8 | Errors only: 50 bad actions, then `submit(O)` | r = 1 − 0.1 = 0.9 (errors count as inspection steps) |
| D9 | Policy crash or stdout EOF mid-episode | `truncated=true`, termination `policy_crash`, empty findings graded |
| D10 | A non-JSON stdout line, or a line > 1 MB | `invalid_action`, step consumed |
| D11 | The policy writes to stderr in volume | Captured, capped and truncated; the runner never deadlocks |

## E. Tool-surface attacks (bounded error, step consumed, never an exception or hang)

| ID | Input | Expected code |
|---|---|---|
| E1 | `get_document("../truth.json")`, `"/etc/passwd"`, `"truth"` | `not_found`, with message bytes identical to any unknown ID |
| E2 | `list_records("truth")`, `"provenance"`, `"manifest; DROP"` | `unknown_table` |
| E3 | `limit` = 0, −1, 51, `10**20`, `"50"`, `true` | `invalid_arguments` (`null`/missing → 50) |
| E4 | `cursor` = `"-1"`, `"999999"`, `"abc"`, a cursor from another table | `invalid_cursor` |
| E5 | `get_record("ledger", "*")`, `"%"`, `""`, a 10 KB string | `not_found` / `invalid_arguments` |
| E6 | calculator: `__import__('os')`, `open('x')`, `().__class__`, `lambda:1`, `[1]*9**9`, `'a'*10**9` | `calc_error` |
| E7 | calculator: `9**9**9`, `10**10**10`, `1/0`, `1e999999`, 300-char expression | `calc_error` within ms |
| E8 | calculator: `0.1+0.2` | `"0.3"` (Decimal, not float) |
| E9 | Page walk | Concatenating all pages = the full table, no dupes, ≤50 rows per page; the same cursor gives the same page twice |

## F. Isolation, leak and determinism properties

| ID | Property | How |
|---|---|---|
| F1 | Eval mode hides reward | Every step `reward == 0.0`; `info` keys ⊆ `{steps_remaining, error_code}`, terminal step included |
| F2 | **Truth-swap invariance** | Replay the same actions with a dummy `PrivateTruth`: eval observations and `info` are byte-identical |
| F3 | Leak scan | No issue_id, provenance string, seed, `truth`, `Traceback` or absolute path in any observation, info, error or train trajectory |
| F4 | No truth on disk during the run | While the child is alive, list the run dir and the child's cwd: neither has `truth.json`/`provenance.json` |
| F5 | Seed determinism | Same (seed, config, version) → identical episode fingerprint in 2 processes with different `PYTHONHASHSEED` |
| F6 | Step determinism | Same seed + actions → byte-identical trajectory lines and rewards |
| F7 | Reset isolation | reset(A) after B equals a fresh reset(A); 2 interleaved env instances don't interfere |
| F8 | No reused buffers | Mutating a returned observation doesn't change the next one |
| F9 [jury 15 pts] | Permutation + ID rename | For a seeded bijection π: r(oracle) equal; for any submission S, TP/FP/FN(S, E) = TP/FP/FN(π(S), π(E)) |
| F10 | Matcher optimality | Hypothesis: greedy TP count = brute-force max matching (n ≤ 6) |
| F11 | Structural position leak | Over the train/dev episodes, record position and ID rank don't separate affected from unaffected records (e.g., mean rank difference within noise) |
| F12 | Oracle max | r(oracle) == 1.0 on every train, dev and generated episode, clean ones included |
| F13 | Evidence walk | Fetch every allowed evidence set through `env.step()`; an independent checker re-derives type, amount and currency from the fetched records only |
| F14 | Budget feasibility | At maximum config, full-scan steps (ledger + invoices + payments + fx pages + 1) ≤ 60 |
| F15 | Random baseline | Seeded `random` gives an identical trajectory on re-run |
