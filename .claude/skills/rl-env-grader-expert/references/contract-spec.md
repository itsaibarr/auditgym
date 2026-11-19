# AuditGym wire formats

These are the formats the skill recommends. When the repo's `schema/*.json` exists, it wins, so update this file to match it rather than the other way round. All JSON is canonical when written to disk or hashed: `json.dumps(o, sort_keys=True, separators=(",",":"), ensure_ascii=False)`. Money is an `int` in minor units, FX rates are decimal **strings**, timestamps are ISO-8601 with an offset, and missing references are an explicit `null`. No floats appear anywhere.

## 1. Python API

```python
obs, info = env.reset(seed: int | str, config: dict | None = None)
obs, reward, terminated, truncated, info = env.step(action: dict)
```
- `config` (runner side only, never echoed to the policy): `{"mode": "train"|"eval", "split": ..., "n_ledger": ..., "errors": {"duplicate_invoice": {"on": true, "count": 1}, ...}, "interacting": [...], "distractors": {...}}`.
- `env` is constructed from `PublicEpisode` only. `grade(findings, truth, inspection_steps) -> GradeResult` lives in the runner.

## 2. Action: tool-call envelope

```json
{"name": "list_records", "arguments": {"table": "ledger", "cursor": null, "limit": 50}}
```

| name | arguments | result (in `observation.result`) |
|---|---|---|
| `list_records` | `table` ∈ `ledger, invoices, payments, fx`; `cursor`: string or null; `limit`: int 1..50 or null (→ 50) | `{"table","rows":[...≤50],"next_cursor": str or null,"total": int}` |
| `get_record` | `table`, `id` (that table's primary key) | `{"table","record":{...}}` |
| `get_document` | `doc_id` (an invoice or payment document) | `{"doc_id","kind":"invoice"\|"payment","document":{...}}` |
| `calculator` | `expression`: str ≤256 | `{"value":"123.45"}` (Decimal string) |
| `submit` | `findings`: the object in §5 | train: `{"submitted": true}`; eval: `{"submitted": true}` (identical whether valid or malformed) |

- **Primary keys** are unique episode-wide with table-specific prefixes: `ledger.row_id` (`L-…`), `invoices.doc_id` / `payments.doc_id` (`D-…`), and `fx.fx_id` = `FX-<date>-<SRC>-<TGT>`. That last one is content-derived, so the ID bijection doesn't change it. `invoice_id`/`payment_id` are business keys: a duplicate invoice may repeat an `invoice_id`, which is why fetching is by `doc_id`.
- `affected_ids` and `evidence_ids` use primary keys. The per-type canonical choice is published in the README and `schema/findings.schema.json`.
- An envelope with unknown keys, a missing `name`, or `arguments` that isn't an object gives `invalid_action`. Unknown or extra arguments give `invalid_arguments`. Both consume a step.

## 3. Observation

Reset observation:
```json
{
  "task": "Audit the period; submit findings with evidence.",
  "manifest": {"schema_version":"1.0","episode_id":"a3f9…16hex","base_currency":"KZT",
               "period_start":"2026-10-01T00:00:00+05:00","period_end":"2026-11-01T00:00:00+05:00",
               "timezone":"Asia/Almaty","counts":{"ledger":412,"invoices":61,"payments":58,"fx":93},
               "rounding":"ROUND_HALF_EVEN"},
  "policy": {"fx_rate_date":{...},"amount_tolerance_minor":{"same_currency":0,"fx_converted":1},"...":"..."},
  "tables": {"ledger":{"primary_key":"row_id","columns":["row_id","timestamp","account","debit_minor","credit_minor","currency","entity_id","invoice_id","payment_id"]}, "...": {}},
  "available_actions": ["list_records","get_record","get_document","calculator","submit"],
  "finding_types": ["duplicate_invoice","unallocated_payment","fx_conversion","period_cutoff"],
  "limits": {"max_steps":100,"page_size":50,"max_findings":30,"max_evidence":10,"max_explanation":1000},
  "steps_remaining": 100
}
```
The type names are placeholders: the repo's enum is authoritative. The manifest never contains the seed, error counts or config knobs.

Step observation (only the requested data, the result or error, and the remaining budget):
```json
{"result": {...} , "error": null, "steps_remaining": 87}
{"result": null, "error": {"code":"not_found","message":"no record with id 'X' in table 'ledger'"}, "steps_remaining": 86}
```
Error codes (closed enum): `unknown_action`, `invalid_action`, `invalid_arguments`, `unknown_table`, `not_found`, `invalid_cursor`, `calc_error`, `episode_over`, `invalid_findings` (train only). Messages are fixed templates that echo only the policy's input, truncated to 64 chars.

## 4. Reward / flags / info

| Situation | reward (train) | reward (eval, to policy) | terminated | truncated |
|---|---|---|---|---|
| Non-terminal step | 0.0 | 0.0 | false | false |
| Valid `submit` | `clip(2F1−1−0.002k,−1,1)` | 0.0 | true | false |
| Malformed `submit` | −1.0 | 0.0 | true | false |
| 100th step not `submit` / 60 s / crash | grade(`[]`) with k = steps used | 0.0 | false | true |
| Step after the end | 0.0 (`episode_over`) | 0.0 | unchanged | unchanged |

`info` in train mode: `{"steps_remaining":n,"error_code":str|null}`. On the terminal step it adds `{"grade":{"tp","fp","fn","precision","recall","f1","inspection_steps","reward","reward_experimental"?,"match_trace":[...]}}`.
`info` in eval mode: `{"steps_remaining":n,"error_code":str|null}` only, on every step.

## 5. findings.json (the `submit` argument)

```json
{"episode_id":"a3f9…","findings":[
  {"type":"duplicate_invoice","affected_ids":["D-7Q2K"],"claimed_amount_minor":1250000,
   "currency":"USD","evidence_ids":["D-7Q2K","D-M3XA","L-0F9C"],"explanation":"Same lines and total as D-M3XA, posted twice."}
]}
```
Schema (strict, `forbid_unknown_fields`): `episode_id` == the manifest value; `findings` ≤30; `type` ∈ enum; `affected_ids` 1..10 strings; `claimed_amount_minor` an int in 0..10^15 (bool rejected); `currency` `^[A-Z]{3}$`; `evidence_ids` 0..10 strings; `explanation` a string ≤1000. Reject NaN/Infinity and duplicate keys. Any violation means the submission is malformed, which gives −1.

## 6. trajectories.jsonl (one line per step, written by the runner)

```json
{"episode_id":"a3f9…","step":3,"action":{"name":"get_record","arguments":{"table":"ledger","id":"L-0F9C"}},
 "observation":{"result":{...},"error":null,"steps_remaining":97},
 "reward":0.0,"terminated":false,"truncated":false,
 "policy_reward":0.0,"mode":"eval","t_ms":4}
```
- The required keys (case §09) are `episode_id, step, action, observation, reward, terminated, truncated`.
- Line `step:0` is the reset: `action:null`, and `observation` is the reset observation.
- `reward` is the env's true reward. `policy_reward` is what the policy saw (0.0 in eval).
- In eval mode, write the file to a run dir the policy can't read, or append the terminal line after the child exits. Observations never contain truth.

## 7. results.jsonl (one line per episode) and summary

```json
{"episode_id":"a3f9…","split":"dev","policy":"reconcile","mode":"eval","is_clean":false,"n_truth":4,
 "termination":"submitted","steps":52,"inspection_steps":51,"wall_s":0.09,
 "tp":3,"fp":1,"fn":1,"precision":0.75,"recall":0.75,"f1":0.75,"reward":0.398,"reward_experimental":null,
 "per_type":{"duplicate_invoice":{"tp":1,"fp":0,"fn":0},"...":{}},
 "match_trace":[{"finding_index":0,"issue_id":"I-2","verdict":"tp","reason":null},
                {"finding_index":3,"issue_id":null,"verdict":"fp","reason":"evidence"}]}
```
`termination` ∈ `submitted | schema_error | max_steps | timeout | policy_crash`.
`summary.json` holds, per policy × split: micro and macro P/R/F1, mean return, mean steps, mean wall time, clean-FP rate, termination counts, per-type table, and `{env_version, schema_version, config_hash, seed_list_hash, python, git_sha}`. The same metadata goes in `run.json`, never in observations.

## 8. Child-process policy protocol (stdin/stdout, one JSON object per line, UTF-8)

Runner → policy:
```json
{"type":"reset","protocol":1,"observation":{...},"info":{"steps_remaining":100,"error_code":null}}
{"type":"step","observation":{...},"reward":0.0,"terminated":false,"truncated":false,"info":{...}}
{"type":"close","reason":"terminated"|"truncated"|"timeout"|"runner_abort"}
```
Policy → runner: exactly one action envelope per line (§2) after each `reset`/`step` message that isn't the end.

Rules:
- Spawn with `Popen([sys.executable,"-I","-m","auditgym.policy_host",spec], env=<scrubbed>, cwd=<empty tmpdir>, start_new_session=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)`. `spec` is `module:Class` (class with `reset(obs, info)` and `act(obs) -> dict`) or `cmd:<argv>` for any language. Use one child per episode.
- `policy_host` calls `resource.setrlimit` (AS≈1.5 GB, CPU 65 s, NOFILE 32, FSIZE 0) before it imports the policy.
- Use a per-step read deadline (`selectors`) and a 60 s episode clock. On expiry: `os.killpg(pgid, SIGKILL)`, `truncated=true`, termination `timeout`.
- Cap each stdout line at 1 MB and stderr at 64 KB (keep the tail). A line that isn't JSON or is over the cap gives `invalid_action`, and the step is consumed.
- EOF or a nonzero exit before the end gives `truncated=true`, termination `policy_crash`.
- In eval mode the policy never receives a reward ≠ 0.0, a grade, or a match trace. `truth.json`, `provenance.json` and the grader log are written only after the child exits.
- The `oracle` is in-runner only. It gets truth through a privileged constructor and must still act through `env.step()`.
