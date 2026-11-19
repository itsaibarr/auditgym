# 05 — Best existing building block per component

Researched 2026-11-19. Release dates, licenses, `requires_python` and dependency lists come from the PyPI JSON API. Repo health comes from the GitHub API. Several claims were **verified empirically** in a throwaway Python 3.11.15 venv (marked ✅ below). Constraints: Python 3.11, offline `docker run --network none`, 2 CPU / 4 GB, 60 s/episode, 1 GB per solution, deadline 22 Nov 2026 23:59 Astana.

Decision rule used (per the user's new preference): **reuse** when the library is solid, light, maintained, and either saves real work or adds correctness evidence; **stdlib** when the library adds risk (CVEs, version-dependent output, heavy deps, float money) for no gain. Changes vs. the earlier "stdlib only" stance: **msgspec**, **tzdata**, **gymnasium (extra)**, **hypothesis**, **pytest-timeout**, plus **mimesis as a dev-time data source**.

---

## 1. Synthetic company data (names, IDs, addresses; kk/ru locales; seeded)

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| **Mimesis** 22.2.0 | https://github.com/lk-geimfari/mimesis · seeding: https://mimesis.name/latest/random_and_seed.html | MIT | 2026-09-23 | **Pick, dev-time only** (bake a vendored name bank) |
| Faker 40.54.0 | https://github.com/joke2k/faker | MIT | 2026-11-10 | Fallback for `en`; no `kk_KZ` |
| Own vendored lists | n/a | ours | n/a | **Runtime choice** (fed by Mimesis) |

Findings:
- **Neither library promises cross-version output stability.** Faker's maintainers say reproducibility will keep breaking because datasets change every release, so the advice is to pin an exact version (https://faker.readthedocs.io/en/master/ ; issue thread mirrored at https://lightrun.com/answers/joke2k-faker-reproducibility-of-results-broken-between-faker-0718-and-faker-080-087). Mimesis's seed docs say nothing about cross-version stability. Its `mimesis/random.py` subclasses stdlib `random.Random`, so it also inherits CPython's caveat that `choice`/`shuffle` may change between versions.
- **Locales:** Faker has `ru_RU` and `uz_UZ` but **no `kk_KZ`** (checked `faker/providers/person/` via the GitHub API). Mimesis ships `datasets/kk/*` and `datasets/ru/*` with gendered first names and surnames ✅ (`Person(Locale.KK, seed=42).full_name()` → `'Ажарбек Ибрагимова'`). That sample pairs a male first name with a female surname, so build `name + surname[gender]` ourselves rather than calling `full_name()`.
- ⚠️ **Mimesis `kk/finance.json` lists real companies** (Air Astana, Halyk Bank, Kaspi, ForteBank: https://github.com/lk-geimfari/mimesis/blob/master/mimesis/datasets/kk/finance.json). The case §12 says all companies must be fictional, so **never use Mimesis/Faker company or bank lists**. Compose counterparties as `surname/word + legal form` (`ТОО`, `АО`, `ИП`, `LLP`, `GmbH`).
- Size: Mimesis 10 MB installed, Faker 11 MB. Neither belongs in the 1 GB image if a 30 KB JSON does the job.

**Pick:** `tools/build_namebank.py` (dev-only) reads Mimesis `kk`/`ru`/`en` person datasets, dedupes and sorts them, and writes `auditgym/data/namebank.json` with `{"source": "mimesis==22.2.0 (MIT)", ...}`, which is committed. At runtime we draw from it with our own seeded RNG (§9). That gives byte-stable output forever, real Kazakh/Russian names, and no runtime dependency. IDs (`INV-…`, `PAY-…`, entity IDs) come from our own formatter plus the seeded bijection used for the paired ID-renaming checks.

## 2. Money & currency (exponents, Decimal rounding, FX)

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| py-moneyed 3.0 | https://github.com/py-moneyed/py-moneyed | BSD-3 | 2022-11-27 (repo last push 2024-04) | Reject: drags **Babel (32 MB installed)**; we store integers anyway |
| `money` 1.3.0 | https://pypi.org/project/money/ | MIT | 2016-04-17 | Reject: dead |
| `prices` 1.1.1 | https://pypi.org/project/prices/ | BSD | 2022-10-07 | Reject: dead; depends on Babel |
| `iso4217` 1.16.20260101 | https://github.com/dahlia/iso4217 | Public domain | 2026-03-03 | **Optional test oracle** for our exponent table |
| pycountry 26.2.16 | https://pypi.org/project/pycountry/ | **LGPL-2.1** | 2026-02-17 | Reject: 7.8 MB, LGPL, no minor-unit field we need |
| Babel 2.18.0 | https://babel.pocoo.org | BSD-3 | 2026-02-01 | Reject: CLDR digits ≠ ISO 4217 (see 02 §1 row 16) |
| `forex-python` & co. | — | — | — | Reject: need network, and real rates are forbidden |

✅ Exponents agree with SIX List One: `iso4217.Currency['kzt'].exponent == 2`, `jpy 0`, `kwd 3`, `clf 4`, `isk 0`, `uyw 4`. py-moneyed's `sub_unit` agrees too (KZT 100, JPY 1, KWD 1000), and `Money(1,'USD') + Money(1,'KZT')` raises `TypeError`. That's nice, but our money is already `int` minor units by spec, so a `Money` class adds nothing.

**Pick: stdlib `decimal`.** Use one `convert(doc_minor, doc_ccy, base_ccy, rate_str, rounding)` that does `(Decimal(doc_minor) * Decimal(rate_str) * Decimal(10) ** (exp_base - exp_doc)).quantize(Decimal(1), rounding=…)` inside a `decimal.localcontext(prec=50, traps=[InvalidOperation, Overflow, DivisionByZero])`. Use a vendored ~15-row exponent dict (SIX List One, 2026-09-17), plus one test asserting it equals `iso4217` if that dev dependency is installed. Fowler-style `allocate()` for splitting payments without losing remainders is about 10 lines.

## 3. Double-entry engine for the hidden ledger

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| python-accounting 2.1.0 | https://github.com/ekmungai/python-accounting | MIT | 2026-10-30 | **Reject as dependency**: its `pyproject.toml` hard-requires `sqlalchemy`, **`mysqlclient`** and **`psycopg2`** (compiled DB drivers). Borrow the *Assignment/clearing* concept only. |
| pyluca 3.1.3 | https://github.com/datasignstech/pyluca | MIT | 2025-04-14 (GitHub tag; **not on PyPI**, 404) | Reject: `JournalEntry(dr_amount: float, cr_amount: float)`, i.e. float money |
| django-hordak 2.0.0 | https://pypi.org/project/django-hordak/ | MIT | 2024-11-29 | Reject: Django, MPTT, py-moneyed, requests… Concepts only (leg sign convention, `Transaction`/`Leg`) |
| Beancount 3.2.3 | https://github.com/beancount/beancount | **GPL-2.0-only** | 2026-05-05 | **Optional dev-only cross-check**, never shipped or imported |
| hledger / ledger-cli | https://hledger.org · https://github.com/ledger/ledger | GPL-3 / BSD | active | Same as Beancount (manual sanity check) |

Beancount license implications: GPL obligations attach when you *distribute* GPL code or a derivative. A dev-only script that **exports our ledger to `.beancount` text and runs the `bean-check` CLI as a separate process** (beancount v3 still ships `bean-check`; `bean-query` moved to `beanquery`: https://beancount.github.io/docs/installing_beancount_v3/) doesn't make our code a derivative, so MIT/Apache stays clean. Two conditions: don't put beancount in `uv.lock` main/dev groups that land in the Docker image, and don't `import beancount` from shipped modules. The value is modest, because `bean-check` re-verifies I1/I2 (balanced entries, trial balance) and balance assertions that our own invariants already check. Treat it as a nice demo slide, not a requirement.

**Pick: our own ~150-line posting engine** (templates T1–T13 from 02 §4.5, `int` minor units, invariants I1–I7 as pure functions). No Python double-entry library is both light and integer/Decimal-safe.

## 4. Safe calculator (arithmetic + decimals only)

| Candidate | URL | License | Last release | Security record | Verdict |
|---|---|---|---|---|---|
| simpleeval 1.0.8 | https://github.com/danthedeckie/simpleeval | MIT | 2026-09-12 | **CVE-2026-32640** (attribute chains reach `os`/`sys`, fixed 1.0.5, CVSS up to 9.8: https://nvd.nist.gov/vuln/detail/CVE-2026-32640) plus **3 more advisories on 2026-09-12** (`operator` module escape, `set`/`frozenset` bypass, extra `os` funcs: https://github.com/danthedeckie/simpleeval/security/advisories) | Reject |
| asteval 1.0.10 | https://github.com/lmfit/asteval | MIT | 2026-08-21 | **CVE-2025-24359** format-string sandbox escape (fixed 1.0.6: https://osv.dev/vulnerability/CVE-2025-24359); further escapes in March 2025 (https://seclists.org/fulldisclosure/2025/Mar/0) | Reject: it's a full Python-subset interpreter, far more surface than we need |
| numexpr 2.14.2 | https://github.com/pydata/numexpr | MIT | 2026-07-18 | CVE-2023-39631 code execution via `evaluate` (fixed 2.8.5: https://advisories.gitlab.com/pypi/numexpr/CVE-2023-39631/) | Reject: needs numpy, float only |
| **Own `ast` walker over `Decimal`** | https://docs.python.org/3.11/library/ast.html | ours | n/a | Whitelist of 5 node types | **Pick** |

✅ simpleeval with `names={}, functions={}` blocks `().__class__`, `9**9**9` and `'a'*10**9`, but `0.1+0.2` returns `0.30000000000000004` (float). Money needs `Decimal`, and making simpleeval Decimal-native means overriding its number handling, which is more code than writing the whole thing. Its six advisories in eight months come from being a general evaluator; ours only needs numbers.

**Pick:** about 60 lines, using:
- `ast.parse(expr, mode="eval")` with `len(expr) ≤ 256`, all exceptions caught, and a bounded error code.
- Allowed nodes: `Expression`, `BinOp(Add|Sub|Mult|Div|Mod?)`, `UnaryOp(USub|UAdd)`, `Constant(int|float literal → Decimal(str(token)))`. Take the literal text via `ast.get_source_segment` so `0.1` becomes `Decimal('0.1')`, never a float.
- Either no `**`, or `Pow` with an integer exponent `|e| ≤ 12`.
- A local `decimal` context: `prec=34`, `Emax=10**6`, traps → bounded `calc_error`.
- Borrow simpleeval's *limit ideas* (MAX_POWER, length cap) as citations, not code.
- Test with Hypothesis plus a corpus of known simpleeval/asteval escape payloads (all must return `calc_error`).

## 5. Schemas & validation (actions, findings, manifest) + JSON Schema export

| Candidate | URL | License | Last release | Weight | Verdict |
|---|---|---|---|---|---|
| **msgspec** 0.22.0 | https://github.com/msgspec/msgspec · https://jcristharif.com/msgspec/jsonschema.html | BSD-3 | 2026-09-29 (0.20 2025-11, 0.21 2026-04; repo pushed 2026-11-18) | 1 compiled pkg, **0 deps**, 0.5 MB, cp311 wheels x86_64+aarch64 | **Pick** |
| pydantic 2.14.3 | https://docs.pydantic.dev | MIT | 2026-11-05 | 4 deps incl. pydantic-core (Rust, 5.2 MB) | Runner-up (nicer errors and schema titles, 7 MB) |
| jsonschema 4.26.0 | https://github.com/python-jsonschema/jsonschema | MIT | 2026-01-07 | attrs, referencing, rpds-py (Rust), specs | Reject: validation only, hand-written schemas drift from code |
| fastjsonschema 2.22.3 | https://github.com/horejsek/python-fastjsonschema | BSD-3 | 2026-11-01 | pure, 0 deps | Viable if you'd rather hand-write `.json` schemas (it `exec`s generated code, which is fine for our own schemas) |
| attrs 26.1.0 + cattrs 26.2.1 | https://github.com/python-attrs/cattrs | MIT | 2026-09-26 | 3 deps | Reject: no JSON Schema export |

✅ msgspec, verified on 3.11 with a `Findings` Struct (`forbid_unknown_fields=True`, `Meta(max_length=30/10/1000)`, `Literal` type enum, currency `pattern`):
- `claimed_amount_minor: true`, `100.0` and `"100"` are **all rejected** with a path (`$.findings[0].claimed_amount_minor`). Strict by default, so bool≠int, which the "invalid structure → −1" rule needs.
- Unknown fields are rejected.
- `msgspec.json.schema(Findings)` emits JSON Schema 2020-12 with `$defs`. Commit it as `schema/findings.schema.json` and reuse it for LLM `tools`.
- Gotchas: (a) `10**30` was **accepted** for an `int`, so add `Meta(ge=-10**15, le=10**15)`. (b) Duplicate JSON keys are accepted (last wins, the same as stdlib `json`), so document it, or reject it with a pre-pass using `json.loads(object_pairs_hook=…)` if you want to be strict.
- Actions: decode `{"name": Literal[...], "arguments": dict}` as an envelope, then `msgspec.convert(arguments, ArgsByName[name])`. A bad shape yields a bounded `invalid_action` error, uses a step, and doesn't end the episode.

## 6. One-to-one matching with deterministic tie-breaks

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| `scipy.optimize.linear_sum_assignment` (scipy 1.17.1 = last for py3.11; 1.18 needs ≥3.12) | https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html | BSD-3 | 2026-08-21 (1.18.1) | Reject |
| munkres 3.0.0 | https://pypi.org/project/munkres/ | Apache-2.0 | 2026-10-18 | Reject: ⚠️ **PyPI name changed hands** (bmc 1.1.4 in 2020 → new maintainer pushed 2.0.0 and 3.0.0 with compiled wheels within 12 days, Oct 2026; homepage `iameishit/Paldita-munkres`). Supply-chain freshness flag. |
| hungarian-algorithm 0.1.11 | https://pypi.org/project/hungarian-algorithm/ | MIT | 2020-05-24 | Reject: dead |
| networkx 3.6.1 `max_weight_matching` | https://networkx.org | BSD-3 | 3.7 needs ≥3.12 | Reject: 2 MB for one call |
| **Eligibility + canonical greedy (own, ~30 lines)** | — | ours | — | **Pick** |

✅ **scipy's tie choice depends on input order.** With a 2×2 all-tie cost matrix, the original order gives `(F_a,I_x),(F_b,I_y)`; permuting the rows gives `(F_a,I_y),(F_b,I_x)`. The docs make no tie-break promise. That's dangerous for the jury's paired "permute order / rename IDs" checks (15 pts). It also costs **93 MB installed** (scipy 71 MB + numpy 22 MB).

Why greedy is *optimal* here, not just simple: a TP needs an exact **type + canonical `affected_ids`** match, plus amount within tolerance, currency, and evidence. Two generator assertions make the bipartite graph a disjoint union of stars, and on a star "first eligible finding wins" achieves the maximum matching:
- Assert in the generator that `(type, sorted(affected_ids))` is **unique across truth issues**. If two would collide, merge them into one issue.
- Each finding is then eligible for **at most one** issue.

Published rule: iterate truth issues by `issue_id`. For each issue, the matched finding is the eligible one with the lowest submission index. All others are FP, with reason `duplicate` / `no_issue` / `amount` / `currency` / `evidence`, logged in the match trace. TP/FP/FN counts are then invariant to record order and ID renaming by construction. A Hypothesis test can compare against brute-force maximum matching (`itertools.permutations`, n ≤ 6) to prove it.

## 7. Env contract & checking

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| **Gymnasium** 1.4.0 | https://gymnasium.farama.org/api/env/ · checker: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/utils/env_checker.py | MIT | 2026-10-05 | **Pick as optional extra** `[gym]`, installed in Docker so `check` runs Gymnasium's own `check_env` on our wrapper |
| PettingZoo 1.27.0 | https://pettingzoo.farama.org | MIT | 2026-08-13 | Reject: multi-agent API, irrelevant |
| OpenEnv (`openenv` 0.14.0 / `openenv-core` 0.3.0) | https://pypi.org/project/openenv/ | BSD-3 | 2026-11-19 | Reject: requires FastAPI, uvicorn, **gradio**, openai, fastmcp, huggingface_hub, websockets…; too heavy for 1 GB and offline |

Notes:
- Core stays a plain class (the case allows this). `contrib/gymnasium_env.py` (~40 lines) subclasses `gym.Env` with `action_space = observation_space = Text(max_length=…, charset=<our alphabet>)` (JSON strings) and calls `super().reset(seed=seed)` because `check_reset_seed_determinism` asserts `env.unwrapped._np_random` is set.
- The default `Text` charset is alphanumeric only (https://gymnasium.farama.org/api/spaces/fundamental/), so the charset must include JSON punctuation and Cyrillic.
- Free bonus: `check_env` calls `step(action_space.sample())` with random strings, which exercises our invalid-input path.
- Gymnasium pulls numpy. numpy **2.5.x needs Python ≥3.12**, so pin **numpy 2.4.6** (2026-05-18, cp311 wheels).
- Copy the *patterns* of `check_step_determinism` and `data_equivalence` into our own stdlib checks (MIT), so `check` doesn't depend on the extra being installed.

## 8. Process isolation for the policy runner

| Option | URL | Verdict |
|---|---|---|
| **`subprocess.Popen` + JSON lines over stdin/stdout** | https://docs.python.org/3.11/library/subprocess.html | **Pick** |
| `multiprocessing` spawn | https://docs.python.org/3.11/library/multiprocessing.html | Reject: pickling and shared runner code make it easy to leak objects; not language-agnostic |
| `resource.setrlimit` | https://docs.python.org/3.11/library/resource.html | **Pick**: call it *inside* `policy_host` as its first statement (not via `preexec_fn`) |
| Docker flags | https://docs.docker.com/engine/network/drivers/none/ | **Pick**: `--network none --read-only --tmpfs /tmp --cap-drop ALL --pids-limit 256 --memory 4g --cpus 2` (documented in README) |
| py-landlock / sandlock | https://pypi.org/project/py-landlock/ · https://pypi.org/project/sandlock/ | Optional, not default: need kernel ≥5.13 (sandlock ≥6.7–6.12); jury kernel unknown |
| nsjail / bubblewrap / firejail | — | Reject: need user namespaces, which Docker's default seccomp profile often blocks |

Recipe (all stdlib):
- Spawn with `Popen([sys.executable, "-I", "-m", "auditgym.policy_host", spec], stdin=PIPE, stdout=PIPE, stderr=PIPE, env={"PATH": "/usr/bin", "PYTHONHASHSEED": "0"}, cwd=<empty tmpdir>, start_new_session=True)`.
- If running as root in the container, add `user="nobody", extra_groups=()`. The docs warn `user=` alone keeps supplementary groups.
- `policy_host` first runs `resource.setrlimit` for `RLIMIT_AS` (~1.5 GB), `RLIMIT_CPU` (65 s), `RLIMIT_NOFILE` (32) and `RLIMIT_FSIZE` (0), then imports the policy.
- Per-step deadline via `selectors` on the stdout fd (or a reader thread plus `queue.get(timeout=…)`), and an episode wall clock of 60 s, which yields `truncated=true` with empty findings.
- On timeout, `os.killpg(proc.pid, SIGKILL)`. The docs note `communicate(timeout)` does **not** kill the child.
- Avoid `preexec_fn`: the docs call it "NOT SAFE … in the presence of threads".

## 9. Determinism / reproducibility

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| **`random.Random` (stdlib)** | https://docs.python.org/3/library/random.html#notes-on-reproducibility | PSF | — | **Pick**, plus our own `choice`/`shuffle`/`randbelow` built on `.random()` |
| numpy `Generator(PCG64)` | https://numpy.org/neps/nep-0019-rng-policy.html | BSD | — | Reject: NEP 19 guarantees streams only for bit-generator `bytes/integers/random`; **`Generator` methods may change in X.Y releases** |
| **stdlib canonical JSON** `json.dumps(o, sort_keys=True, separators=(",",":"), ensure_ascii=False)` | — | — | — | **Pick** |
| rfc8785 0.1.4 (Trail of Bits) | https://github.com/trailofbits/rfc8785.py | Apache-2.0 | 2024-09-27 (repo pushed 2026-11-11), pure, 0 deps | **Optional test oracle** |
| jcs 0.3.0 | https://github.com/titusz/jcs | Apache-2.0 | 2026-11-06 (5 stars) | Reject: too young |
| canonicaljson 2.0.0 (Matrix) | https://github.com/matrix-org/python-canonicaljson | Apache-2.0 | 2023-03-15 | Reject: Matrix flavour, not JCS |
| **`hashlib.sha256` / `hmac`** | stdlib | — | — | **Pick**: episode fingerprint = sha256(canonical JSON of all public files); `episode_id = hmac(secret, seed)[:16]` |
| **`tzdata` 2026.6** (PyPI) + `PYTHONTZPATH=""` | https://pypi.org/project/tzdata/ · https://docs.python.org/3.11/library/zoneinfo.html | Apache-2.0 | 2026-10-28, pure, 0 deps | **Pick (runtime dep)** |

Notes:
- ✅ RFC 8785 equals stdlib canonical JSON for our value domain. On a sample with Cyrillic, `€`, `null`, booleans, negative ints and decimal *strings*, `rfc8785.dumps(obj) == json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()`. This holds because we never emit floats (money is int, rates are strings) and keys are ASCII, so UTF-16 vs code-point key order can't differ. Add a one-line guard: assert there are no floats in emitted data.
- **tzdata is a determinism finding, not a convenience.** `zoneinfo` uses system tz files first, and the system copy differs between the jury's image, macOS and Debian. Debian trixie also moved legacy names to `tzdata-legacy` (open docker-library/python issue: https://issues.ecosyste.ms/hosts/GitHub/repositories/docker-library%2Fpython/issues). Kazakhstan's offsets changed recently (Asia/Almaty moved to +05 in 2024). Setting `PYTHONTZPATH=""` (or `zoneinfo.reset_tzpath(to=())` at import) forces the **pinned** `tzdata` wheel, so period boundaries and DST distractors are identical everywhere.
- Seed derivation: `random.Random(hashlib.sha256(f"{seed}|{stage}".encode()).digest())` per generator stage. Bytes seeding (version 2) is stable, and `.random()` is the only method CPython guarantees across versions. Python is pinned to 3.11 in Docker anyway.

## 10. Testing

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| **pytest** 9.2.0 | https://docs.pytest.org | MIT | 2026-10-27 | **Pick** |
| **Hypothesis** 6.170.15 | https://hypothesis.readthedocs.io | MPL-2.0 (dev-only, no copyleft impact on our code) | 2026-11-06, needs py≥3.11 | **Pick**: grader invariants + `RuleBasedStateMachine` for the env contract |
| **pytest-timeout** 2.4.0 | https://github.com/pytest-dev/pytest-timeout | MIT | 2025-05-05 | **Pick**: the single `check` command can never hang |
| syrupy 7.0.2 | https://github.com/syrupy-project/syrupy | MIT | 2026-11-16 | Optional: golden sha256 fingerprints in a committed JSON do the same with stdlib |
| mutmut 3.8.1 | https://github.com/boxed/mutmut | BSD-3 | 2026-10-28 | **One-off via `uvx mutmut`** on `matcher.py`/`reward.py`; report the kill rate in the design note. Not in lock or image (pulls libcst, textual). |
| cosmic-ray 8.9.0 | https://github.com/sixty-north/cosmic-ray | MIT | 2026-11-10 | Reject: 13 deps incl. aiohttp, sqlalchemy |
| hypothesis-jsonschema 0.23.2 | https://github.com/python-jsonschema/hypothesis-jsonschema | MPL-2.0 | 2026-10-27 | Reject: needs jsonschema/rpds-py; hand strategies are better for targeted attacks |

Hypothesis setup (Context7: `/websites/hypothesis_readthedocs_io_en`): register a profile `settings.register_profile("check", derandomize=True, database=None, deadline=None, max_examples=200)` and load it in `auditgym check`. That makes the check reproducible and avoids writing `.hypothesis/` on a `--read-only` container. Properties to encode:
- Duplicates never raise F1.
- Flag-all reward ≤ silence on clean books.
- Oracle ≥ any perturbation.
- Reward is invariant under record permutation and ID bijection.
- Step cost only separates equal-F1 answers by ≤ 0.2.
- Calculator never raises past its boundary.

## 11. Trajectory logging & run reports

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| jsonlines 4.0.0 | https://github.com/wbolster/jsonlines | BSD | 2023-09-01 (repo idle since 2024-08) | Reject: wraps 2 lines of stdlib, pulls attrs |
| orjson 3.13.0 | https://github.com/ijl/orjson | MPL-2.0 AND (Apache-2.0 OR MIT) | 2026-10-07 | Reject: speed is irrelevant at ≤100 steps; a second serializer risks byte differences vs our canonical form |
| structlog 26.1.0 | https://github.com/hynek/structlog | MIT OR Apache-2.0 | 2026-06-06 | Reject: the trajectory *is* the structured log; stdlib `logging` for stderr |
| Jinja2 3.1.6 | https://jinja.palletsprojects.com | BSD-3 | 2025-03-05 | Reject: a one-page `report.html` is ~80 lines of f-strings plus `html.escape` |

**Pick: stdlib.** Write `trajectories.jsonl` with the same canonical `dumps` as §9, one `write()` + `flush()` per step, so a crash keeps every completed line. `summary.md` holds the P/R/F1/return/steps/wall_s table per policy and per error type, and `report.html` is optional.

## 12. Packaging, reproducible deps, Docker, LICENSE

| Candidate | URL | License | Last release | Verdict |
|---|---|---|---|---|
| **uv** 0.13.9 + `uv.lock` | https://docs.astral.sh/uv/guides/integration/docker/ | MIT OR Apache-2.0 | 2026-11-17 | **Pick** |
| pip-tools 7.7.0 | https://github.com/jazzband/pip-tools | BSD-3 | 2026-11-16 | Fallback only: `uv export --format requirements-txt --hashes > requirements.lock` serves pip users without a second tool |

Dockerfile essentials (Context7 `/astral-sh/uv/0.13.5` Docker guide):
```dockerfile
FROM python:3.11-slim-bookworm@sha256:<pin-digest>
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONHASHSEED=0 PYTHONTZPATH="" PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra gym --group test
COPY . /app
RUN uv sync --locked --extra gym --group test
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["auditgym"]
CMD ["check"]
```
Notes:
- Run with `docker run --rm --network none … auditgym check`. Call the venv binary directly, **not `uv run`**, because `uv run` may try to sync. If you must use it, set `UV_NO_SYNC=1` / `UV_OFFLINE=1`.
- Pin the base image by digest, and use a `--locked` install so a drifted lock fails the build.

**LICENSE: MIT.** Every runtime dep is MIT, BSD or Apache, so MIT is compatible. It's the shortest and most common license in RL-env repos (Gymnasium, verifiers). Copied Gymnasium checker patterns are MIT. Apache-2.0 only adds a patent grant, which doesn't matter for a hackathon env. Keep Beancount (GPL) out of the image (§3). Hypothesis (MPL-2.0) is test-only and unmodified, so it doesn't matter.

CLI: stdlib `argparse` subparsers is enough for ~8 subcommands. Click 8 would be the pick only if the CLI grows (pure, BSD). Typer pulls rich and friends.

---

## Dependency bill of materials

**Runtime (`[project.dependencies]`, in the image):**

| Package | Pin | Justification |
|---|---|---|
| msgspec | `==0.22.0` | Strict decode of policy JSON plus JSON Schema export from one source of truth. 0 deps, 0.5 MB. |
| tzdata | `==2026.6` | Pins IANA tz rules so period cutoffs are identical on every machine (with `PYTHONTZPATH=""`). |

**Optional extra `gym` (installed in the image so `check` runs Gymnasium's `check_env`):**

| Package | Pin | Justification |
|---|---|---|
| gymnasium | `==1.4.0` | Official contract checker named by the case. Used by `contrib/gymnasium_env.py` only. |
| numpy | `==2.4.6` | Gymnasium dependency. The last line supporting Python 3.11 (2.5 needs ≥3.12). |
| cloudpickle | `==3.1.3` | Gymnasium dependency. |
| farama-notifications | `==0.0.6` | Gymnasium dependency. |
| typing-extensions | `==4.16.0` | Gymnasium dependency. |

**Test group (in the image; `auditgym check` runs them):**

| Package | Pin | Justification |
|---|---|---|
| pytest | `==9.2.0` | Required test deliverable. |
| hypothesis | `==6.170.15` | Property tests for the grader invariants the jury scores (25 pts reward robustness). |
| sortedcontainers | `==2.4.0` | Hypothesis dependency. |
| pytest-timeout | `==2.4.0` | Guarantees the single check command terminates. |

**Dev-only (laptop/CI, never in the image):**

| Package | Pin | Justification |
|---|---|---|
| mimesis | `==22.2.0` | One-time generation of the committed kk/ru/en name bank (persons only, never companies). |
| iso4217 | `==1.16.20260101` | Test oracle for the vendored exponent table (skipped if absent). |
| rfc8785 | `==0.1.4` | Test oracle proving our canonical JSON equals RFC 8785 JCS. |
| mutmut | `==3.8.1` (via `uvx`) | One-off mutation score for `matcher.py`/`reward.py`, quoted in the design note. |
| uv | `==0.13.9` | Lock and sync; Docker copies the same version. |
| beancount | `==3.2.3` (optional, isolated venv) | `bean-check` cross-check of exported books; GPL, so a CLI subprocess only. |

pytest's own transitive dependencies (iniconfig, pluggy, packaging, pygments) resolve via `uv.lock`. Runtime pip-style pulls aren't needed; `uv export --hashes` gives a pip fallback.

## Keep stdlib (and why)

- **`decimal`**: money/FX math and rounding modes. Every money library adds Babel or floats and no correctness, since our amounts are `int` minor units by spec.
- **`random.Random` + `hashlib`**: seeded per-stage streams. numpy `Generator` streams may change across X.Y releases (NEP 19). Faker and Mimesis output changes across versions.
- **`ast`**: the calculator. simpleeval, asteval and numexpr together have 7+ sandbox-escape advisories since 2023, and simpleeval computes in float.
- **`json` (canonical dumps), `csv`**: all I/O. Byte-identical to RFC 8785 for our float-free data ✅, so no orjson, jsonlines or JCS dependency.
- **`hmac`/`hashlib`**: episode IDs and fingerprints.
- **`subprocess`, `selectors`, `resource`, `signal`, `os.killpg`**: policy isolation. Language-agnostic and kernel-independent, and works under `--network none`.
- **`zoneinfo`**: time zones, backed by the pinned `tzdata` wheel.
- **`argparse`**: CLI.
- **`logging`, `html`**: logs and the optional report.
- **Plain Python for matching**: unique `(type, affected_ids)` keys make canonical greedy *exactly* maximal. scipy is 93 MB and its tie choice depends on input order ✅.
- **Plain Python posting engine**: the only Python double-entry libraries are float-based (pyluca), DB-driver-bound (python-accounting) or Django (hordak).
