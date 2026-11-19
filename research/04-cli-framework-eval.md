# 04 — CLI/TUI framework evaluation for `auditgym`

Date: 2026-11-19. Deadline: 2026-11-22 23:59 (Astana). Inputs: case.pdf §07–§11 and `01-oss-rl-env-references.md` §6 (stdlib-only core, `auditgym check|run|compare|play`, exit codes 0/1/2).

## TL;DR: decision

**The CLI is Python only: stdlib `argparse` for parsing and dispatch, plus `rich==15.0.0` for output, and only when stdout is a TTY. `play` is a stdlib `cmd.Cmd` REPL that uses Rich for rendering. The demo is recorded with charmbracelet VHS v0.12.1 from a committed `demo.tape`.** No Node anywhere: no termcn, Ink, Clack, Commander or oclif. There's no Typer either.

Why, in one paragraph: the case says outright that "Красивый UI … баллов не дают" (a pretty UI earns no points, case p.5) and "Не тратьте основной срок на … UI" (don't spend the main time on UI, p.4). It also sets a **1 GB per-solution** budget (p.4). The Engineering score rewards one-command runs, versions, logs, failure analysis and limitations, and the UI itself earns nothing. A Node front-end costs **+208 MB** of runtime plus 24 MB of `node_modules`, a second toolchain (a TSX build), and a JSON bridge to the Python core, and it returns no points. Within Python, Rich already provides the polished tables. argparse already provides subcommands, `--help` and **exit code 2 on usage errors**, which matches our 0/1/2 convention for free, and it can be tested with plain `main(argv) -> int` + pytest `capsys`. Typer would add little, and its last few releases have been churning (see below).

---

## 1. What termcn actually is

- "Beautiful terminal UI components, built on Ink and OpenTUI. 100% Free, Zero config, one command setup." It's MIT-licensed, created 2026-04-03, has about 1.2k stars, and was last pushed 2026-11-11. https://github.com/shadcn-labs/termcn · https://www.termcn.dev/
- It's a **shadcn-style registry of React (TSX) source files**, not an npm library. You copy components into your repo with `npx shadcn@latest add @termcn/ink/table` (https://github.com/shadcn-labs/termcn/blob/main/apps/web/content/docs/(root)/registry.mdx), and they land in `components/ui/`. The runtime dependency is `ink`, declared unversioned (registry.json).
- The catalog is large. Table ("sortable, selectable, paginated"), Data Grid, Select, Text Input, Spinner, Progress, Tabs, Command Palette, Diff View, JSON, Bar/Sparkline/Heat Map charts, and templates such as App Shell and Usage Monitor. https://www.termcn.dev/llms.txt
- **Verdict:** it's the best way to make a *Node/React* TUI look good quickly. For us it would need Node ≥22, React ≥19.3, Ink 8, a TSX build step and a process bridge to Python. It solves a problem we don't have: we need one good table and a REPL.

## 2. Comparison table

Versions, dates, licenses and dependency weights were fetched live from PyPI, npm and the GitHub API on 2026-11-19. Install sizes and import times were measured locally in clean `uv` Python 3.11 venvs and npm dirs, best of 5 runs of `-X importtime`.

| Candidate | Lang | License | Latest (date) | Install weight | Import/start | Py 3.11 | Non-TTY/CI behavior | Tables | Interactive play | Testability |
|---|---|---|---|---|---|---|---|---|---|---|
| **argparse** (stdlib) | Py | PSF | ships with 3.11 | 0 | 4 ms | ✅ | Plain text always; usage error → exit 2 | none (we write md/json) | n/a (pair with `cmd`) | `main(argv)` + `capsys` |
| **Rich** | Py | MIT | 15.0.0 (2026-04-12) | 7.1 MB, 4 dists (rich, pygments, markdown-it-py, mdurl) | 24 ms (console+table) | ✅ (≥3.9) | Auto: no ANSI when not a TTY; honors `NO_COLOR`, `TERM=dumb`, `TTY_COMPATIBLE`, `TTY_INTERACTIVE`, `COLUMNS` ([docs](https://github.com/textualize/rich/blob/main/docs/source/console.rst)). **Gotcha (measured):** box chars stay, and width defaults to 80, so wide tables wrap or ellipsize when piped | `rich.table.Table`, `box.MARKDOWN` | `Prompt.ask(choices=…)`, panels | `Console(record=True).export_text()` |
| Click | Py | BSD-3 | 8.5.1 (2026-11-06) | 0.5 MB, 0 deps | 16 ms | ✅ (≥3.10) | Plain; strips ANSI when piped | none | `click.prompt` | `click.testing.CliRunner` |
| Typer | Py | MIT | 0.29.2 (2026-11-18) | **15.8 MB, 12 dists** (now pulls **pydantic 2.14 + pydantic-core**; vendors click) | 30 ms | ✅ (≥3.11, 3.10 dropped in 0.29.2) | Rich help/errors; plain when piped | via Rich | `typer.prompt` | `typer.testing.CliRunner` ([docs](https://typer.tiangolo.com/tutorial/testing)) |
| rich-click | Py | MIT | 1.9.9 (2026-09-05) | Click + Rich | ~Click+Rich | ✅ | as Click/Rich | via Rich | as Click | CliRunner |
| cyclopts | Py | Apache-2.0 | 5.5.2 (2026-11-16) | 10.0 MB, 8 dists | 39 ms | ✅ (≥3.11) | Rich help; plain when piped | via Rich | no | `app(argv)` direct |
| Textual | Py | MIT | 8.2.8 (2026-06-30) | 11.8 MB, 9 dists | **106 ms** | ✅ | Full-screen app; **needs a TTY** (`docker run -it`) | DataTable widget | Excellent (widgets, keybindings) | `App.run_test()` Pilot (async) |
| questionary | Py | MIT | 2.1.1 (2025-08-28) | 5.0 MB (prompt_toolkit) | 68 ms | ✅ | Prompts need a TTY | none | Arrow-key menus | awkward (pt input pipes) |
| InquirerPy | Py | MIT | 0.3.4 (**2022-06-27**) | prompt_toolkit + pfzy | — | ✅ | needs TTY | none | menus | awkward. **Unmaintained (last push 2024-08). Reject.** |
| termcn | TSX | MIT | registry, pushed 2026-11-11 | source copied in + Ink/React | Node start | ✗ (Node) | Inherits Ink | Table/DataGrid | Rich widget set | ink-testing-library |
| Ink | JS | MIT | 8.0.0 (2026-10-03), Node ≥22, React ≥19.3 peer | **24 MB node_modules, 40 pkgs** (measured) | Node start | ✗ | In CI or non-TTY "only the last frame is rendered on exit" ([readme](https://github.com/vadimdemedes/ink/blob/master/readme.md)); raw-mode input throws without a TTY | flexbox `<Box>`s | `useInput` | ink-testing-library, `renderToString` |
| @clack/prompts | JS | MIT | 1.9.0 (2026-11-09), Node ≥20.12 | 0.28 MB, 5 pkgs | Node start | ✗ | `CI=true` → non-animated spinner lines; `taskLog` buffers when not a TTY ([src](https://github.com/bombshell-dev/clack/blob/main/packages/prompts/src/spinner.ts)) | none | select/text/confirm/spinner | mock streams |
| Commander.js | JS | MIT | 15.1.0 (2026-11-09), Node ≥22.12 | 0.24 MB, 0 deps | Node start | ✗ | Plain | none | none | `parseAsync([...])` |
| oclif (@oclif/core) | JS/TS | MIT | 5.1.12 (2026-11-09), Node ≥22 | 4.0 MB, 39 pkgs (generator `oclif` 6.0.15 pulls the AWS SDK) | slowest | ✗ | Plain | `ux.table` removed in v4 (needs plugin) | via inquirer | `@oclif/test` |

The Node runtime cost was measured: the official `node-v22.23.5-linux-x64` is a 31 MB download and **208 MB extracted**, of which `bin/node` is 128 MB. That's 20% of the 1 GB budget before a single line of UI code.

## 3. Scoring against our goals

Each goal is scored 0–2. Weights reflect the jury rubric and the 3-day deadline.

| Goal (weight) | argparse + Rich | Typer (+Rich) | Click / rich-click | cyclopts | Textual | Node: Commander + Clack | Node: Ink + termcn |
|---|---|---|---|---|---|---|---|
| Single runtime, offline image, fits 1 GB (×3) | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| `check`/`compare` degrade to plain text + JSON when piped (×3) | 2 | 2 | 2 | 2 | 1 | 1 | 0 |
| Testable inside `auditgym check` with pytest (×2) | 2 | 2 | 2 | 2 | 1 | 0 (cross-lang) | 0 |
| Time to build in 3 days (×2) | 2 | 2 | 2 | 1 (less familiar) | 0 | 1 | 0 |
| Jury UX: readable compare table, good `--help` (×1) | 2 (Rich table; plain help) | 2 | 2 | 2 | 2 | 1 | 2 |
| `play` mode quality (×1) | 1 (REPL + tab complete) | 1 | 1 | 1 | 2 | 2 | 2 |
| Dependency stability/pinning (×1) | 2 (Rich only) | 0 (pydantic added 0.29.0, breaking changes in 0.27/0.29, 3.10 dropped 0.29.2) | 2 | 1 | 1 | 2 | 1 (termcn unversioned `ink` dep) |
| **Weighted total (max 26)** | **25** | 23 | 25 | 22 | 15 | 10 | 6 |

argparse and Click tie. **argparse wins the tie** because it adds zero dependencies (01 §6.3 already settled on a stdlib-only core), its exit code 2 on usage errors matches our 0/1/2 contract, and `main(argv) -> int` is as testable as CliRunner. If the team prefers decorators, `click==8.5.1` is the acceptable substitute (still 0 transitive deps). **Not Typer:** 0.29.0 (2026-11-09) made pydantic a required dependency and removed `click_type`, and 0.27.0 changed metavar printing ([releases](https://github.com/fastapi/typer/releases)). That's too much churn to adopt 3 days before a deadline, and pinning 0.28.0 to dodge it means starting on a version that's already stale.

## 4. Node vs Python: the trade-off, settled

**A Node CLI is not worth it, and neither is a hybrid.**

- **Cost:** +208 MB of Node and +24 MB of Ink/React inside a 1 GB solution budget. That means two lockfiles, a TSX build (termcn ships source that needs `tsx`/esbuild/bun), a subprocess+JSON bridge to the Python runner (which duplicates the policy wire protocol from 01 §6.1-B), and two test runners. All of it has to be offline-buildable.
- **Benefit:** a nicer `play` screen and nicer tables. That's worth 0 rubric points ("Красивый UI … баллов не дают").
- **Risk:** in the jury's `docker run --rm --network none auditgym check` (no `-t`), stdout is not a TTY. Ink then renders only the final frame and raw-mode input throws, so the Node layer adds nothing in the one command that's actually scored.
- **The only hybrid worth considering** is a separate, optional `ui/` folder **outside** the Docker image that reads `runs/*/summary.json`. That's post-deadline work at best. Don't start it before 2026-11-22.

## 5. Demo recorder

| Tool | License | Latest | How | Output | Verdict |
|---|---|---|---|---|---|
| **charmbracelet VHS** | MIT | v0.12.1 (2026-09-24) | Scripted `.tape` (`Type`, `Enter`, `Sleep`, `Wait /regex/`, `Hide/Show`, `Env`, `Require`) | **mp4, gif, webm, PNG frames, several `Output` lines per tape** ([README](https://github.com/charmbracelet/vhs)) | **Pick.** It's reproducible: re-render the video after every fix with `vhs demo.tape`. Install with `brew install vhs` (needs `ttyd` + `ffmpeg`; ffmpeg is already present locally) or `docker run --rm -v $PWD:/vhs ghcr.io/charmbracelet/vhs demo.tape` |
| asciinema + agg | GPL-3.0 (tools only, not shipped) | asciinema 3.2.1 (2026-06-16), agg 1.9.0 (2026-05-29) | Live `asciinema rec` → `.cast` → `agg` | GIF only (gifski); mp4 needs ffmpeg | Fallback for unscripted live takes. Not reproducible |

Run the tape on the host, typing the real jury commands (`docker run --rm --network none auditgym check`), so the video shows exactly what the jury will run. Use `Wait /PASS|FAIL/` instead of fixed `Sleep`s so timing changes don't break it.

```elixir
# demo.tape
Output demo.mp4
Output demo.gif
Require docker
Set FontSize 18
Set Width 1400
Set Height 800
Set Theme "Catppuccin Mocha"
Type "docker run --rm --network none auditgym check" Enter
Wait+Screen@120s /checks passed|FAILED/
Sleep 3s
Type "docker run --rm -t -v $PWD/runs:/app/runs auditgym compare runs/random runs/flag_all runs/reconcile runs/oracle" Enter
Wait+Screen /oracle/
Sleep 4s
Type "docker run --rm -it auditgym play --split dev --episode 0" Enter
Wait+Screen /audit>/
Type "list ledger" Enter
Sleep 2s
Type "get INV-0042" Enter
Sleep 2s
Type "flag duplicate_invoice INV-0042 INV-0057" Enter
Sleep 1s
Type "submit" Enter
Wait+Screen /F1/
Sleep 4s
```

## 6. Pinned versions

```toml
# pyproject.toml
[project]
requires-python = ">=3.11,<3.12"
dependencies = ["rich==15.0.0"]          # pulls markdown-it-py, mdurl, pygments: pin via lock (uv lock / pip-compile)
[project.optional-dependencies]
dev = ["pytest==9.2.0"]
[project.scripts]
auditgym = "auditgym.cli:main"
```
- Base image: `python:3.11.18-slim-bookworm` (latest 3.11 slim tag on Docker Hub, 2026-11-19), with `PYTHONHASHSEED=0`.
- Optional: `rich-argparse==1.8.0` (`RichHelpFormatter`, 5 ms import) for colored `--help`. Skip it unless there's spare time on day 3.
- Demo (host only, not in the image): VHS v0.12.1. Fallback asciinema 3.2.1 + agg 1.9.0.
- Keep the presentation boundary strict: **only `auditgym/cli/render.py` imports `rich`.** Core, generator, grader, env and runner stay stdlib-only, so `check`'s core assertions can't be broken by a UI dependency.

## 7. `auditgym compare`: output sketch

Interactive (TTY, `docker run -t …` or local shell), a Rich table:

```
 auditgym compare · split=dev · 10 episodes (2 clean) · env 0.1.0 · config 3f9a2c1 · seeds sha 9be1…

  policy        P      R     F1   return   steps   clean-FP%   time/ep   term (sub/max/to/err)
 ──────────────────────────────────────────────────────────────────────────────────────────────
  oracle     1.00   1.00   1.00    0.94     31.2         0.0     0.04s    10/0/0/0
  reconcile  0.82   0.64   0.72    0.32     58.7        50.0     0.09s     9/1/0/0
  flag_all   0.07   0.91   0.13   -0.94    100.0       100.0     0.11s     0/10/0/0
  random     0.04   0.09   0.06   -1.00    100.0        50.0     0.10s     0/10/0/0
  mine       0.71   0.70   0.70    0.31     44.0         0.0     1.92s    10/0/0/0

 best per column in bold · clean-FP% = share of clean episodes with ≥1 finding · details: runs/<name>/results.jsonl
 failures by category (mine): missed fx_rounding ×3 · wrong evidence ×1 · late submit ×0   → auditgym explain runs/mine
```

- Columns: P, R, F1 (micro over findings), mean return (case formula `clip(2*F1 - 1 - 0.002*steps, -1, 1)`), mean steps, clean-FP%, wall time per episode, and a termination breakdown. Add a header line with env version, config hash and seed-list hash, which covers "версии конфигураций" (config versions).
- Same rows as data: `compare` always writes `runs/compare.json` and `runs/compare.md`.
- Flags: `--format {auto,table,md,json,csv}` (default `auto` = `table` when stdout is a TTY, `md` otherwise), `--sort f1`, `--split`.
- The one-line "failures by category" footer comes from the matcher trace (01 §6.1-C). It feeds the "measured failure analysis" criterion directly and is worth more than any styling.

Non-TTY (`| cat`, CI, `docker run` without `-t`) defaults to a **GitHub-markdown table produced by a ~10-line stdlib function, not by Rich.** Measured: when piped, Rich drops ANSI but keeps Unicode box rules and wraps at 80 columns. The markdown output can go straight into the README or the design note.

## 8. `auditgym play`: UX

`cmd.Cmd` subclass (stdlib), prompt `audit> `, with readline tab-completion of action names, table names and record IDs seen so far. Each command maps 1:1 to an env action, so a human uses exactly the same tool interface as a policy:

```
$ docker run --rm -it auditgym play --split dev --episode 0
╭─ episode dev/ep_00 · mode=train · step 0/100 ──────────────────────────────╮
│ Entity: Kestrel Trading LLP · period 2026-Q3 · currency KZT                 │
│ tables: ledger(412) invoices(97) bank(380) fx_rates(92) vendors(31)         │
╰─────────────────────────────────────────────────────────────────────────────╯
audit> help
  list <table> [cursor] [limit≤50]   get <id>   calc <expr>
  flag <type> <id> [<id>…] [--amount X]   unflag <n>   findings   submit   raw <json>   quit
audit> list invoices
  (Rich table of 50 rows, cursor=next shown)          step 1/100 · reward 0.00
audit> flag duplicate_invoice INV-0042 INV-0057
  ✓ recorded finding #1                                step 3/100
audit> submit
  F1 0.67 · P 1.00 · R 0.50 · return 0.33 · missed: fx_misposting(BNK-0211)   (reveal shown only in train mode)
  trajectory → runs/play-2026-11-19T1530/trajectories.jsonl
```

- `raw <json>` sends a literal action dict, which helps when debugging the schema.
- **Non-TTY play is a feature:** if stdin is not a TTY, read one command per line, echo it, and print plain text. Then `printf 'list invoices\nsubmit\n' | auditgym play …` is scriptable, is covered by a pytest test, and shows up in the demo.
- Why not Textual or questionary: Textual means 106 ms import, an async app model and Pilot-based tests, which costs at least a day. Arrow-key menus are worse than typing for entering record IDs. Either can be added later behind `play --tui` if time allows. That won't happen before the deadline.

## 9. Keeping `check` machine-readable

1. **No Rich in `check`'s main path.** Print one plain line per assertion to stdout: `PASS contract.reset_returns_json_dict 0.01s` / `FAIL reward.flag_all_negative 0.03s  expected<0 got 0.12`. Finish with a summary line `58/58 checks passed in 41.2s · env 0.1.0 · config 3f9a2c1 · python 3.11.18`.
2. **Exit codes:** 0 = all pass, 1 = any check failed, 2 = usage/internal error. argparse already exits 2 on bad flags, so the code only needs to make the internal-error path return 2 as well.
3. **`--json`** prints exactly one JSON document to stdout (`{"version":…, "config_hash":…, "checks":[{"id","status","seconds","detail"}], "passed":n, "failed":m}`). All human logs go to **stderr**, so `auditgym check --json | jq .failed` always works.
4. **Always write artifacts:** `runs/check/summary.json` and `runs/check/summary.md`, whatever the output mode.
5. **No spinners or progress bars**, or at most a per-section counter on stderr when `sys.stderr.isatty()`. Respect `NO_COLOR`. Never print ANSI when not a TTY (Rich does this by default and the plain path never emits it).
6. **Test it:** pytest runs `main(["check","--json","--fast"])` in-process and asserts the JSON validates and that piped output contains no `\x1b[`.

## 10. Rejected, with the one-line reason

- **termcn / Ink:** a great React TUI kit, but the wrong runtime. +232 MB, a TSX build, and it only renders the last frame without a TTY.
- **@clack/prompts:** the nicest prompts in JS, but `play` alone doesn't justify adding Node.
- **Commander.js:** a fine parser, but argparse does the same job with no second runtime.
- **oclif:** a framework for large plugin-based CLIs. The `oclif` generator pulls the AWS SDK. Overkill.
- **Typer:** good DX, but too much churn this month (pydantic now required, `click_type` removed) and 15.8 MB for what argparse already gives us.
- **cyclopts:** modern and clean, but a third API to learn with no benefit over argparse for 8 subcommands.
- **rich-click:** only useful if we picked Click. A nice option if the team switches to Click.
- **Textual:** the best Python TUI, but day-scale effort for zero points. Possible post-deadline `play --tui`.
- **questionary / InquirerPy:** menus are a poor fit for ID entry. InquirerPy has had no release since 2022.

## Sources

- termcn: https://www.termcn.dev/ · https://www.termcn.dev/llms.txt · https://github.com/shadcn-labs/termcn (Context7 `/shadcn-labs/termcn`)
- Ink CI/non-interactive behavior: https://github.com/vadimdemedes/ink/blob/master/readme.md (Context7 `/vadimdemedes/ink`)
- Clack CI/TTY behavior: https://github.com/bombshell-dev/clack/blob/main/packages/prompts/src/spinner.ts · task-log.ts (Context7 `/bombshell-dev/clack`)
- Commander: https://github.com/tj/commander.js · oclif: https://github.com/oclif/oclif
- Rich env vars and TTY detection: https://github.com/textualize/rich/blob/main/docs/source/console.rst (Context7 `/textualize/rich`) · Rich prompt/export: https://rich.readthedocs.io/en/stable/prompt.html
- Typer testing: https://typer.tiangolo.com/tutorial/testing · Typer releases (0.27–0.29.2): https://github.com/fastapi/typer/releases
- Click: https://github.com/pallets/click · cyclopts: https://github.com/BrianPugh/cyclopts · rich-click: https://github.com/ewels/rich-click · Textual: https://github.com/Textualize/textual · questionary: https://github.com/tmbo/questionary · InquirerPy: https://github.com/kazhala/InquirerPy
- VHS: https://github.com/charmbracelet/vhs (Context7 `/charmbracelet/vhs`) · asciinema: https://github.com/asciinema/asciinema · agg: https://docs.asciinema.org/manual/agg/usage/
- Versions and weights: https://pypi.org/pypi/<pkg>/json and `npm view` (2026-11-19). Node tarball: https://nodejs.org/dist/latest-v22.x/. Python image tags: https://hub.docker.com/_/python
- Case: `research/case.pdf` p.4 ("1 ГБ на решение", "Не тратьте основной срок на … UI"), p.5 ("Красивый UI … баллов не дают", Engineering criteria).
