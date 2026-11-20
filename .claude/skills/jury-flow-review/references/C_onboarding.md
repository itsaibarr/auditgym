# C. Onboarding / long-running terminal flows: patterns for AuditGym

The user's four complaints map to these patterns: you can't tell anything is happening (1, 2, 3, 4, 5), tables look like raw text (13), the wording is unclear (6, 7, 9, 14), and nobody knows what each command does (10, 11, 12).

## Patterns

**1. Print within 100 ms and name the phase and its cost.** Sources: clig.dev ("Responsive is more important than fast. Print something within 100ms"), https://clig.dev/ ; Vercel CLI guidelines ("Spinners name their phase"), https://vercel.com/design/cli
```
AuditGym first run: setting up Python 3.11 and dependencies
This happens once and takes 1-3 min. Later runs start in about 2 s.
```
Where it fits: the first line the npx/curl launcher prints, before uv starts downloading anything.

**2. Say once when a step runs long ("Still <verb>ing").** Source: Vercel guidelines: "Still <verb>ing. Most <operations> finish within <duration>." They never use "This might take a while." https://vercel.com/design/cli
```
⠼ Installing dependencies  1m 34s
  Still installing. Most first runs finish within 3 min.
```
Where it fits: uv bootstrap after about 90 s, and the pytest step if it passes 45 s.

**3. A step checklist with an n/m counter and a frozen duration per step.** Source: BuildKit tty mode prints the header `"[+] %s %.1fs (%d/%d) %s"` and adds `FINISHED` at the end. Each step is one line with `" => "` and a right-aligned `" %3.1fs"` timer. https://github.com/moby/buildkit/blob/master/util/progress/progressui/display.go
```
[+] Tour 18.2s (2/4)
 ✓ 1 Self-test (142 tests)               12.1s
 ⠙ 2 Policies on hidden episodes  9/24    6.1s
 · 3 Paired checks   · 4 Replay
```
Where it fits: the 4-step tour. It is the main fix for "can't tell if anything is happening".

**4. Show a live tail of the active step's output, hide it on success, and dump it on failure.** Source: BuildKit shows faint `" => => # %s"` lines only for active steps and prints the full log in `printErrorLogs` on error. clig.dev: "Hide logs behind progress bars when things succeed, but print the logs if an error occurs."
```
 ⠙ 1 Self-test  87/142                    7.9s
     test_ledger.py::test_cutoff_split PASSED
     test_ledger.py::test_fx_rounding PASSED
```
Where it fits: the 30 s pytest step. The test names scrolling past prove the step is working, and the step folds to one ✓ line when it finishes.

**5. Use a determinate count only when the total is real.** Sources: Playwright line reporter `[23/124]` and list reporter `✓ title (438ms)`, https://playwright.dev/docs/test-reporters ; cargo `Building [=====>   ] 12/40`; Vercel: "Progress bars need a real total. Never invent percentages."
```
 ⠙ 2 Policies on hidden episodes  ━━━━━━━╸━━━  17/24  oracle, naive, random
```
Where it fits: the pytest count (get the total first with `pytest --co -q`) and episodes x policies in step 2. The uv download stays a spinner because AuditGym has no total for it.

**6. Collapse a finished phase into a short receipt with a count and a time.** Source: uv prints `Resolved 43 packages in 11ms`, `Prepared 2 packages in 627ms`, `Installed 43 packages in 208ms`. https://docs.astral.sh/uv/ and the PyPI README examples.
```
✓ Python 3.11 ready            (cached in ~/.cache/auditgym)
✓ Installed 38 packages in 41s
```
Where it fits: the setup transcript ends with these 2 lines, and the uv noise above them goes away.

**7. Right-align a short verb label in a fixed gutter.** Sources: cargo prints the status word bold green, right-justified in 12 columns (`   Compiling foo`, `    Finished dev [...] in 2.3s`), https://doc.rust-lang.org/stable/nightly-rustc/src/cargo/core/shell.rs.html ; create-astro uses `align(label(text),'end',7)` plus a cyan `◼` info gutter, https://github.com/withastro/astro/blob/main/packages/create-astro/src/messages.ts
```
     Testing  142 unit + integration tests
   Simulating  24 hidden episodes x 3 policies
     Checking  paired variants (should flip / should hold)
   Replaying  episode ep_07 step by step
```
Where it fits: every tour step header. A verb in plain English beats jargon like "paired checks".

**8. Lead with the outcome, then label/value rows.** Sources: Vercel deploy ends with `✓ Ready in 41s`, then rows with a 16-character label column, values from column 18, and no colons. https://vercel.com/design/cli ; `supabase start` groups its URLs and keys under headings, https://supabase.com/docs/guides/local-development/cli/getting-started
```
✓ Tour complete in 41s. Grader separates good from bad policies.
  Report          runs/2026-11-21/scorecard.json
  Replay          runs/2026-11-21/ep_07.html
```
Where it fits: directly above the scorecard table, so the verdict is the first thing read.

**9. Add a single summary line of counts under the details.** Sources: Terraform `Plan: 1 to add, 0 to change, 0 to destroy.` and `Apply complete! Resources: 1 added...`, https://developer.hashicorp.com/terraform/tutorials/aws-get-started/aws-create ; pytest `== 1 failed, 1 passed, 1 skipped in 0.12s ==`, https://docs.pytest.org/en/stable/how-to/output.html
```
24 episodes · 3 policies · 6/6 paired checks held · 0 grader errors
```
Where it fits: the scorecard caption. Someone who never reads the table still gets the result.

**10. In next steps, put the reason first and the command on its own line.** Source: Vercel: "Each states its benefit, then the command on its own line." Cyan marks runnable commands. create-astro puts a `next` badge on its block ("Run `npm run dev` to start the dev server. q + ENTER to stop.").
```
 next  Grade your own agent against the hidden episodes:
           auditgym eval --agent path/to/agent.py
       See exactly what one episode looked like, step by step:
           auditgym replay ep_07
```
Where it fits: the final command list. Reason first, then the command, addresses "didn't understand what each command does and when to use it".

**11. Show only the next steps that apply to the current state, recovery first.** Sources: create-t3-app wraps each next step in a condition, for example `cd` only if projectName != "." and `db:push` only if an ORM is in use, https://github.com/t3-oss/create-t3-app/blob/main/cli/src/helpers/logNextSteps.ts ; Vercel: "required recovery first", and at most 2 recovery options.
```
✗ 1 paired check failed (fx_rounding should flip, held).
  See the two episodes side by side:
      auditgym replay --pair fx_rounding
```
Where it fits: the end of the tour when something failed. Don't show the happy-path list in that case.

**12. End the installer with what was installed, where, and how to run it again.** Source: the Bun installer prints `bun was installed successfully to ~/.bun/bin/bun`, then `Added "~/.bun/bin" to $PATH in "~/.zshrc"`, then `To get started, run:` with the commands indented in bold. It uses `curl --progress-bar`. https://bun.sh/install
```
AuditGym is installed in ~/.cache/auditgym (412 MB). Remove it with: auditgym uninstall
Run the tour again (about 40 s, no download):
    npx auditgym tour
```
Where it fits: the first-run setup step, just before the tour starts. It also tells the juror the 1-3 min wait won't happen again.

**13. Use borderless tables, and give non-TTY output a real fallback.** Sources: Vercel: "Human tables have no borders". Non-TTY output keeps the header, uses tab-separated columns, never truncates, and never prints a fake-pretty table. Rich detects a terminal through `isatty`, overridden by `TTY_COMPATIBLE=1/0` and `FORCE_COLOR`. https://rich.readthedocs.io/en/stable/console.html
```
  Policy     Score   Caught   False flags
 ─────────────────────────────────────────
  oracle     0.97    23/24    0
  naive      0.41    10/24    7
```
Where it fits: the scorecard. The likely cause of "tables looked like raw text" is that the Node/sh launcher pipes the child's stdout. Rich then sees no TTY, drops color, and turns Live/Progress off. Fix: spawn with `stdio: 'inherit'`. If AuditGym must pipe, pass `FORCE_COLOR=1` and `COLUMNS=<parent width>` when `process.stdout.isTTY` is true.

**14. Use a small fixed set of glyphs that never carries meaning alone.** Source: the Vercel glyph table (`✓` one primary result, `!` warning, `✗` fatal), with no emoji, no "Success!" banners, and an ASCII fallback with words. Homebrew uses `==>` section headers and puts `==> Caveats` last, https://docs.brew.sh
```
✓ held   fx_rounding/base vs fx_rounding/flip   score 0.91 -> 0.12
! slow   ep_19 took 4.2s (budget 2s)
```
Where it fits: paired-check rows and replay annotations. Write "held/flipped" as words, because a red/green color alone disappears in a non-TTY.

## Rich implementation notes (top 5)

**A. Step checklist with frozen durations (3, 5, 7).**
```python
steps = Progress(
    SpinnerColumn(finished_text="[green]✓"),
    TextColumn("{task.description}"),
    BarColumn(bar_width=20), MofNCompleteColumn(),
    TimeElapsedColumn(), console=console)
ids = [steps.add_task(label, total=n, start=False) for label, n in plan]
steps.start_task(ids[0]); steps.update(ids[0], advance=1)
steps.update(ids[0], completed=n)   # finished: spinner -> ✓, time freezes
```
`TimeElapsedColumn` uses `finished_time` once a task completes, so every row keeps its own duration. Pass `total=None` for indeterminate rows, which renders a pulsing bar. You can't mix columns within one `Progress`. If you need two layouts, put several `Progress` instances inside one `Live(Group(...))`. Rich's `examples/dynamic_progress.py` does this with `Group(Panel(Group(a, b, c)), overall)`: https://github.com/Textualize/rich/blob/master/examples/dynamic_progress.py

**B. Live log tail for pytest (4).** Run `subprocess.Popen([... "pytest", "-v", "--no-header", "-rN"], stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`. Push each line into `collections.deque(maxlen=4)`. Count the lines that end in `PASSED|FAILED|ERROR` to advance the task. Render `Live(get_renderable=lambda: Group(steps, Padding(Text("\n".join(tail), style="dim"), (0, 0, 0, 5))), refresh_per_second=8, transient=False)`. On success, clear the tail. On failure, `console.print` the whole buffer after `live.stop()`. Use `live.console.print()` (or `progress.console.print()`) to write permanent lines above the live area.

**C. Indeterminate waits with escalation (1, 2).** Wrap the wait in `with console.status("Installing dependencies…", spinner="dots") as st:`. From the polling loop, call `st.update("Installing dependencies… still running. Most first runs finish within 3 min.")` once elapsed passes 90 s. The uv bootstrap runs before Python exists, so this belongs in the sh/Node launcher. Don't pass `--quiet` to uv there: uv draws its own download bars on a TTY. Print the phase line from pattern 1 first.

**D. Receipts, gutter labels, outcome block (6, 7, 8).** Build a `Table.grid(padding=(0, 2))` with `add_column(justify="right", style="bold green", width=12)` for the verb and `add_column()` for the text. Rows look like `grid.add_row("Installed", "38 packages in 41s")`. For the outcome block use the same grid with a left label column `width=16, style="bold"`. Use `Rule(style="dim")` between phases instead of more `Panel`s.

**E. Scorecard and TTY fallback (9, 13).** Build the scorecard as `Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold", caption="24 episodes · 3 policies · 6/6 paired held", caption_justify="left")`, with `justify="right"` on numeric columns. Before the Live display starts, branch on `console.is_terminal` / `console.is_interactive`. When the console is not a terminal, print one plain line per finished step and tab-separated tables, and skip Live entirely. Rich also accepts `safe_box=True` for ASCII boxes and honours `NO_COLOR`, `FORCE_COLOR`, `TTY_COMPATIBLE`, and `TTY_INTERACTIVE`. Send progress to `Console(stderr=True)` and keep stdout for results, as clig.dev and Vercel both prescribe.

Charm reference: `gum spin --title "..." --show-output -- cmd` is the shell-launcher counterpart of `console.status`, if the launcher could add a dependency. It can't (zero-dep bootstrap), so use printf with a spinner loop instead. https://github.com/charmbracelet/gum

Notes: the cargo 12-column justification is confirmed from forks (cargo-equip, scarb-ui) and the Shell API. Homebrew, Jest, Stripe CLI, and rustup output were not re-fetched, and no pattern depends on them.
