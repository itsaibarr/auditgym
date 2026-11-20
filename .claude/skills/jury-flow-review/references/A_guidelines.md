# A. CLI guidelines and wait-feedback research, applied to the AuditGym tour

Fetched 2026-11-21. Quotes are verbatim unless marked (para.). Myers 1985 was OCR'd from the CMU scan. Harrison 2007/2010 are quoted from the authors' PDFs. Dickey's Medium post returned 403, so 12-Factor quotes are from a secondary summary (charlesreid1.com/wiki/12_Factor_CLI_Apps). The Primer CLI pages now redirect, so Primer quotes come from `github.com/primer/design/content/native/cli/*.mdx`.

## 1. Checkable rules

**R1. Print within 100 ms, before any network call.**
clig.dev: "Print something to the user in <100ms." "If you're making a network request, print something before you do it". https://clig.dev/#robustness-guidelines
→ The launcher prints "AuditGym tour · first run downloads Python (~1–3 min)" *before* uv starts. The silent download is the most likely "is anything happening?" moment.

**R2. Any wait over ~10 s needs percent-done or X of Y, an estimate and a way to stop.**
Nielsen: "10 seconds is about the limit for keeping the user's attention focused on the dialogue." "percent-done progress indicators should be used for operations taking more than about 10 seconds". Such operations also need "a clearly signposted way for the user to interrupt" (para.). https://www.nngroup.com/articles/response-times-3-important-limits/
→ The download (1–3 min), step 1 (30–40 s) and step 2 each need a counter and "~N s left". Show "Ctrl-C to stop" once.

**R3. Waits of 2–10 s get a looped indicator plus text.**
NN/g: looped animation "should be reserved for actions that take between 2-10 seconds". Include "a text explanation of the process, for example 'Updating address 3 of 50.'" https://www.nngroup.com/articles/progress-indicators/
→ Steps 3–4 use a spinner *with* a status line ("Replaying episode 7/20"), never a bare spinner.

**R4. Make X of Y the default.**
Evil Martians: "opt for the X of Y pattern whenever you're handling step-by-step processes and can measure the progress of each". "if they're static, this means there might be a hiccup somewhere." https://evilmartians.com/chronicles/cli-ux-best-practices-3-patterns-for-improving-progress-displays. NN/g: "consider showing the number of steps."
→ Use "Step 2 of 4", "checks 87/147" and "policy 3/5 · episode 12/20". All totals are known in advance.

**R5. A spinner is a last resort and must move only on real progress.**
Ubuntu: "Use a spinner as a last resort, if there is no more informative way of ensuring that the progress line is changing." https://discourse.ubuntu.com/t/progress-indication/20333. Evil Martians: a basic spinner "won't alert the user if the app gets stuck in a loop".
→ A Rich spinner animating on a timer over a captured pytest subprocess breaks this rule. Stream per-check results and tick on each one.

**R6. Update progress text every 0.5–2 s.**
Ubuntu: "Progress text should update every 0.5 to 2 seconds." When no determinate progress exists, show elapsed time as "(21s…)" (para.).
→ If no check or episode finishes for 2 s, tick an elapsed counter.

**R7. Show remaining time at the line end, and drop it or say "stalled" when unreliable.**
Ubuntu: "Update the estimate as frequently as other progress text". If the estimate becomes unreliable, replace it with a word such as "stalled" (para.). Costa: estimate from previous runs, as CI does (para.). https://lucasfcosta.com/2022/06/01/ux-patterns-cli-tools.html
→ The durations are known, so put "~35 s" in each step header.

**R8. A bar fills exactly once, with one aggregate bar for parallel work.**
Ubuntu: a bar should "fill, for a successful task, exactly once". Evil Martians: use "a single progress bar that reports the overall progress of multiple actions" and "If you're just dealing with a single task or a couple of steps in sequence, it might be better to skip the bars."
→ Step 2 gets one bar over 100 policy-episodes, not five resetting bars. The other steps use counters.

**R9. Put slow or variable stages first, with no stall near the end.**
Harrison et al. 2007 (22 participants, 990 comparisons): "participants perceived progress bars with pauses as taking longer"; "accelerating progress was strongly favored"; both effects are amplified "when located towards the end". "users are most willing to tolerate negative progress behavior … at the beginning". https://www.chrisharrison.net/index.php/Research/ProgressBars
→ The leak scan and scorecard rendering must not hang silently at the end. If the leak scan is slow, move it earlier or give it its own counter.

**R10. Split long runs into meaningful named steps.**
Atlassian: "break long-running tasks into a series of meaningful steps". https://www.atlassian.com/blog/it-teams/10-design-principles-for-delightful-clis
→ Headers state step, purpose and time: "Step 2 of 4 · Grading 5 baseline agents on 20 unseen audit cases (~60 s)".

**R11. Use a gerund while running and the past tense when done, then clear the live display.**
Ubuntu: start with a capitalised gerund and an ellipsis, such as "Building…" (para.). Evil Martians: "The log should switch to past simple or present perfect tense"; "clearing spinners and progress bars once an action is completed"; "green colors and checkmark icons" for success.
→ Each step collapses to one line: "✓ Step 1 of 4 · 147/147 checks passed (34 s)".

**R12. Give a reaction for every action.**
clig.dev: "If you change state, tell the user." Atlassian: "For every action a user performs, your CLI should provide an equal and appropriate reaction".
→ Every step ends with an outcome line, including the files written.

**R13. Color and icons reinforce meaning, never carry it.**
Primer: "Only use color to enhance meaning, not to communicate meaning." Icons: "✓ Success", "✗ Failure", "! Alert", "- Neutral". https://github.com/primer/design/blob/main/content/native/cli/foundations.mdx. clig.dev: "if everything is a different color, then the color means nothing". Heroku: yellow and red "are saved for errors and warning messages." https://devcenter.heroku.com/articles/cli-style-guide
→ Every scorecard cell says PASS, FAIL or WARN in text.

**R14. Turn off animation and color for non-TTY output and opt-outs.**
clig.dev: "If `stdout` is not an interactive terminal, don't display any animations". It also honors `NO_COLOR`. 12-Factor: respect `TERM=dumb`, `NO_COLOR` and `--no-color`.
→ Test `| tee`, `TERM=dumb` and the npx child process's TTY inheritance. Rich's fallback there is a likely cause of "tables looked like raw text".

**R15. Tables need a header and must survive narrow terminals.**
Primer: "Lists use tables to show information." "A header is used for context." Machine output uses "Tabs between columns" and "State is explicitly written, not implied from color". 12-Factor: "Not the fancy tables with borders."
→ The scorecard uses Criterion | Result | Evidence. Verify it at 80 columns, because Rich folds narrow tables into unaligned text.

**R16. Put the most important information last.**
clig.dev: "Put the most important information at the end of the output." "Display output on success, but keep it brief."
→ The final lines are the verdict ("6/6 jury criteria PASS") plus at most three next commands.

**R17. Suggest the next commands, each with its purpose.**
clig.dev: "Suggest commands the user should run." Atlassian: "suggesting the next best step users should take at the end of each command."
→ Each command gets a "what you'll see" line, such as "`auditgym replay 7`: watch one agent audit step by step".

**R18. Write for the reader, not the authors.**
clig.dev: "By default, don't output information that's only understandable by the creators of the software." Primer: "Use unambiguous language". Atlassian: instructions of "no more than 3 sentences (or around 50–75 characters)".
→ Gloss "paired checks", "hidden episodes" and "leak scan" once, for example "leak scan: proves the agent never saw the answer key".

**R19. Progress goes to stderr and results to stdout.**
Heroku: "Actions are displayed on stderr because they are out-of-band information on a running task." 12-Factor: "stdout is for output, stderr is for messaging."
→ With this split, `> results.txt` captures a clean scorecard.

**R20. On failure, rewrite the error for humans and show the logs.**
clig.dev: "Catch errors and rewrite them for humans." "if there is an error, make sure you print out the logs." Atlassian: "include suggestions for how to fix it."
→ Print the failing check names, a one-line cause, the log path and a rerun command.

## 2. Thresholds and numbers

| Number | Meaning | Source |
|---|---|---|
| <100 ms | First output must appear | clig.dev |
| 0.1 s / 1 s / 10 s | Instant / flow of thought kept / attention kept | Nielsen |
| <2 s | No progress needed | Ubuntu |
| 2–10 s | Looped indicator plus explanatory text | NN/g |
| ≥10 s | Percent-done or step count, estimate, interrupt | Nielsen, NN/g |
| 0.5–2 s | Interval between progress-text updates | Ubuntu |
| ×3 | Users with a progress indicator waited about 3 times longer | NN/g (UNL study) |
| 86.1% | Share of 48 subjects who "liked progress indicators" (p = 0.0006) | Myers CHI '85, https://www.cs.cmu.edu/~bam/papers/percentdoneCHI85.pdf |
| 11% | Drop in perceived duration with backwards-decelerating ribbed bars | Harrison/Yeo/Hudson CHI '10, https://www.chrisharrison.net/index.php/Research/ProgressBars2 |
| 50–75 chars | Maximum length of an instruction paragraph | Atlassian |
| 80 cols | Width that descriptions must fit | Heroku |

Myers: without an indicator, subjects "got bored with the screen and looked around the room". Novices are likely "to panic … and think that the computer has crashed". A juror on first run is a novice.

## 3. Anti-patterns to remove

1. Silence during the uv/Python download ("staring at a blinking cursor", Evil Martians).
2. A timer-driven spinner over a captured subprocess (R5).
3. Bars that reset for each policy, or stacked bars (R8).
4. A stall right before the end (R9).
5. Status shown only by color (R13).
6. Leftover gerunds such as "Running checks…" after completion, which Evil Martians calls "misleading" (R11).
7. Box tables that degrade in a non-TTY or narrow terminal (R14, R15).
8. Internal jargon in step names and command descriptions (R18).
9. A verdict buried above a long command list (R16).
10. Progress characters mixed into stdout (R19).
