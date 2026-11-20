---
name: jury-flow-review
description: Review AuditGym's jury experience the way a real first-time juror lives it, from screen recordings on a timeline instead of from final text output. Use when scoring or iterating on the one-command tour, the CLI wording, the demo video or any "is it clear / does it feel broken / do I know what to do next" question, and when acting as a UX-council reviewer for the jury flow.
---

# Jury-flow review (from recordings, not logs)

Why this exists: reviewers who read the tour's final text scored it 9.5-10/10, and the first real user still could not tell
whether anything was happening during a 25 s wait, and did not know what each command was for. A log has no time in it.
A juror sits through every second. So the evidence is the **recording**, sampled on a timeline, plus a mechanical list of
the stretches where the screen did not visibly change.

## 1. Make the evidence (the orchestrator does this; reviewers only read it)

1. Record both flows with VHS (1920x1080, the same font and theme as the demo):
   - `submission/demo.tape` (warm cache, the 3-minute demo video jurors watch);
   - `scripts/cold.tape` from this skill (empty HOME, the README one-liner, nothing cached: the real first run).
2. Run `scripts/video_kit.sh <video.mp4> <outdir> 2 3`. It writes `frames/tNNNNs.png` (one frame every 2 s, named by the
   second it shows) and `freezes.txt` with two lists: FROZEN (nothing changed) and NEAR-STATIC (at most a spinner or a
   timer changed: the eye sees no progress).
3. Hand reviewers: the frame folder, `freezes.txt`, the tape (to know what was typed), and `references/` (research).

## 2. Reviewer brief (paste into each reviewer)

Persona: a juror (ML or fintech background) on project 9 of 15, tired, no README open, never saw this repo. You watch the
recording once, in order. You are **not** on the team; you are not asked to be nice. Read the frames in time order
(`t0000s.png`, `t0002s.png`, ...); open every frame inside a NEAR-STATIC stretch, not just its ends.

Step 1, evidence log (required; a score without it is void). One row per screen change and per static stretch >= 2 s:

`[mm:ss-mm:ss] SCREEN: <literal on-screen text> | JUROR THINKS: "<first-person sentence>" | FEELS: confident/curious/unsure/anxious/bored | FAILS: CW-Q1..Q4, H1/H2/H6 or none`

- Cognitive walkthrough at each step: Q1 will the juror try to achieve the right effect; Q2 will they notice the right
  action is available; Q3 will they connect the action to the effect; Q4 will they see that progress is being made
  ("can I tell it is working and how long is left?").
- Heuristics: H1 visibility of system status, H2 match with the real world (plain words, no internal names),
  H6 recognition over recall (every shown command says what it is for).
- Record T_first_output (Enter to first character), T_first_value (Enter to the first results table), T_wow (Enter to the
  scorecard), the number of confusion events, and the seconds of dead air that nothing on screen explains.

Step 2, penalty caps (the score cannot exceed the lowest cap that applies; check every row and say yes/no):

| Condition | Cap |
|---|---|
| Blank terminal > 1 s after Enter with no explanation | 8 |
| Any static stretch > 5 s with no stage label or reason on screen | 7 |
| Any stretch > 10 s without a count, percent, ETA or live per-item text | 6 |
| A setup/download wait whose purpose or length is not stated before it starts | 7 |
| T_first_value > 60 s, or > 30 s with no "what you will see next" | 7 |
| A command or next step shown without a one-line purpose | 8 (-0.5 per extra one) |
| Jargon with no plain explanation on a juror-facing line | 8 |
| >= 2 confusion events | 6 |
| >= 4 confusion events, or a juror would plausibly press Ctrl-C | 4 |
| A traceback, warning or FAILED not explained as expected on the same screen | 5 |
| Scorecard numbers with no stated meaning (scale, what is good, link to the 5 criteria) | 7 |
| Ends without a clear "you are done; here is what to try next" | 8 |
| Demo video makes no claim about the project in its first 10 s | 7 |

Step 3, anchored level (highest that fully fits, then apply the caps):
- **10**: zero confusion events; every wait >= 2 s states purpose and length, every wait >= 10 s shows live progress;
  output < 1 s, first value < 30 s; every command has a purpose; the juror could explain the project to a colleague.
- **8**: at most one brief (< 5 s) moment of uncertainty that the next line resolves; waits explained, sometimes no ETA.
- **6**: gets through with effort; 1-2 confusion events or one unexplained 5-10 s wait or several purpose-less commands.
- **4**: noticeable "is it broken?" time, or cannot say what the scorecard means; doubt spills onto technical criteria.
- **2**: would likely interrupt, or decides "it does not work" before the first value. **0**: fails or incomprehensible.

Step 4, output: score; the cap that bound it (or none); the three worst evidence rows with one concrete fix each. A score
above 8 is allowed only if you quote the row from the **longest** wait in which the juror FEELS confident, and say why.

## 3. Council protocol

- Three reviewers, same evidence, independent: (a) first-time juror (persona above), (b) jury replica (also uses
  `../genesis-jury/SKILL.md`: does the flow make each of the 5 criteria's evidence obvious?), (c) CLI-UX expert (checks the
  20 rules in `references/A_guidelines.md` and the patterns in `references/B_peers.md` / `C_onboarding.md`, with timestamps).
- Pass = all three >= 9 in the same round, each with the evidence log. Fix, re-record, re-run the kit, new round.
- Calibration: the rubric must score the pre-fix cold recording (25 s near-static test suite) at <= 6. If a reviewer
  scores it higher, the brief is too lenient; tighten before trusting any 9.
- Reviewers write only to their own scratch folder, never edit the repo, `~/.claude` or settings, and skip-and-report
  anything that would need an approval.

## References (research, 2026-11-21)
- `references/A_guidelines.md`: clig.dev, NN/g response-time limits and progress indicators, Evil Martians, Ubuntu progress
  guidelines, Primer, Heroku, Atlassian; Myers 1985 and Harrison 2007/2010 on perceived waits. 20 checkable rules.
- `references/B_peers.md`: how Inspect AI, promptfoo, lm-eval, SWE-bench, Harbor/Terminal-Bench, HELM, tau2, DeepEval
  design their first run, live progress, results and next steps.
- `references/C_onboarding.md`: uv, cargo, BuildKit, create-astro/t3, Vercel, Playwright, pytest patterns; Rich APIs.
- `references/D_judging.md`: how judges decide (first 30 s, halo), Maister's psychology of waiting, cognitive walkthrough,
  why AI reviewers inflate scores; the source of the rubric above.
