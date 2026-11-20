# D. How jurors actually judge, and a review method that feels what they feel

## Bottom line
The council scored 9.5–10 because it reviewed a **text log**, and a log has no time in it. A 20-second frozen screen shows up as one line between two others. A juror sits through all 20 seconds, wondering whether it crashed. The fix is to review the **screen recording on a timeline**. Measure static intervals mechanically, make the reviewer answer the cognitive-walkthrough questions at every screen change, and score with an anchored rubric that **caps** the score when a defect appears. Each score must also cite timestamped evidence.

## 1. How judges evaluate
- **Time is short.** MLH's judging plan budgets ~4 min per project (2 min demo, 1 min questions and scoring, 1 min moving between tables) and elsewhere says judges "only have 3 minutes per team" ([MLH](https://guide.mlh.io/general-information/judging-and-submissions/judging-plan)). Our jurors get more than that (one command plus a 3-min video), but they judge under the same pressure: they rank many projects in one block.
- **The opening sets the score.** A year-of-judging write-up says: "That is where the score is set." A slow opening meant "I had spent a third of the slot without knowing what I was looking at." The same writer adds: "a live demo removes doubt, and doubt is what a judge is actually managing under time pressure." Judges compare each project "against the six they just saw" ([dev.to, Kurbaitaev](https://dev.to/kurbaitaev/what-judges-actually-score-notes-from-a-year-of-hackathon-judging-3p4l)).
- **Snap judgments stick.** Ratings from 50 ms views of a web page correlate highly with ratings from 500 ms views ([Lindgaard et al. 2006](https://tandfonline.com/doi/abs/10.1080/01449290500330448)). Silent clips under 30 s predicted end-of-semester teacher evaluations ([Ambady & Rosenthal 1993](https://www.apa.org/monitor/mar05/slices)). The **halo effect** spreads one strong trait over unrelated ones ([Thorndike 1920](https://www.britannica.com/science/halo-effect)). The reverse also holds: an early "is it broken?" moment taints contract, data and reward criteria the juror never inspects closely.
- **Order and fatigue matter.** Serial-position effects appear in jury ratings, and later performers tend to score higher ([Bruine de Bruin 2005, "Save the last dance"](https://research.tue.nl/en/publications/save-the-last-dance-for-me-unwanted-serial-position-effects-in-ju/)). We cannot control our slot, so the first 30 s must work for a tired juror.
- **Video.** Wistia found a steep drop in the first 10–15 s. Cutting the intro so content started within 10 s halved the early loss ([Wistia](https://wistia.com/learn/marketing/engagement-graph-learnings)). Early loss grows with video length ([Wistia](https://wistia.com/library/understanding-audience-retention)).

## 2. Psychology of waiting
- **Maister (1985)**: unoccupied time feels longer than occupied time, people want to get started (waits before the process begins feel worst), uncertain waits feel longer than known finite ones, unexplained waits feel longer than explained ones, and anxiety makes waits feel longer ([summary](https://www.queue-it.com/blog/psychology-of-queuing/); [original](https://davidmaister.com/?p=747)). Our first-run download and 30–40 s test suite hit all of these at once: it comes before any value, has no stated end, gives no reason, and a juror who fears "it broke on my machine" is anxious.
- **Nielsen's limits**: 0.1 s feels instantaneous, 1 s keeps the user's flow of thought, and 10 s is "about the limit for keeping the user's attention focused." Past 10 s, use a percent-done indicator. When the total is unknown, show what is being processed (for example, item names), and treat a generic spinner as a last resort ([NN/g](https://www.nngroup.com/articles/response-times-3-important-limits/)).
- **CLI-specific** ([clig.dev](https://clig.dev/)): "Print something to the user in <100ms." "If your program displays no output for a while, it will look broken." "Responsive is more important than fast." The guide also recommends suggesting the next commands to run and explaining "what has just happened."
- **Perceived duration can be engineered.** How a progress bar behaves changes how long the same wait feels, and pauses early in the bar are tolerated better than pauses at the end ([Harrison et al.](https://www.chrisharrison.net/index.php/Research/ProgressBars); [figlab 2010](https://www.figlab.com/research/2010/faster-progress-bars)).

## 3. Usability methods for a recorded CLI session
- **Cognitive walkthrough** ([Wharton et al. 1994, via Wikipedia](https://en.wikipedia.org/wiki/Cognitive_walkthrough); [Washington](https://faculty.washington.edu/ajko/books/design-methods/analytical)). Ask four questions at every step: (Q1) will the user try to achieve the right effect, (Q2) will they notice the correct action is available, (Q3) will they connect the action with the effect, and (Q4) will they see that progress is being made? Any "no" is a failure. For a passive watcher, Q4 becomes "can I tell it's working and how long is left?", and Q2/Q3 become "do I know what each printed command is for?" That second gap is exactly what the real user hit.
- **Nielsen heuristics** ([NN/g](https://nngroup.com/articles/ten-usability-heuristics/)). For CLIs, three matter most. *H1 Visibility of system status*: NN/g notes CLIs "do not display the current state of the system, nor do they give immediate feedback" by default ([NN/g](https://nngroup.com/articles/visibility-system-status)). *H2 Match with the real world*: say "checking the grader can't be gamed", not "pytest -k reward". *H6 Recognition over recall*: next-step commands should come with what each one does.
- **Think-aloud.** Concurrent think-aloud finds more problems. Video-cued retrospective think-aloud adds explanation, but recall decays with delay ([van den Haak et al. / Hertzum review, RUC](https://rucforsk.ruc.dk/ws/files/105748016/Concurrent_or_Retrospective_Thinking_Aloud_in_Usability.pdf)). Cheapest real signal: have one human who has never seen the project watch the recording and say out loud what they think is happening. Write down timestamps.
- **Time to first value** ([Instruqt TTHW](https://instruqt.com/glossary/time-to-hello-world)). Measure seconds from Enter to the first moment the juror sees *our* value (the first results table), and separately the time to the "wow" (the scorecard).
- **Second-by-second analysis.** Run `ffmpeg -i rec.mp4 -vf freezedetect=n=-60dB:d=2 -f null -` to list every interval of 2 s or more where the frame did not change. Every one of those intervals must be explained by text that was already on screen. Sample one frame per second for the reviewer, never just the final log.

## 4. Why AI reviewers inflate, and the counter-design
- **The main cause is modality.** The model read output, not elapsed time, so dead air and the "did it hang?" feeling cannot be seen in what it reviewed.
- **Sycophancy.** RLHF models drift toward the user's stated views ([Sharma et al. 2023](https://arxiv.org/abs/2310.13548)). A council briefed with "this is our submission" leans positive.
- **Self-enhancement and verbosity bias** ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)). Models favor output from their own model family and favor longer text. Our flow's output was largely written with the same models.
- **Skewed rating distributions, anchoring across attributes, familiarity bias** ([Stureborg et al. 2024](https://arxiv.org/abs/2405.01724)). Evidence on raw leniency is mixed (some studies find LLM judges harsher on surface errors), so the reliable fix is structural, not "be stricter."
- **Counter-measures.** Use behaviourally anchored levels, because anchors plus frame-of-reference training improve rating accuracy ([BARS/FOR overview](https://sk.sagepub.com/ency/edvol/embed/organizationalpsychology/chpt/frameofreference-training)). Require extractive evidence for every score ([RULERS, arXiv 2601.08654](https://www.arxiv.org/pdf/2601.08654)). Use binary per-item checks rather than one holistic number ([arXiv 2604.06996](https://arxiv.org/pdf/2604.06996.pdf)). Run the penalty checklist **before** a score is chosen, and let it set a cap that a score cannot exceed. Brief the reviewer as a neutral juror, never as "our project." Calibrate once against the real first-time user's reactions: the rubric must score that recording at 6 or below, or it is too lenient.

---

## Rubric: Jury-flow review from a screen recording (0–10)

**Inputs.** The raw recording (not the log). The `freezedetect` output. Frames sampled at 1 fps. The reviewer persona: an ML/fintech juror who has never seen the repo, is on project 9 of 15, and has no README open.

**Step 1: Evidence log (required; a score without it is void).** Write one row per screen change and per static interval of 2 s or more:
`[mm:ss–mm:ss] SCREEN: <literal text/visual on screen> | JUROR THINKS: "<first-person sentence>" | FEELS: confident / curious / unsure / anxious / bored | FAILS: CW-Q1..Q4, H1/H2/H6 or none`
Also record: **T_first_output** (Enter → first character), **T_first_value** (→ first results table), **T_wow** (→ scorecard), the count of **confusion events** (a row where the juror cannot answer "what is happening / is it working / what is this for"), and total **dead air** (static seconds not covered by on-screen explanation).

**Step 2: Penalty checklist and caps.** The final score is the lowest cap that applies.

| Condition (from the evidence log) | Cap |
|---|---|
| T_first_output > 1 s (blank terminal after Enter) | 8 |
| Any static interval > 5 s with no stage label/reason already on screen | 7 |
| Any static interval > 10 s without a percent/count/ETA or live per-item output | 6 |
| Download/setup wait whose duration or purpose is not stated *before* it starts | 7 |
| T_first_value > 60 s, or > 30 s with no "what you'll see next" line | 7 |
| A command or next-step suggestion shown with no one-line purpose | 8 (each additional one: −0.5) |
| Jargon with no plain gloss on a juror-facing line (e.g. raw pytest IDs, internal names) | 8 |
| 2 or more confusion events | 6 |
| 4 or more confusion events, or the juror would plausibly press Ctrl-C | 4 |
| Visible traceback/warning/"FAILED" not explained as expected on the same screen | 5 |
| Scorecard numbers without a stated meaning (scale, what good is, tie to the 5 criteria) | 7 |
| Ending without a clear "you're done; here's what to try next" | 8 |
| Demo video: no project claim in the first 10 s | 7 |

**Step 3: Anchored level.** Pick the highest level whose description fully fits, then apply the caps.

- **10**: Zero confusion events. Every wait of 2 s or more is labeled with its purpose and length, and every wait of 10 s or more shows live progress. Output starts within 1 s and the first value appears within 30 s. Every command shown has its purpose stated. The juror could explain the project to a colleague right after watching. A naive human viewer confirmed this.
- **8**: At most one brief moment of uncertainty (under 5 s), resolved by the next line. Waits are explained but sometimes without an ETA. Everything else is at the 10 level.
- **6**: The juror gets there but has to work for it: one or two confusion events, or one unexplained wait of 5–10 s, or a few unexplained commands. The juror's trust holds, but the impression is "rough."
- **4**: The juror spends noticeable time wondering whether it is broken, or cannot say what the scorecard means. Value arrives late. A halo of doubt spreads to the technical criteria.
- **2**: The juror probably interrupts the run or forms a "doesn't work" impression before the first value.
- **0**: The run fails, or nothing interpretable appears.

**Step 4: Output.** Give the score, the binding cap (or none), the three worst evidence rows, and one concrete fix per row. A reviewer may not give a score above 8 without quoting at least one row where the juror FEELS "confident" during the longest wait.
