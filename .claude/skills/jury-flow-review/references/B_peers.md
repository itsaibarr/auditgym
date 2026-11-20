# B. How peer eval harnesses design their CLI flow (for AuditGym)

Sources were checked on 2026-11-21 against the live docs and, where the docs say nothing about terminal output, the current source code (noted as "src").

## Comparison

| Tool | First command a newcomer runs | During the run | How results appear | What it suggests next | How you plug in your own agent or model |
|---|---|---|---|---|---|
| **Inspect AI** | `inspect eval simpleqa.py --model openai/...` | A Rich panel titled `task (N samples): model`, a progress bar, live metrics, and a footer with HTTP retries and tok/s (src) | A metrics table per scorer, total time, tokens per model, a clickable `Log: logs/…` link, and `WARNING: x of y samples had errors` | `inspect view`, which updates live as samples finish; `inspect eval-retry <log>` when a run is interrupted | `--model` provider string, a `solver=` (a `react()` agent), or extensions |
| **promptfoo** | `npx promptfoo@latest init --example getting-started` → `eval` → `view` | Progress bar | A grid with test variables as rows and `[provider] prompt` as columns. Cells read green `[PASS]`, red `[FAIL]` or `[ERROR]`. Below it: `Results: ✓ N passed (x%)`, failed, errors, duration (src) | `» View results: promptfoo view`, `» Create shareable URL: promptfoo share` | `providers:` in the YAML, `-r` on the command line, or custom Python/JS |
| **lm-eval-harness** | `lm_eval --model hf --model_args pretrained=… --tasks hellaswag` | Logs | A Markdown table: `Tasks/Version/Filter/n-shot/Metric/↑↓/Value/±/Stderr` (src) | `lm-eval ls tasks`, `--log_samples` | `local-completions` with a `base_url`, an LM subclass, or plugins |
| **OpenAI evals** | `oaieval gpt-3.5-turbo test-match` | Log lines | `Final report:` followed by `key: value` lines; JSONL logs in `/tmp/evallogs` (src) | None (it points to a third-party log viewer) | The `CompletionFn` protocol |
| **SWE-bench** | `swebench eval verified --gold -i sympy__sympy-20590 --run-id validate-gold` ("Test your installation") | A count of how many ran successfully or failed | A three-way split: resolved, unresolved, and errors (no report produced). Each instance gets a folder with `report.json`, `test_output.txt` and `patch.diff` | Run the full set, rerun with `--instance_ids`, use a fresh `--run_id` | A predictions JSONL with `instance_id`, `model_name_or_path`, `model_patch` |
| **Terminal-Bench / Harbor** | `harbor run -t hello-world/hello-world -a codex -m …` | Progress; streaming is available | A grid with `Resolved Trials`, `Accuracy`, then `Results written to …` (tb src) | `harbor view ./jobs`, `harbor upload` | `-a oracle` checks setup ("reward was 1"); custom agents via `--agent-import-path path.to:Agent` |
| **HELM** | `helm-run --run-entries mmlu:…,model=… --suite my-suite --max-eval-instances 10` | Log lines | `helm-summarize` writes JSON and LaTeX; `helm-server` serves Leaderboards, Models, Scenarios, Predictions | The three commands are run by hand in order | `model_deployments` config |
| **Gymnasium** | `gym.make` → `reset` → a loop of `step(action_space.sample())` | `render_mode="human"` | One line each defining observation, reward, `terminated` and `truncated` | `utils.play`, `check_env` | Implement the `Env` API |
| **τ²/τ³-bench** | `tau2 run --domain airline --agent-llm … --num-tasks 5` (`--dry-run` prints the plan) | Logs | Results in `data/simulations/`, browsed with `tau2 view` | `tau2 intro` gives an overview of domains, commands and examples; `tau2 play` lets you act as the agent or user | `--agent-llm`, the Agent Developer Guide, the Gym interface |
| **DeepEval** | `deepeval test run test_example.py` (pytest-style) | Per-metric spinners | A table with `Test case/Metric/Score/Status/Overall Success Rate`. Each score cell reads `0.8 (threshold=…, evaluation model=…, reason=…)`, then a `Pass Rate` line (src) | `» Run 'deepeval view'` (needs a Confident AI login) | Put your app's real output in `actual_output`; `@observe` for agents |
| **METR Vivaria** | A Docker Compose web app | A web UI showing traces | Run pages where you can tag and comment on trace points | — | Deprecated: METR tells new projects to use Inspect |

## Patterns worth copying (ranked by how much they help AuditGym)

1. **Present the oracle as a check that the install works, not as a competitor.** SWE-bench calls its `--gold` run "Test your installation" ([README](https://github.com/SWE-bench/SWE-bench)). Harbor says the oracle run should end with "reward was 1" and gives a troubleshooting list for when it doesn't ([create-a-task](https://docs.harborframework.com/tutorials/create-a-task.md)). **AuditGym moment:** the baseline table. Mark the oracle row as `install check: must be 1.00 ✓`, and present random and flag_all as "floor" rows and expert as the "bar to beat". Jurors then see the score range instead of five unexplained rows.

2. **Keep wrong answers apart from crashes.** promptfoo prints `[FAIL]` when an assertion fails and `[ERROR]` when a provider or code error happens ([table.ts](https://github.com/promptfoo/promptfoo/blob/main/src/table.ts)). SWE-bench splits results into resolved, unresolved and errors ([eval guide](https://www.swebench.com/SWE-bench/guides/evaluation/)). Inspect prints `WARNING: 3 of 20 samples had errors`. **AuditGym moment:** "run your own agent". An agent that crashes or times out should be counted as `errored`, not shown as reward 0.00.

3. **End each run with a short `»` block of commands, each labelled by what it does.** promptfoo prints `» View results: promptfoo view` and `» Create shareable URL: promptfoo share` ([summary.ts](https://github.com/promptfoo/promptfoo/blob/main/src/util/eval/summary.ts)). **AuditGym moment:** the copy-paste next steps. Lead each line with the user's goal ("Test YOUR agent:", "Compare two runs:"), show three at most, and move the rest into `auditgym intro`.

4. **On failure or interrupt, print the exact command to recover.** Inspect prints "Resume task with: `inspect eval-retry <log>`" ([results.py](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_display/core/results.py)). **AuditGym moment:** a truncated uv download, Ctrl-C partway through the tour, or a failing check. Print the one command that resumes or reruns only that stage.

5. **Show which direction is better and how uncertain each number is.** lm-eval adds a `↑`/`↓` column and `± stderr` to every metric ([utils.py make_table](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/utils.py)). promptfoo's `--repeat` reports "N unstable" groups with a 95% CI. **AuditGym moment:** the precision/recall/F1/reward table over 20 seeds. Add `↑` and `±` (or a CI), and print one line under the table defining each column, the way Gymnasium defines `terminated` and `truncated` ([basic usage](https://gymnasium.farama.org/introduction/basic_usage/)).

6. **Put the reason and the threshold next to every verdict.** DeepEval prints `0.8 (threshold=0.5, …, reason=…)` beside each score ([test_run.py](https://github.com/confident-ai/deepeval/blob/main/deepeval/test_run/test_run.py)). **AuditGym moment:** the scorecard per jury criterion. Each line should read `✓ criterion — evidence (147/147 checks) — why it matters — see <path>`.

7. **Always print where the artifacts went, and offer one way to open them.** Inspect prints a clickable `Log: logs/…` link and offers `inspect view bundle`, which builds a static viewer ([log viewer](https://inspect.aisi.org.uk/log-viewer.html)). Terminal-Bench prints `Results written to …`. **AuditGym moment:** the end of every stage, and the replay and compare commands, which should take that path as their argument. A static `report.html` is better than a server for jurors.

8. **Give the CLI a single command that maps out everything else.** τ-bench says "Run `tau2 intro` for an overview of available domains, commands, and examples" ([README](https://github.com/sierra-research/tau2-bench)). **AuditGym moment:** the user's complaint that it was hard to tell what each command is for. Add `auditgym intro`, a table of goal → command → when to use it → time and cost.

9. **Name the live progress after the stage it is running.** Inspect titles its panel `task (N samples): model` with live metrics, and the same panel becomes the final results view ([panel.py](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_display/core/panel.py)). **AuditGym moment:** the tour. Use one header per stage, such as `[2/5] Baselines — 5 policies × 20 hidden seeds`, with a running F1 instead of a scrolling log.

10. **Offer two ways to bring an agent: a file or code.** The lowest bar is SWE-bench's predictions JSONL with three fields. Harbor and tb take code via `--agent-import-path path.to.agent:YourAgent` ([tbench first steps](https://www.tbench.ai/docs/first-steps)). **AuditGym moment:** "run your own agent". Accept `--agent mymod:MyAgent` and also `--predictions flags.jsonl`, and document both in a single short schema block.

11. **Generate an example the user can edit.** `npx promptfoo@latest init --example getting-started` creates a folder with a config and a README ([getting started](https://www.promptfoo.dev/docs/getting-started/)). **AuditGym moment:** before "run your own agent". Add `auditgym init my-agent`, which writes an agent stub (a copy of `reconcile`), a README, and the exact command to run it.

12. **Let users preview a run without executing it.** `tau2 run … --dry-run` "prints the plan without making calls". **AuditGym moment:** "make your own hidden split" and "run your agent". Show the seeds, the episode count, where output goes and the expected time before anything runs.

13. **Make re-grading its own verb, separate from running.** Inspect has separate `eval`, `score`, `view` and `eval-retry` commands. Harbor's `regrade` "run[s] an updated verifier against recorded outputs without rerunning the agent" ([docs index](https://docs.harborframework.com/llms.txt)). **AuditGym moment:** replay and integrity. Explain them as "re-grade a recorded run, no agent needed".

14. **Group `--help` by purpose and give exit codes a meaning.** Terminal-Bench groups its flags into Dataset, Agent, Output, Logging and Concurrency panels ([runs.py](https://github.com/laude-institute/terminal-bench/blob/main/terminal_bench/cli/tb/runs.py)). promptfoo exits with `100` when tests fail ([CLI docs](https://www.promptfoo.dev/docs/usage/command-line/)). **AuditGym moment:** a juror running `--help`, and CI or a `curl | sh` caller checking whether the tour passed.

15. **Let users act as the agent themselves.** Peers do this with `tau2 play` and `gymnasium.utils.play` ([utils](https://gymnasium.farama.org/api/utils/)). AuditGym already has this. Present it as "Be the agent: audit one episode by hand (5 min)" so users understand the task before they write code.

## What not to copy

- **HELM's chain of three manual commands** (`helm-run` → `helm-summarize` → `helm-server`). AuditGym should summarize automatically and write the report in the same run.
- **Logins and cloud upsells.** DeepEval's `deepeval view` needs a Confident AI login, and promptfoo nudges users toward `share`/promptfoo.app. Jurors must stay local, with no accounts.
- **A huge first command.** lm-eval packs everything into a `--model_args k=v,k=v` string, and tau2 needs about 6 LLM flags. AuditGym's zero-argument tour is a strength; keep the advanced flags out of the first screen.
- **Logs with no viewer.** OpenAI evals writes only JSONL to `/tmp/evallogs` and points to an unsupported third-party viewer. Results need a path you can open.
- **Docker, GPU or Modal as a prerequisite.** SWE-bench, Harbor and Vivaria (a deprecated Docker Compose web app) all need them. The uv bootstrap is better; don't add infrastructure.
- **Celebratory filler.** DeepEval prints "Evaluation completed 🎉 … ❤️ 🏡". Keep summaries factual.
- **A long-running server as the only way to see results.** Prefer a static HTML file, as Inspect's `view bundle` and promptfoo's `-o html` do.
