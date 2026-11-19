# 01 — Open-source references for the Audit Gym RL environment

Date: 2026-11-19. Scope: the Genesis Hackathon FinTech case (`research/case.pdf`): a seeded fictional-company generator with hidden injected errors, a Gymnasium-style JSON `reset/step` contract, a grader that runs outside the agent process, the terminal reward `clip(2*F1 - 1 - 0.002*steps, -1, 1)`, 4 baseline adapters, and an offline Docker check. The jury runs its own policies against our environment.

Star counts, licenses and last-push dates come from the GitHub REST API (`gh api repos/<owner>/<repo>`) on 2026-11-19. Every claim cites a URL. Where GitHub reports `NOASSERTION`, the LICENSE file was read directly.

---

## 1. Summary table

| # | Project | URL | License | Stars / last push | What to borrow |
|---|---|---|---|---|---|
| 1 | **Gymnasium** | https://github.com/Farama-Foundation/Gymnasium | MIT | 12.8k / 2026-11-12 (v1.4.0, 2026-10-05) | The `reset(seed, options)->(obs, info)` / `step->5-tuple` semantics, plus the checks in `env_checker` (seed determinism, step determinism, no reused buffers). Copy the checks; don't take the dependency. |
| 2 | **OpenEnv** (Meta/HF, now `openenv/OpenEnv`) | https://github.com/openenv/OpenEnv | BSD-3-Clause | 2.9k / 2026-11-19 | Client/server split; `/schema` endpoint; **simulation vs production mode**, where production mode hides rewards and step control; `openenv validate` with severity levels and exit codes. |
| 3 | **Prime Intellect verifiers** | https://github.com/PrimeIntellect-ai/verifiers | MIT | 4.8k / 2026-11-19 | `vf-validate --only-gold / --only-noop` (oracle check and do-nothing check); isolated grading sandbox; `stage_verifier` for private assets; `Task.rescore(trace)`. |
| 4 | **Nous Atropos** | https://github.com/NousResearch/atropos | MIT | 1.4k / **archived** | `process` subcommand writes rollouts as JSONL plus a static HTML viewer. Archived, so pattern only. |
| 5 | **OpenPipe ART** | https://github.com/OpenPipe/ART | Apache-2.0 | 10.9k / 2026-11-19 | Shows the trainer-side `Trajectory` + reward shape. Its RULER LLM judge is the opposite of what we need. |
| 6 | **TRL** (`environment_factory`) | https://github.com/huggingface/trl | Apache-2.0 | 19.8k / 2026-11-19 | The minimal "env as a class whose public methods are the tools" contract. A cheap shim that shows our env is trainable. |
| 7 | **reasoning-gym** | https://github.com/open-thought/reasoning-gym | Apache-2.0 | 1.6k / 2026-11-09 | Procedural generator API: `create_dataset(name, size, seed, **config)` plus `score_answer(answer, entry)`. |
| 8 | **TextArena** | https://github.com/TextArena/TextArena | MIT | 440 / 2026-10-05 | Simplest possible policy contract: an agent is a callable `str -> str`. |
| 9 | **AgentGym** | https://github.com/WooooDyy/AgentGym | MIT | 862 / 2026-05-30 | Env as an HTTP service (`/createEnv`, `/observation`, `/available_actions`, `/step`, `/reset`), plus a trajectory replay viewer. |
| 10 | **τ-bench (v1)** | https://github.com/sierra-research/tau-bench | MIT | 1.5k / 2026-03-18 | `run.py --agent-strategy … --task-ids …`; ships historical trajectories; auto error identification. |
| 11 | **τ²/τ³-bench** | https://github.com/sierra-research/tau2-bench | MIT | 2.5k / 2026-11-10 | Gym wrapper that accepts **JSON tool-call actions**; `info["tools"]`; agent registry `--agent`; reward = product of components; `tau2 view`; `--dry-run`. |
| 12 | **BrowserGym** | https://github.com/ServiceNow/BrowserGym | Apache-2.0 (LICENSE file) | 1.4k / 2026-11-18 | Benchmarks as Gymnasium-registered IDs (`gym.make("browsergym/…")`); task = `AbstractBrowserTask`. |
| 13 | **SWE-Gym** | https://github.com/SWE-Gym/SWE-Gym | Apache-2.0 | 764 / 2026-11-01 | Cautionary example: a git-history shortcut leaked the answer (see SWE-bench #465). |
| 14 | **SWE-bench** harness | https://github.com/SWE-bench/SWE-bench | MIT | 6.2k / 2026-09-18 | `swebench eval … --gold` (oracle) and `swebench report <run_id>` (re-grade saved logs). Predictions file is separate from the grader. |
| 15 | **ToolSandbox** (Apple) | https://github.com/apple-aiml-research/ToolSandbox | Apple custom license (read before copying) | 294 / 2026-09-11 | `--agent`/`--user` CLI enums; a `Cli` human agent for manual play; `result_summary.json` per run. |
| 16 | **Inspect AI** (UK AISI) | https://github.com/UKGovernmentBEIS/inspect_ai | MIT | 3.3k / 2026-11-19 | `inspect eval task --solver file.py@name -S k=v`; `mockllm/model` for offline runs; `--no-score` and scoring as separate steps; `inspect view`; `--json` machine-readable output. |
| 17 | **lm-evaluation-harness** | https://github.com/EleutherAI/lm-evaluation-harness | MIT | 14.5k / 2026-11-04 | `--model local-chat-completions` (any OpenAI-compatible endpoint); `--log_samples --output_path`. |
| 18 | **Harbor** (Terminal-Bench) | https://github.com/harbor-framework/harbor | Apache-2.0 | 7.2k / 2026-11-19 | `harbor run -a module.path:Class`; built-in `oracle` and `nop` agents; separate verifier container; ATIF trajectory JSON. |
| 19 | **NeMo Gym** (NVIDIA) | https://github.com/NVIDIA-NeMo/Gym | Apache-2.0 | 1.3k / 2026-11-19 | `gym eval run --input … --output rollouts.jsonl` prints aggregate metrics; `gym eval reverify` recomputes rewards from stored rollouts. |
| 20 | **rLLM** | https://github.com/rllm-org/rllm | Apache-2.0 | 5.9k / 2026-10-16 | `rllm eval <name>`; separates a `@rllm.rollout` agent from the evaluator. |
| 21 | **GEM** (axon-rl) | https://github.com/axon-rl/gem | Apache-2.0 | 519 / 2026-01-21 | Gym API for LLM text envs (`gem.make`, `sample_random_action`). |
| 22 | **Agentic Benchmark Checklist (ABC)** | https://github.com/uiuc-kang-lab/agentic-benchmarks | none stated | 66 / 2025-07-31 | **The checklist to test our grader against** (oracle solver, trivial agents, list-all-answers robustness, ground-truth isolation). |
| 23 | **ImpossibleBench** | https://github.com/safety-research/impossiblebench | MIT | 82 / 2026-10-13 | Measures cheating with deliberately unsolvable tasks. Equivalent for us: clean companies where any finding is a false positive. |

Archived or legacy, not to use: `openai/gym` (archived, https://github.com/openai/gym); Atropos (archived, banner at https://github.com/NousResearch/atropos).

---

## 2. Detailed notes per project

### 2.1 Gymnasium — the contract the case cites
- **How an env is defined:** subclass `gymnasium.Env`; `reset(*, seed=None, options=None) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)` (https://gymnasium.farama.org/api/env/). `terminated` means the MDP reached a terminal state; `truncated` means a limit outside the MDP was hit (time or steps). This maps exactly to our "submit -> terminated" and "100 steps or 60 s -> truncated".
- **Seeding:** the first line of `reset` should be `super().reset(seed=seed)`. Passing `seed=None` must not reset an existing RNG, and an int seed always does (https://gymnasium.farama.org/api/env/ ; https://github.com/farama-foundation/gymnasium/blob/main/docs/introduction/create_custom_env.md). The case's `reset(seed, config)` is the same idea, with `config` playing the role of `options`.
- **Returned data must be new objects**, not mutated buffers ("the returned observation and info must be new data rather than a buffer that a previous call already returned", https://gymnasium.farama.org/api/env/). For us, deep-copy or re-serialize every observation so a policy can't mutate env state through a shared dict.
- **Env checker:** `gymnasium.utils.env_checker.check_env` runs `check_reset_seed_determinism`, `check_step_determinism`, `check_returned_data_not_reused`, `check_reset_return_type` and others, comparing results with `data_equivalence` (source: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/utils/env_checker.py ; docs: https://gymnasium.farama.org/api/utils/). **Copy these checks as our own tests.** Running them through `check_env` would force us to declare `observation_space`/`action_space`. A JSON-dict observation fits poorly into `spaces.Text`, whose default charset is alphanumeric.
- **Dependencies:** `numpy`, `cloudpickle`, `typing-extensions`, `farama-notifications` (https://github.com/Farama-Foundation/Gymnasium/blob/main/pyproject.toml). They're light, but numpy is not needed for anything else in our build.
- **Versioning:** env IDs carry a `-vN` suffix that is bumped whenever results could change (README, https://github.com/Farama-Foundation/Gymnasium). Copy this as `AuditGym-v1` plus `schema_version` in the manifest.

### 2.2 OpenEnv (Meta-PyTorch / Hugging Face; repo moved to `openenv/OpenEnv`)
- **Env definition:** an `Environment` subclass implementing `reset()`, `step()` and a `state` property. Typed `Action` and `Observation` subclasses carry `done` and `reward`, so `step()` returns an observation (wrapped in `StepResult`) instead of a 5-tuple. `State` holds at least `episode_id` and `step_count` (https://huggingface.co/docs/openenv/guides/concepts).
- **Transport and isolation:** client/server over HTTP/WebSocket. Each env is a FastAPI app (`create_app`) packaged as a Docker image. Clients connect by URL, `from_docker_image`, or `AutoEnv.from_env` (same page; README https://github.com/openenv/OpenEnv).
- **Hiding rewards from the agent:** in **simulation mode**, `/reset`, `/step`, `/state` and `/ws` are exposed for training. In **production mode**, agents get only MCP tools over `/mcp` with "no episode control", and calls there "never produce a reward, a step count or `done`" (https://huggingface.co/docs/openenv/guides/simulation-vs-production). This is the case's "open training mode returns the terminal reward; closed check keeps it with the grader" requirement, built into the server.
- **Validation CLI:** `openenv validate ./my_env --level static` (no Docker) and `--level runtime` (builds the image, replays a JSON plan of one reset plus actions, checks reward range, observation schema and network policy). Exit codes: 0 pass/warn, 1 fail, 2 unsupported package, 3 internal error. `--json` and `--output report.json` produce machine-readable reports. A `no-network` mode probes that the subject truly has no network (https://huggingface.co/docs/openenv/reference/cli).
- **Rollout collection:** `openenv collect … --provider scripted` writes `results.jsonl` with no API key (same page). This is the model for a scripted-baseline trajectory dump.
- **Verdict:** borrow the mode split and the exit-code and report discipline. Don't depend on it: it needs FastAPI, uvicorn and WebSocket, and the API "may still change" (README).

### 2.3 Prime Intellect `verifiers` and Environments Hub
- **Concepts (v1):** a `Taskset` holds tasks (a serializable `TaskData` row plus behaviour hooks), a `Harness` is the program the model runs in, an `Env` defines control flow, and a `Trace` records messages, rewards, metrics and per-call records (https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/overview.md).
- **CLI:** `vf-validate`, `vf-debug` and `vf-run`. Full model evals moved to `uv run eval` in prime-rl, so `vf-eval` is no longer the entry point (https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/getting_started.md).
- **The pattern most worth copying:** `vf-validate my-task --only-gold` runs the reference solution and expects it to pass. `--only-noop` scores the untouched environment, and "Any task that already scores 1.0 or higher with no agent actions is flagged as invalid". The run writes `results.jsonl`, `summary.json` and a resolved config, and `--resume` is supported (https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/debugging.md). For us that becomes "oracle reward ≈ max" and "silence on books with errors < oracle; silence on clean books = 1.0".
- **Out-of-process grading:** `ScoringConfig(mode="isolated")` copies only the declared output artifacts out of the solver sandbox, destroys it, and grades in a fresh container. `stage_verifier` copies private tests into the grading sandbox only (https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/building-environments.md). Model traffic goes through an interception server, so the trace is built server-side and "prevents agents tampering with their own traces". Hooks can "block reward hacks" (https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/architecture.md).
- **Re-score:** `Task.rescore(trace)` recomputes rewards from a saved trace without re-running the agent (debugging.md).

### 2.4 Nous Atropos (archived)
- Environments are microservices that push rollout groups to a trajectory API, and trainers pull batches from it (https://github.com/NousResearch/atropos).
- `python env.py process --env.data_path_to_save_groups out.jsonl` runs inference-only rollouts and writes **both `out.jsonl` and a static `out.html` viewer**. `evaluate` runs the env's eval method (same README). A self-contained HTML viewer next to the JSONL is a cheap demo artefact.
- The repo is archived ("no longer maintained"), so use it as a pattern only.

### 2.5 OpenPipe ART
- Client/server split: the code you write produces a `Trajectory` (system/user/assistant messages), your code assigns `reward`, and trajectory groups are sent to a GRPO trainer server (https://github.com/OpenPipe/ART).
- RULER is an LLM-as-judge reward (same README). The case says reward must depend on fields and evidence, not persuasive text, so **don't use an LLM judge**.

### 2.6 TRL `environment_factory`
- The trainer creates one env instance per generation. `reset(**kwargs)` returns the first observation string. **Every public method becomes a tool**, with its schema built from the docstring `Args:`. Exceptions are fed back as tool errors, and the reward function reads state off the env objects (`[env.reward for env in environments]`) (https://huggingface.co/docs/trl/openenv).
- TRL explicitly advises: "We do not recommend using generic methods like `step(action)`… the model needs meaningful tool names" (same page). Our five actions (`list_records`, `get_record`, `get_document`, `calculator`, `submit`) map 1:1 to methods.
- **Cheap win:** a ~30-line `AuditToolEnv` shim in `examples/` shows the env is trainable with TRL without a dependency in the core.

### 2.7 reasoning-gym
- `reasoning_gym.create_dataset('leg_counting', size=10, seed=42, **config)`. Each entry carries `question`, `answer` and `metadata`; `data.score_answer(answer, entry)` is the verifier (https://github.com/open-thought/reasoning-gym).
- This is our generator contract in miniature: `generate(seed, config) -> Episode(public_files, private_truth)` plus `grade(findings, private_truth)`. Keep the generator pure and the grader pure, and the env is just the thin interactive layer between them.

### 2.8 TextArena
- "Agents only need to implement a `__call__` function that accepts a string observation and returns a string action". The env loop is `get_observation()` / `step(action)` / `close()` -> rewards (https://github.com/TextArena/TextArena).
- Lesson: the jury's adapter bar should be about this low. Our version: a Python class with `act(obs: dict) -> dict`, or any executable that reads JSON lines and writes JSON lines.

### 2.9 AgentGym
- Each environment is a separate HTTP server exposing `/createEnv`, `/observation`, `/available_actions`, `/step` and `/reset`. An `EnvClient` wraps the endpoints and an `AgentController` connects agent and env (https://github.com/WooooDyy/AgentGym).
- Trajectories (AgentTraj-L) are ShareGPT-style `conversations` lists with an `item_id` (same README). A 2025 frontend replays trajectories step by step.
- Lesson: `available_actions` in the observation helps weaker policies. We can list the 5 actions plus the table names in the first observation.

### 2.10 τ-bench (v1) and τ²/τ³-bench (Sierra)
- **v1 CLI:** `python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-strategy llm --max-concurrency 10 --task-ids 2 4 6`. The repo ships `historical_trajectories/` and an `auto_error_identification.py` failure analyzer (https://github.com/sierra-research/tau-bench). The case's "measured failure analysis" item maps to that analyzer.
- **τ² gym wrapper:** `gym.make(TAU_BENCH_ENV_ID, domain=…, task_id=…)`; `info` carries `tools` and `policy`. **Actions are strings, either JSON `{"name": "search_flights", "arguments": {...}}` or function style `search_flights(origin='NYC')`** (https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md). Use the JSON `{name, arguments}` shape so LLM tool calls map directly.
- **Agent plug-in:** subclass `HalfDuplexAgent` (`get_init_state`, `generate_next_message`), then register a factory `create_agent(tools, domain_policy, **kwargs)` so it's selectable with `--agent my_agent` (https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/README.md).
- **Grading:** the reward is the **product** of components in `reward_basis`. The DB check replays the reference actions on a fresh "gold" env and compares DB end-state hashes, so any path that reaches the same state passes (https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md). This is the same principle as our "accept any complete allowed evidence set", not a single canonical path.
- **UX:** `--dry-run` prints the plan without calls, `tau2 view` browses results, `tau2 intro` gives an overview, and frozen benchmarks have "pinned data, so runs can be checked and compared" (README https://github.com/sierra-research/tau2-bench).
- **Warning from the ABC paper:** on v1, an agent returning empty responses passed tasks that were impossible by design. The authors released a patch because "a trivial do-nothing agent" scored up to 38% (https://github.com/uiuc-kang-lab/agentic-benchmarks ; https://arxiv.org/abs/2507.02825).

### 2.11 BrowserGym
- Each benchmark registers Gymnasium IDs such as `gym.make("browsergym/miniwob.choose-list")` and is enumerated through `gym.envs.registry`. New tasks subclass `AbstractBrowserTask` (https://github.com/ServiceNow/BrowserGym). The loop is plain Gymnasium.
- Companion AgentLab runs studies and comparisons (same README). Lesson: a registry of named splits (`AuditGym-v1/train`, `/dev`) is better UX than raw seed lists.

### 2.12 SWE-Gym and SWE-bench — leakage case study
- SWE-Gym warns: "SWE-Gym envs contain a git shortcut similar to SWE-bench #465. Please apply a git clean up script before using it to prevent models reward hacking" (https://github.com/SWE-Gym/SWE-Gym).
- SWE-bench #465: agents ran `git log --all` and saw the future commits that fixed the issue (https://github.com/SWE-bench/SWE-bench/issues/465). The general lesson: **anything reachable from the agent's sandbox that correlates with truth is a leak.** For us that includes provenance, truth files, the seed, and also ID ordering or naming artifacts of injected rows.
- Harness UX: `swebench eval verified -p <predictions> --run-id <id>`, `swebench eval verified --gold` (reference patches = oracle), and `swebench report <run_id>` (re-grade saved logs) (https://github.com/SWE-bench/SWE-bench). The policy's output (a predictions file) is fully separated from the grader.

### 2.13 ToolSandbox (Apple)
- `tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenario wifi_off`. There's a `Cli` agent/user so a human can play. Artifacts include `result_summary.json` per run directory (https://github.com/apple-aiml-research/ToolSandbox).
- The license is Apple's custom license, so copy patterns, not code.

### 2.14 Inspect AI (UK AISI)
- **Swap the policy from the CLI:** `inspect eval ctf.py --solver=ctf_agent -S attempts=5`, or `inspect eval evals/mytask --solver evals/my_agent` (https://inspect.aisi.org.uk/tasks.html ; https://inspect.aisi.org.uk/extensions-components.html). `--solver` takes a name or `file.py@name`.
- **Offline runs:** `mockllm/model` always returns a fixed output (or custom outputs) and needs no credentials, which is how Inspect tests its CLI without network (https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/model/_providers/mockllm.py).
- **Separating execution from scoring:** `--no-score` on eval, then scoring as a later step. `--json` emits `launch` and `done` JSON records for CI. `inspect view` serves logs on `127.0.0.1:7575` (context7 docs for `/ukgovernmentbeis/inspect_ai`; `src/inspect_ai/_cli/eval.py`, `src/inspect_ai/_view/view.py`).
- **Limits as flags:** `--message-limit`, `--time-limit`, `--token-limit` and `--limit` (number of samples) (https://inspect.aisi.org.uk/tasks.html). Our step budget and wall-clock limit should be visible run options, but fixed in "official" mode.

### 2.15 lm-evaluation-harness
- `lm_eval --model hf --model_args pretrained=… --tasks hellaswag`. Any OpenAI-compatible server works via `--model local-completions` (or `local-chat-completions`) with `base_url=…`. `--log_samples --output_path results` saves per-sample outputs (https://github.com/EleutherAI/lm-evaluation-harness).
- Lesson: the `--model <adapter-type> --model_args k=v,k=v` split is the de-facto way to plug in heterogeneous policies.

### 2.16 Harbor (Terminal-Bench 2.0 harness)
- `harbor run -p <path/to/dataset> -a <agent> -m <model> -n <concurrent> -o <jobs-dir>` (https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/jobs/run-a-job.mdx).
- **Custom agent = import path:** "pass `module.path:ClassName` to `--agent`", with `--agent-kwarg` and `--agent-env` for options (https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/agents/custom-agents.mdx). **This is the convention to adopt for `--policy`.**
- **Built-in `oracle` and `nop` agents** (https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/oracle.py). Oracle runs `solution/solve.sh` "to sanity-check that a task is solvable" (https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/tasks/solution.mdx).
- **Separate verifier container:** `[verifier] environment_mode = "separate"` starts a fresh environment for tests, and "The agent's filesystem changes are not inherited". A dedicated verifier image has tests baked in (https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/tasks/separate-verifier.mdx).
- **ATIF trajectory format:** `trajectory.json` with `schema_version: "ATIF-v1.8"`, `agent{name,version,model_name}` and `steps[{step_id, source, message, tool_calls, observation, metrics}]` (https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/agents/atif.mdx). Optionally emit an ATIF export next to our required `trajectories.jsonl`.

### 2.17 NeMo Gym (NVIDIA)
- An environment is a dataset, an agent harness, a verifier and state. Servers are split into resources servers (tools plus verification), agent servers and model servers (https://github.com/NVIDIA-NeMo/Gym).
- `gym eval run --agent mcqa_simple_agent --input data/example.jsonl --output results/rollouts.jsonl --limit 5 --num-repeats 1` prints aggregate metrics (`mean/reward`, `pass@1`) and writes `*_aggregate_metrics.json`. `gym eval reverify` recomputes rewards from stored rollouts, and `gym eval profile` gives per-task pass rates (same README).
- It requires Python 3.13 and API keys for its quickstart, so it's not a dependency candidate under a Python 3.11, offline constraint.

### 2.18 rLLM
- `rllm eval gsm8k` and `rllm train gsm8k`. The agent is a `@rllm.rollout` function `(task, config) -> Episode`, and the evaluator is separate (https://github.com/rllm-org/rllm). The same agent code drives eval and training.

### 2.19 GEM (axon-rl)
- `gem.make("game:GuessTheNumber-v0")`, `env.reset()`, `env.step(action)` returning the Gymnasium 5-tuple with **string actions**, plus `env.sample_random_action()`. Tool calling is added via a wrapper (https://github.com/axon-rl/gem). This confirms the "Gym API + text in/out" convention for LLM envs. A built-in random action sampler maps to our random-explorer baseline.

---

## 3. Reward hacking and specification-gaming references

| Source | Lesson for our grader |
|---|---|
| DeepMind, "Specification gaming" (cited by the case): https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ | Any gap between the literal reward and intent will be found. Test the reward function itself as an adversary would. |
| ABC checklist: https://github.com/uiuc-kang-lab/agentic-benchmarks/blob/main/ABC.md , paper https://arxiv.org/abs/2507.02825 | Directly relevant items: **I.b.2** "robust against systematically listing all possible answers" (our flag-everything attack); **I.h.2** "minimizes the possibility of success by random guessing" (guessed IDs); **II.4** residual state cleared between runs (reset isolation); **II.5** "Agent is completely isolated from any ground truth information"; **II.8** "Benchmark includes an Oracle solver"; **III.13** "Reports results of trivial agents (e.g., one that does nothing)". Map each test to these IDs in the design note. |
| ABC τ-bench patch: https://github.com/uiuc-kang-lab/agentic-benchmarks (news section) | A do-nothing agent scored up to 38%. Our "silence" baseline must score `2*0-1 = -1` on books with errors and `+1 - ε` on clean books. Assert both. |
| SWE-bench #465: https://github.com/SWE-bench/SWE-bench/issues/465 | Leaks come from side channels such as history, metadata and ordering, not only from the truth file. |
| ImpossibleBench: https://github.com/safety-research/impossiblebench ; write-up https://www.lesswrong.com/posts/qJYMbrabcQqCZ7iqm/impossiblebench-measuring-reward-hacking-in-llm-coding-1 | Measures a "cheating rate" on tasks that can't be solved honestly. Equivalent for us: on **clean companies** any finding is a false positive, so report the FP rate on clean episodes per policy. |
| METR, "Recent Frontier Models Are Reward Hacking": https://metr.org/blog/2025-06-05-recent-reward-hacking | Models monkey-patched timers and graders. Our grader must not run in the policy process, and the policy must not be able to write grader state. |
| verifiers noop/gold validation: https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/debugging.md | Automate "no-op must not get max reward" and "gold must get max reward" for every generated episode. |
| OpenEnv production mode: https://huggingface.co/docs/openenv/guides/simulation-vs-production | Hide the reward channel structurally in eval mode, not by convention. |

**Adversarial-submission test suite to implement.** Each is a unit test; the expected outcome is in brackets.
1. Empty findings on books with errors [F1=0, reward = -1 - cost], compared with oracle [≈ +1].
2. Empty findings on clean books [F1=1, reward ≈ +1]. Any single finding on clean books [F1=0, reward ≈ -1].
3. Flag everything: one finding per (type × record), capped at 30 [low precision, reward < silence on most episodes].
4. Exact duplicates of a correct finding [the second copy is an FP; F1 strictly lower than with one copy].
5. Correct type and IDs but **no evidence**, or evidence that is a non-allowed subset [not a TP].
6. Correct evidence but `affected_ids` permuted, upper-cased, or with whitespace [TP only if the normalization rule says so; publish that rule].
7. Amount off by tolerance ± 1 minor unit; wrong currency; same number in a different currency [boundary tests].
8. Guessed or nonexistent IDs and evidence IDs that don't exist in the episode [FP; no crash; no "does not exist in truth" message].
9. Explanation-only variation (persuasive text, "this is correct", 1,000 vs 1,001 chars) [reward invariant to text below the cap; over the cap is a schema error giving -1].
10. Schema violations: 31 findings, 11 evidence IDs, wrong types, missing `episode_id`, wrong `episode_id`, non-JSON [terminated, reward -1, error only in the grader log].
11. Submitting twice, stepping after `terminated`, stepping after `truncated` [bounded error; no second grading].
12. Timeout (no `submit` within 100 steps or 60 s) [truncated, empty findings graded].
13. Calculator abuse: `__import__('os')`, `open(...)`, `9**9**9`, very long strings, `1/0` [bounded error, not an exception or hang].
14. Path and ID injection: `get_document("../truth.json")`, `get_record("ledger", "*")`, negative or huge `limit`, a forged `cursor` [bounded error].
15. Metamorphic: permute row order and rename all IDs with a bijection [the oracle's reward is identical once IDs are mapped]. This is the jury's 20 paired checks.
16. Step-cost monotonicity: the same findings at 10 and at 90 steps differ by exactly `0.002*80`, so step count only slightly separates equal answers.

---

## 4. CLI and UX patterns — concrete commands seen

| Need | Project | Command |
|---|---|---|
| Single contract check | OpenEnv | `openenv validate ./my_env --level static --skip-build` ; `--level runtime --local --output report.json` (exit 0/1/2/3) — https://huggingface.co/docs/openenv/reference/cli |
| Oracle and no-op sanity | verifiers | `uv run vf-validate my-task -n 1 --only-gold` ; `--only-noop` — https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/debugging.md |
| Oracle run | Harbor | `harbor run -p examples/tasks/hello-world-multiple-solutions -a oracle` — https://github.com/harbor-framework/harbor/blob/main/docs-mintlify/tasks/solution.mdx |
| Oracle run | SWE-bench | `swebench eval verified --gold` — https://github.com/SWE-bench/SWE-bench |
| Plug in a custom policy | Harbor | `harbor run -t hello-world/hello-world -a examples.agents.marker_agent:MarkerAgent` — custom-agents.mdx |
| Plug in a custom policy | Inspect | `inspect eval evals/mytask --solver evals/my_agent -S attempts=5` — https://inspect.aisi.org.uk/extensions-components.html |
| Plug in a custom policy | τ² | register a factory, then `--agent my_agent` — https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/README.md |
| Run on a split, get metrics | NeMo Gym | `gym eval run --agent X --input data.jsonl --output rollouts.jsonl --limit 5` (prints `mean/reward`) — https://github.com/NVIDIA-NeMo/Gym |
| Run on a split | τ² | `tau2 run --domain airline --agent-llm gpt-4.1 --num-trials 1 --num-tasks 5` ; `--dry-run` — https://github.com/sierra-research/tau2-bench |
| Re-grade without re-running | SWE-bench / NeMo / verifiers | `swebench report <run_id>` ; `gym eval reverify` ; `Task.rescore(trace)` |
| Browse results | τ² / Inspect / Atropos | `tau2 view --file …` ; `inspect view` ; `process` writes `.jsonl` + `.html` |
| Machine-readable output | Inspect | `inspect eval … --json` (launch/done records) |
| Offline smoke run with no model | Inspect | `--model mockllm/model` |

---

## 5. Policy adapter patterns observed

| Pattern | Who uses it | Fit for us |
|---|---|---|
| **Python import path** `module.path:Class` | Harbor `-a`, Inspect `--solver file.py@name`, `openenv import --env-class module:Class` | **Yes, primary.** Zero ceremony for Python teams. |
| **Named registry** (`--agent name`) | τ², ToolSandbox, lm-eval `--model` | **Yes, for our 4 baselines:** `random`, `reconcile`, `flag_all`, `oracle`. |
| **Callable `str -> str`** | TextArena | Our `act(obs: dict) -> dict` is the JSON-typed equivalent. |
| **HTTP / WebSocket env server** | OpenEnv, AgentGym, NeMo Gym, Atropos | Optional only. Ports and servers add failure modes in an offline `--network none` container. |
| **Subprocess + stdin/stdout JSON lines** | MCP stdio transport (used by verifiers and OpenEnv tools); installed-CLI agents in Harbor | **Yes, secondary**, for any-language policies. It also gives process isolation for free. |
| **OpenAI-compatible endpoint** | lm-eval `local-chat-completions`, ToolSandbox via vLLM, `openenv collect --llm-endpoint` | Example LLM adapter only (`--policy-arg base_url=http://localhost:11434/v1`). Never part of the official check. |
| **Env-as-tools class** | TRL `environment_factory` | Ship as an example shim to show trainability. |

---

## 6. RECOMMENDATION

### 6.1 Interface design to adopt

**A. Core contract (stdlib only, Python 3.11):** `auditgym.env.AuditEnv` with
```python
obs, info = env.reset(seed: int | str, config: dict | None = None)
obs, reward, terminated, truncated, info = env.step(action: dict)
```
- Plain class with Gymnasium semantics; don't inherit `gymnasium.Env` (the case allows this, p.2). Observations, actions and info are **JSON-serializable dicts** only, and every return is freshly built, following Gymnasium's no-reused-buffer rule.
- **Action shape = tool call** (τ² JSON format, OpenAI function-call shape): `{"name": "list_records", "arguments": {"table": "ledger", "cursor": null, "limit": 50}}`. Publish `schema/actions.json` (JSON Schema per action, OpenEnv `/schema` style) and put the same list in the first observation's `available_actions` (AgentGym style). An LLM adapter can pass that file straight in as `tools`.
- **Two modes, enforced structurally (OpenEnv simulation/production):** `mode="train"` returns the terminal reward on `submit`. In `mode="eval"`, `reward` is always `0.0`, `info` carries only `steps_remaining` and a bounded `error` code, and grading happens in the runner process after the episode, written to a file the policy never sees.
- **Generator/grader as pure functions (reasoning-gym style):** `generate(seed, config) -> (PublicEpisode, PrivateTruth)` and `grade(findings, truth) -> GradeResult`. The env is the thin interactive layer between them, so the grader is testable without the env.
- **Optional extras, outside the core:** (1) `auditgym/contrib/gymnasium_env.py`, a ~40-line wrapper with JSON-string `Text` spaces registered as `AuditGym-v1` (pinned `gymnasium==1.4.0` extra). (2) `examples/trl_tool_env.py`, a TRL `environment_factory` shim. Neither is imported by the check.

**B. Isolation (case §03/§04; ABC II.5):**
- The **runner process owns** the generator, truth and grader. The **policy always runs in a child process** in eval mode, including Python `module:Class` policies (via `python -m auditgym.policy_host module:Class`). Otherwise an in-process policy could `gc.get_objects()` its way to the truth (the METR monkey-patching lesson).
- Wire protocol over stdin/stdout, one JSON object per line. Runner to policy: `{"type":"reset","observation":…,"info":…}` / `{"type":"step","observation":…,"reward":…,"terminated":…,"truncated":…,"info":…}` / `{"type":"close"}`. Policy to runner: one action object per line. Run the child with a scrubbed environment, a temporary empty cwd, and a per-step and per-episode wall-clock timeout (60 s per episode leads to `truncated=true` with empty findings).
- In closed runs, truth is never written to disk while the policy is alive. `truth.json` and `provenance.json` are written after the policy exits, into the run directory. The Docker image contains **no** closed-seed data.
- The oracle is privileged: a built-in that runs **inside the runner** (it needs truth) but still acts only through `env.step()`, so its trajectory goes through the same leak scanner. It's not available through `cmd:`.

**C. Trajectory and results format:**
- `trajectories.jsonl`, one line per step with the case's required keys: `episode_id, step, action, observation, reward, terminated, truncated`. Run-level metadata (env version, config hash, policy spec, seed list, Python version, git SHA) goes in `run.json`, **not** in observations. Seeds are treated as secret in eval mode.
- `results.jsonl`, one line per episode: `tp, fp, fn, precision, recall, f1, return, steps, wall_s, termination` (`submitted|max_steps|timeout|schema_error|policy_crash`), plus a matcher trace (which finding matched which issue, and why others failed). That trace is the "measured failure analysis" (τ-bench `auto_error_identification` analogue).
- `summary.json` aggregates per split, with a separate column for the clean-episode FP rate (the ImpossibleBench analogue).
- Optional: `export --atif` (Harbor ATIF-v1.8) and a self-contained `report.html` (Atropos `process` style) for the demo.

### 6.2 CLI design (single entrypoint `python -m auditgym`, console script `auditgym`)

```bash
# THE single verification command (offline; exit 0 = pass, 1 = fail, 2 = usage/internal)
auditgym check                                   # == pytest + contract/determinism/leak checks
                                                 #    + 4 baselines on train+dev + expected-ordering asserts
docker build -t auditgym . && docker run --rm --network none auditgym check

# Data
auditgym generate --split train --out data/train      # 10 episodes; also --seed S --config cfg.json
auditgym generate --split dev   --out data/dev

# Run any policy on a split -> runs/<name>/{run.json,trajectories.jsonl,results.jsonl,summary.json}
auditgym run --policy reconcile                       --split dev --mode eval --out runs/reconcile
auditgym run --policy my_pkg.agents:MyPolicy          --split dev --policy-arg temperature=0
auditgym run --policy "cmd:node my_agent.js"          --split dev       # any language, JSON lines
auditgym run --policy examples.llm_policy:OpenAICompatPolicy \
             --policy-arg base_url=http://localhost:11434/v1 --policy-arg model=qwen3:8b --split dev

# Compare -> markdown + JSON table: policy x split -> P, R, F1, return, steps, wall_s, clean-FP%
auditgym compare runs/random runs/reconcile runs/flag_all runs/oracle runs/mine
auditgym run --policy random,reconcile,flag_all,oracle --split dev --compare   # one-shot

# Re-grade / replay / audit (SWE-bench report, NeMo reverify, verifiers rescore analogues)
auditgym grade  --episode data/dev/ep_03 --findings findings.json   # offline re-grade, prints match trace
auditgym replay runs/reconcile/trajectories.jsonl                   # re-execute actions, diff obs+rewards
auditgym leakscan runs/reconcile                                    # grep observations/info/errors for truth tokens
auditgym perturb --episode data/dev/ep_03 --permute --rename-ids    # paired-check generator

# Manual exploration (ToolSandbox `Cli` agent / tau2 play analogue)
auditgym play --split dev --episode 0
```

What `auditgym check` asserts, mapped to the jury's 100 points:
- **Contract & isolation (25):** reset/step types; 100-step cap; `submit` gives terminated; cap or timeout gives truncated; unknown action or ID gives a bounded error and uses a step; page ≤ 50 and stable cursors; calculator sandbox; `leakscan` clean on every trajectory.
- **Data & reference (25):** per episode, books balance before injection; every injected issue changes the books by exactly the declared amount; each truth issue's allowed evidence set is reachable through the public tools (the oracle proves it by fetching it); clean episodes have zero issues.
- **Reward robustness (25):** the 16 adversarial tests in §3.
- **Repro & generalization (15):** same seed, config and actions give byte-identical trajectories (Gymnasium `check_reset_seed_determinism` / `check_step_determinism` copied); `perturb` keeps the oracle reward invariant.
- **Engineering (10):** `check` prints versions, the config hash and timing, and writes `runs/check/summary.md`.

### 6.3 Reuse as a dependency vs copy as a pattern

| Item | Decision | Why |
|---|---|---|
| Python stdlib (`json`, `csv`, `decimal`, `random`, `hashlib`, `argparse`, `subprocess`, `ast`) | **Dependency (only runtime deps)** | Offline, 2 CPU / 4 GB, Python 3.11, nothing to pin. |
| `pytest` (pinned) | **Dev dependency**, installed at `docker build` | The tests are a required deliverable. |
| `gymnasium==1.4.0` | **Optional extra** for `contrib/gymnasium_env.py` | The jury may want `gym.make`; the core must not need numpy. |
| Gymnasium `env_checker` determinism and no-reuse checks | **Copy pattern** (MIT) | We need them on JSON dicts, not Spaces. |
| OpenEnv simulation/production split, validate exit codes | **Copy pattern** | FastAPI, WebSocket and Docker-in-Docker don't fit the offline check. |
| verifiers gold/noop validation | **Copy pattern** | Becomes `oracle` and `silence` assertions in `check`. |
| Harbor `module:Class` agents, `oracle`/`nop`, ATIF | **Copy pattern** | CLI convention and optional export format. |
| τ² JSON tool-call action shape, `info["tools"]` | **Copy pattern** | LLM-friendly actions. |
| TRL `environment_factory` | **Example shim only** | Shows trainability with no core dependency. |
| Inspect / lm-eval / NeMo Gym / Harbor / rLLM / verifiers as runtimes | **Do not depend** | They need Python 3.12–3.13+ (τ² needs ≥3.12, NeMo Gym 3.13, Harbor tasksets 3.12+), network or API keys, or Docker sandboxes. They're far too heavy for 1 GB and offline. |
| LLM judges (ART RULER, τ² NL assertions) | **Avoid** | The case says reward must depend on fields and evidence, not persuasive text. |
| ToolSandbox code | **Don't copy code** | Apple custom license. |

### 6.4 Pitfalls (with sources)

1. **The seed is a master key.** With an open generator, anyone who knows (or brute-forces) the seed regenerates the truth. Keep seeds out of observations, `episode_id`, file names and error messages. Derive `episode_id` from an HMAC rather than the raw seed. Accept arbitrary-length int/str seeds so the jury can use 128-bit ones. This is the SWE-bench #465 lesson applied to us: https://github.com/SWE-bench/SWE-bench/issues/465
2. **Python `random` is only stable for `random()` across versions.** `choice`, `shuffle`, `sample` and `randrange` "are subject to change across Python versions" (https://docs.python.org/3/library/random.html#notes-on-reproducibility). Pin Python 3.11 in Docker *and* build helpers on `Random.random()` with an explicit algorithm, using a separate `Random(hash_of(seed, "component"))` stream per generator stage.
3. **Hidden nondeterminism:** `set` iteration over strings (`PYTHONHASHSEED`), `dict` built from sets, `datetime.now()`, float FX math, locale-dependent formatting. Use `Decimal` for FX, sort before emitting, use `json.dumps(sort_keys=True, separators=…)`, and set `PYTHONHASHSEED=0` in the Dockerfile anyway. Test byte identity (Gymnasium `data_equivalence(exact=True)` idea: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/utils/env_checker.py).
4. **Structural leaks:** injected rows appended last, injected IDs with gaps or suffixes, duplicate invoices whose `doc_id` sorts adjacent, error messages like "already flagged" or "not a known issue". Shuffle after injection, re-ID everything after injection, and keep error codes truth-agnostic (ABC II.5: https://github.com/uiuc-kang-lab/agentic-benchmarks/blob/main/ABC.md).
5. **`info` is an observation too.** τ²'s gym `info` carries `simulation_run` state (https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md). Our `info` must never include grading diagnostics in eval mode, and `leakscan` must scan `info` and errors as well as `observation`.
6. **Trivial agents scoring well** (τ-bench do-nothing at 38%: https://github.com/uiuc-kang-lab/agentic-benchmarks). Make silence vs oracle vs flag-everything an asserted ordering in `check`, not just a reported number (verifiers `--only-noop`: https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/debugging.md).
7. **An in-process policy can tamper with the grader** (METR: https://metr.org/blog/2025-06-05-recent-reward-hacking). Always use a subprocess in eval mode, and the runner computes the reward from its own state, never from anything the policy returns besides `findings`.
8. **Calculator as an escape hatch.** Never `eval()`. Parse with `ast` into `Decimal` arithmetic, whitelist only `BinOp`/`UnaryOp`/`Constant`, cap expression length and exponent size.
9. **Ambiguous matching.** One-to-one matching with ties must be deterministic and published (case §05). Use a stable order (truth `issue_id` asc, finding index asc) or an optimal assignment with a deterministic tie-break, and log the match trace. τ² shows the value of documenting exactly what gates reward: https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
10. **Shared or mutable observation objects** let a policy edit env state. Re-serialize every observation (https://gymnasium.farama.org/api/env/).
11. **Over-engineering the transport.** HTTP and WebSocket servers (OpenEnv, AgentGym) add ports, readiness races and concurrency settings: TRL had to document session drops and `MAX_CONCURRENT_ENVS` (https://huggingface.co/docs/trl/openenv). Stdio JSON lines are enough for the jury and for `--network none`.
12. **Dependency drift in the offline check.** Pin everything with hashes (`pip install --require-hashes` or `uv.lock`) and install at build time. The run must succeed with `--network none`, which is also how OpenEnv's runtime validator proves no-network (https://huggingface.co/docs/openenv/reference/cli).
