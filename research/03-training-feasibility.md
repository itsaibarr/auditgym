# 03 — Feasibility of an optional training demo

Date: 19 Nov 2026. Deadline: 22 Nov 2026, 23:59 Astana.
Scope: whether to train a policy against the Genesis audit gym, and if so how, given that training earns **0 points on its own**, must not be required by the official check (no GPU, no paid APIs, no network), and any claim of improvement needs a controlled comparison on held-out data (case.pdf §01, §07, §12).

Labels used below:
- **[measured]**: I ran it or read it directly from a primary source.
- **[cited]**: a number from someone else's source, with the URL given.
- **[estimate]**: my own back-of-envelope arithmetic, with the assumptions shown. Treat these as order-of-magnitude only.

---

## 0. TL;DR

1. **Skip LLM RL.** A 1–4B model on a free T4 cannot work through this environment as specified. One 50-row ledger page is about **5,000 Qwen3 tokens as compact JSON** [measured, §1.2], so a 2,000-row ledger is about 200k tokens. Qwen3 small models have a native context of 40,960 tokens [measured from config.json]. The best open sub-4B tool-callers score **22–58% on BFCL multi-turn** [measured from the BFCL CSV, §2]. The heuristic baseline will very likely beat any LLM you can train by Saturday.
2. **If you train anything, train a CPU-only learned candidate scorer:** logistic regression or gradient boosting over reconciliation candidates, with the decision threshold (and optionally a few weights) tuned directly on episode return using a derivative-free search such as evolution strategies or random search. Fit on train seeds, evaluate once on held-out dev seeds plus freshly generated held-out seeds, and use a paired bootstrap against the heuristic. Cost: about 4–6 hours of engineering and minutes of CPU. It runs inside the official 2-CPU/4 GB limit, so it can even be part of the one-command check.
3. **Use Kaggle, not Kamatera.** Kamatera's own site has **no GPU products at all**: 0 mentions of "GPU" on its homepage, pricing, cloud-servers and free-trial pages, and no GPU URLs in its sitemap [measured, §4.2]. The trial is one CPU server worth up to $100 for 30 days, needs a credit card, and starts charging if you don't cancel. Kaggle gives T4×2 with roughly 30 GPU-h/week and 12-h sessions, and can be driven programmatically with `kaggle kernels push`/`status`/`output`.
4. **The only optional LLM work worth doing** is a timeboxed zero-shot run on Kaggle T4 with a small tool-calling model, constrained decoding and a sliding-window memory, on 5–10 *small* dev episodes. Its value is a **measured failure analysis** (invalid-ID rate, malformed-JSON rate, context overflows), which the "Engineering and explanation" criterion explicitly rewards (case.pdf §10). Do it only after every mandatory check is green.

---

## 1. How a small LLM would actually interact with this env

### 1.1 The loop, concretely

```
obs0 = env.reset(seed, config)              # task + manifest + short policy + budget
messages = [system(tool schemas + rules + output contract), user(obs0)]
for t in range(100):
    raw = llm.generate(messages, constrained_to=ACTION_SCHEMA)   # one JSON action
    action = parse_and_validate(raw)        # agent side: JSON parse, schema, ID whitelist
    if action is None:                      # still unparseable after k local re-samples
        action = {"type": "list_records", ...}   # or a safe fallback; avoid wasting a step
    obs, r, term, trunc, info = env.step(action)   # bad ID or bad action => bounded error, costs a step (§04)
    messages = memory.update(messages, action, obs)    # truncate or summarize, see 1.3
    if term or trunc: break
```

Rules in the case that matter for an LLM:

- An unknown ID or action returns a bounded error **and costs a step** (case.pdf §04).
- A malformed `submit` structure **ends the episode with reward −1** (case.pdf §05). This is the most dangerous failure mode for a small model. The agent harness must validate the findings JSON against the schema *before* calling `submit`, locally and without spending an env step.
- The step penalty is small (0.002/step, so −0.2 at 100 steps), but the 100-step cap is hard and truncation gives an empty answer (§04, §05).
- In closed evaluation, reward is withheld until the run ends (§04), so the policy cannot learn anything within an episode.

### 1.2 Context growth: measured token counts

I tokenized a synthetic 50-row `ledger` page with the actual Qwen3 tokenizer (`Qwen/Qwen3-1.7B/tokenizer.json`), using the columns from case.pdf §03 and realistic IDs, ISO timestamps and minor-unit integers [measured]:

| Rendering of one 50-row page | chars | tokens | tokens/row |
|---|---|---|---|
| JSON with default separators | 11,351 | 5,907 | 118 |
| JSON, compact separators | 10,443 | 5,042 | 101 |
| CSV-in-a-string | 4,120 | 3,692 | 74 |

Qwen tokenizers split digits individually, so ledgers full of numbers are expensive. Consequences:

- A 200-row ledger is 4 pages, about 20k tokens. A 2,000-row ledger is 40 pages, about 200k tokens. Invoices, payments and FX come on top of that.
- Qwen3-0.6B/1.7B/4B have `max_position_embeddings = 40960` [measured, HF config.json: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json, https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json].
- KV cache for Qwen3-1.7B (28 layers × 8 KV heads × 128 dim × K,V × 2 bytes) is about 112 KiB/token, so a 32k-token episode needs about 3.5 GiB of KV per sequence [estimate from config]. On a 16 GB T4 that rules out batched RL rollouts with full history.
- Conclusion: **an LLM agent cannot keep raw history.** It needs a memory scaffold:
  1. Keep the system prompt, the task and manifest, and a running *notes* block that the agent writes (candidate IDs, running totals).
  2. Keep only the last 1–2 raw observations.
  3. Drop older observations, replacing each with a one-line stub (`[page ledger c_50..c_100 read]`).

  The catch is that the notes are now the model's own transcription of IDs and amounts. That is exactly where small models hallucinate, so the notes must be validated against the IDs actually seen (see 1.4).
- Keep the env contract unchanged. You *can* make observations compact (minimal separators, no pretty-printing), which saves about 15% [measured above], without changing the schema.

### 1.3 What the reconciliation task asks of the model

To find a duplicate invoice, a partial or missing payment allocation, an FX date or rate error, or a period cutoff error, the agent must join rows across `ledger`, `invoices`, `payments` and `fx` and do minor-unit arithmetic with currency exponents. A heuristic does that join exactly in a few milliseconds. A 1–4B model has to do it in context across pages it can no longer see. That is the core reason to expect an LLM policy to be **well below** the heuristic baseline within 3 days. It is my judgement, and it is consistent with the multi-turn scores in §2.

### 1.4 Hallucination mitigations, ranked by value for this env

1. **ID whitelisting on the agent side (highest value, zero training).** Keep the set of IDs actually returned by the env. At each step, build the action JSON schema with `enum` constraints for `id`, `doc_id`, `affected_ids` and `evidence_ids` drawn from that set. The model then *cannot* emit an invented ID. This is legitimate agent scaffolding. It does not touch the env or leak truth.
2. **Grammar or JSON-schema constrained decoding.**
   - vLLM structured outputs support `json` (schema), `regex`, `choice`, `grammar` and `structural_tag`, with xgrammar or guidance backends (https://docs.vllm.ai/en/latest/features/structured_outputs.html).
   - XGrammar reports "near-zero overhead" structured generation (https://arxiv.org/abs/2411.15100). XGrammar-2 targets agentic workloads whose structure changes during generation, i.e. per-step schemas (https://arxiv.org/pdf/2601.04426v2, https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation).
   - llama.cpp converts a subset of JSON Schema to GBNF for its CPU/GPU server (`response_format`/`json_schema`). Limitations: no `prefixItems`, no nested `$ref`, `minimum`/`maximum` for integers only. `additionalProperties:false` gives "faster grammars and fewer hallucinations". The schema is **not** injected into the prompt, so the prompt must still describe the format (https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).
   - Caveat: strict format constraints can hurt reasoning. GPT-3.5 on GSM8K dropped from 76.60% to 49.25% under JSON mode (https://arxiv.org/abs/2408.02442). Mitigation: let the model write a short free-text "thought" field first, then the constrained action, or constrain only the action span (vLLM `structural_tag`).
3. **Tool-call fine-tuned base models** such as xLAM-2 and Qwen3/3.5. See §2.
4. **Env-side ID validation** is already required: unknown ID gives a bounded error and costs a step (case.pdf §04). It protects the env, not the agent's score.
5. **Short observations:** compact JSON, `limit` choices, reading invoices and payments (small tables) before the ledger.
6. **Pre-submit validator:** a local schema check plus an "every affected/evidence ID was observed" check, to avoid the −1 malformed-submit penalty.

Related literature:
- Fabricated tool calls are more frequent on an unconstrained raw-JSON surface than on a single-registry surface, and model scale did not reduce them (https://arxiv.org/pdf/2609.19425).
- RL for reasoning can *increase* tool hallucination ("Reasoning Trap", arXiv:2510.22977, summarized at https://www.beri.net/article/reasoning-trap-tool-hallucination-enterprise-agents; secondary source).

---

## 2. Which small open models are best at multi-turn tool use (0.5–9B)

The primary data is the BFCL v4 leaderboard CSV, last updated 2026-04-12 (https://gorilla.cs.berkeley.edu/leaderboard.html; raw data at https://gorilla.cs.berkeley.edu/data_overall.csv). I filtered it myself [measured]:

| Model (mode) | BFCL rank / 109 | Overall | **Multi-turn** | Live | Irrelevance |
|---|---|---|---|---|---|
| xLAM-2-8b-fc-r (FC) | 34 | 46.68% | **70.00%** | 67.95% | 63.28% |
| BitAgent-Bounty-8B | 36 | 46.23% | 62.38% | 93.12% | 97.48% |
| Qwen3-8B (FC) | 39 | 42.57% | 41.75% | 80.53% | 79.07% |
| ToolACE-2-8B (FC) | 40 | 42.44% | 38.38% | 77.42% | 90.79% |
| **xLAM-2-3b-fc-r (FC)** | 42 | 41.22% | **58.38%** | 62.92% | 63.45% |
| Nanbeige4-3B-Thinking-2511 (FC) | 25 | 51.40% | 51.12% | 79.42% | 83.09% |
| Qwen3-4B-Instruct-2507 (FC) | 54 | 35.68% | 22.12% | 76.39% | 84.93% |
| Arch-Agent-3B | 56 | 35.36% | 34.88% | 72.91% | 74.67% |
| Arch-Agent-1.5B | 60 | 32.14% | 26.62% | 67.73% | 74.83% |
| xLAM-2-1b-fc-r (FC) | 65 | 30.44% | 36.00% | 55.14% | 64.47% |
| Hammer2.1-3b (FC) | 68 | 29.71% | 16.50% | 70.54% | 86.12% |
| Qwen3-1.7B (FC) | 71 | 28.41% | 11.00% | 74.61% | 76.54% |
| Hammer2.1-1.5b (FC) | 75 | 27.88% | 15.62% | 69.50% | 79.40% |
| Qwen3-0.6B (FC) | 92 | 23.93% | 3.62% | 56.62% | 80.84% |
| Llama-3.2-3B-Instruct (FC) | 98 | 21.95% | 4.00% | 58.33% | 52.06% |
| Gemma-3-4b-it (Prompt) | 101 | 19.62% | 0.38% | 60.84% | 53.94% |
| Llama-3.2-1B-Instruct (FC) | 107 | 10.82% | 0.00% | 11.77% | 51.57% |

For comparison, Claude-Opus-4.5 (FC) is rank 1 at 77.47% overall and 68.38% multi-turn (same CSV).

Takeaways:
- **xLAM-2-3b-fc-r is the strongest sub-4B multi-turn model on BFCL.** But its license is **CC-BY-NC-4.0, "for research purposes only"**, it is built on Qwen-2.5, and it needs a custom vLLM tool parser plugin (https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r). A hackathon is non-commercial, but note the license in the README.
- **Qwen3 small models are good at single calls and weak at multi-turn:** Qwen3-1.7B scores 74.6% live but 11% multi-turn.
- **Qwen3.5 small (0.8B/2B/4B/9B, Mar 2026).** Self-reported scores: Qwen3.5-4B BFCL-V4 50.3, TAU2-Bench 79.9; Qwen3.5-9B BFCL-V4 66.1, TAU2 79.1 (https://huggingface.co/Qwen/Qwen3.5-4B). These are vendor numbers and not on the CSV above. Risks on a T4:
  - It is a **hybrid Gated-DeltaNet + attention, vision-language** model.
  - It runs in **thinking mode by default**.
  - The card asks for **vLLM/SGLang/transformers built from main or nightly**.
  - I found no evidence that its linear-attention (FLA/Triton) kernels are tested on Turing/sm75. The FLA CI lists 4090/H100/A770 (https://pypi.org/project/rwkv-fla/0.7.202506150816/).
  - A third-party guide says vLLM fast inference had to be disabled for Qwen3.5 GRPO (https://www.besthub.dev/articles/from-zero-to-deployment-a-complete-qwen3-5-fine-tuning-guide-869c8e1c888c, secondary).
  - **Treat it as unverified on T4.**
- **SmolLM3-3B** (Apache-2.0, 64k context) reports "BFCL" 92.3 without thinking versus Qwen3-4B at 95.0 (https://huggingface.co/HuggingFaceTB/SmolLM3-3B). That is a self-reported single-turn subset and not comparable to the multi-turn column above.
- Independent eval: on BFCL, xLAM-2-3b reaches 55.62% multi-turn versus Qwen3-4B at 35.25%, and the 1–3B tier "significantly outperform[s]" models under 1B (https://arxiv.org/abs/2511.22138).
- RL can lift multi-turn tool use: PROVE (GRPO on about 13K examples, 20 stateful MCP servers) reports up to **+10.2 BFCL multi-turn** on Qwen3-4B/8B (https://arxiv.org/abs/2606.03892). That took a purpose-built data pipeline, not 3 days.

**Pick for a zero-shot demo on T4:** `Qwen3-4B-Instruct-2507` (dense, standard attention, Apache-2.0, supported by vLLM on T4 and by TRL's template patching) or `xLAM-2-3b-fc-r` (better multi-turn, NC license). Do not use Gemma 3 or Llama 3.2 at ≤4B: their multi-turn scores are about 0–4%.

---

## 3. Training recipes, and what fits a single 16 GB T4 / 2×T4

### 3.1 Recipe that is known to work for small models on multi-turn envs

**SFT warm-start on expert trajectories, then GRPO.** TRL's own Wordle example is the clearest public evidence (https://huggingface.co/docs/trl/en/openenv):

- Base Qwen3-1.7B wins **0/30** games. With SFT on **1,000 solver games** in the same tool-call format it wins 10/30. With SFT plus GRPO it wins 20/30, and invalid guesses drop from 41 to 3.
- The same page notes: *"Qwen3-1.7B almost never wins a game, so GRPO from the base model has little to learn from: a group of games where none is won gives no signal."* The same applies here. The terminal reward is −1 to −0.1 for almost every untrained rollout (silence on error books gives 2·0−1 = −1), so GRPO groups will have near-zero variance.
- Compute for that GRPO stage: **150 steps ≈ 2 h 15 min on one A100 80GB**, `num_generations=16`, `max_completion_length=4096`, `max_tool_calling_iterations=12` (same page). A T4 has roughly 1/5 of the A100's dense FP16 compute (T4: 65 TFLOPS FP16, 320 GB/s, 16 GB, no native bf16; https://www.nvidia.com/en-us/data-center/tesla-t4/). The same run would be **at least about 10–15 h on a T4** [estimate], and it would not fit in 16 GB at those settings.

For this env, the analogous recipe would be:
1. Generate oracle/heuristic trajectories on train seeds (the oracle is evaluator-only for official runs, but on *train* seeds you may use truth to produce expert demos).
2. Rewrite them as tool-call transcripts with compact observations and the sliding-window memory from §1.2.
3. Run LoRA SFT. Rejection sampling: keep only trajectories with reward ≥ threshold.
4. Run GRPO with TRL `environment_factory`, wrapping the env as tool methods `list_records`, `get_record`, `get_document`, `calculator`, `submit`. TRL recommends individual named tools over a generic `step(action)` (https://huggingface.co/docs/trl/en/openenv).

### 3.2 Framework fit for a single T4 or 2×T4

| Framework | Multi-turn env support | T4 / single-GPU evidence | Verdict |
|---|---|---|---|
| **TRL `GRPOTrainer` + `environment_factory`** | Yes. Public methods become tools, exceptions are fed back as tool errors, `max_completion_length` caps the *whole* multi-turn conversation, `max_tool_calling_iterations` caps turns, the chat template must be prefix-preserving and TRL auto-patches Qwen3; requires `transformers>=5.2.0`, experimental (https://huggingface.co/docs/trl/en/grpo_trainer, https://huggingface.co/docs/trl/en/openenv) | Wordle Colab notebook uses Qwen3-1.7B, `num_generations=2`, `vllm_gpu_memory_utilization=0.15`, `vllm_max_model_length=3072`, gradient checkpointing (https://huggingface.co/docs/openenv/tutorials/wordle-grpo) | **Best fit.** On 2×T4, use `vllm_mode="server"` with one GPU for vLLM and one for training |
| **Unsloth GRPO** | Mostly single-turn reasoning notebooks. Has OpenEnv/NeMo-Gym examples on Colab | Kaggle T4 notebook for **Qwen3 (4B) GRPO** exists (`accelerator=nvidiaTeslaT4`) (https://unsloth.ai/docs/get-started/unsloth-notebooks). Qwen3-4B GRPO: **~40 steps in 40 min at 14.5 GiB** with `num_gen=2` and standby memory (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl). fp16 recommended for RL (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/fp16-vs-bf16-for-rl) | Good for memory. Multi-turn wiring is on you |
| **OpenPipe ART (+RULER)** | Yes, multi-step agents with GRPO. RULER is an LLM-as-judge reward, which you don't need because your reward is exact | Examples are 3B–27B. Serverless backend needs a W&B key (paid service). No T4 claim (https://github.com/OpenPipe/ART) | Not needed. RULER adds a judge LLM, which conflicts with "no paid APIs" |
| **verifiers / prime-rl** | Yes, multi-turn | Alphabet-sort multi-turn LoRA on Qwen3-4B: "single H100 … just over an hour"; reverse-text 0.6B single-turn on a consumer GPU in minutes (https://github.com/PrimeIntellect-ai/prime-rl). Warns that Qwen3 templates strip `<think>`, breaking the increasing-context requirement | H100-class examples. Unproven on T4 |
| **SkyRL / Agent-R1 / rLLM / RAGEN** | Yes (verl-based) | SkyRL multi-turn examples use 8×H100 (https://www.spheron.network/blog/deploy-skyrl-on-gpu-cloud-train-coding-agents-with-rl-on-h100/). Agent-R1 needs `verl==0.7.0`, examples on Qwen3-4B, no GPU spec (https://github.com/0russwest0/Agent-R1). rLLM uses verl for multi-GPU or tinker (https://github.com/rllm-org/rllm). RAGEN trained Qwen2.5-0.5B on H100/A100 (https://arxiv.org/html/2504.20073v2) | Too heavy for a T4 in 3 days |

Hard constraints on Kaggle hardware:
- vLLM needs compute capability ≥ 7.5, so **T4 is OK and P100 (6.0) is out** (https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html).
- Current PyTorch builds also fail on Kaggle's P100 ("no kernel image is available"; open issue since 20 Mar 2026: https://github.com/Kaggle/docker-python/issues/1546). **Use T4×2 only.**

### 3.3 Realistic throughput on T4 for *this* env [estimate]

Assumptions: Qwen3-1.7B in fp16, LoRA, vLLM server on GPU0 and trainer on GPU1. Episodes capped at about 25 actions with a 6k-token sliding window. About 40 output tokens per action. Prefix caching on.

- **Decode.** About 3.4 GB of weights / 320 GB/s gives about 10 ms per decode step, so about 90 tok/s single-stream. With 8–16 parallel rollouts, aggregate is a few hundred tok/s. A third-party report says ~285 tok/s at batch 64 on a T4 (vLLM V1, model unspecified; https://markaicode.com/benchmarks/vllm-benchmark/, secondary). One 25-action episode needs about 1k generated tokens plus about 25 prefills of 1–5k tokens, so roughly 20–40 s wall per episode when batched 8-wide.
- **Training.** 8 rollouts × about 6k tokens, LoRA forward+backward with checkpointing, about 1–2 min per GRPO step on one T4.
- **Total.** About 2–3 min per GRPO step of 8 rollouts, so 100 steps (800 episodes) takes about 4–5 GPU-h. Add SFT (about 1 h for a few thousand short transcripts) and evaluation (about 0.5 h). **About 6–8 GPU-h and 15–30 engineer-hours**, with a significant chance of ending at "no significant improvement" or "still far below heuristic". The Unsloth Qwen3-4B anchor (about 1 GRPO step per minute at `num_gen=2` on short math prompts) suggests the 4B would be about 2–3× slower than the 1.7B here.

**Known successful small-model multi-turn RL:**
- TRL Wordle with Qwen3-1.7B (SFT+GRPO, A100).
- RAGEN with Qwen2.5-0.5B on Sokoban/FrozenLake. Success only about 20%, and RAGEN documents the "Echo Trap" collapse, needing trajectory filtering (https://arxiv.org/html/2504.20073v2).
- ART with Qwen2.5-3B on Tic-Tac-Toe/Codenames/MCP (https://github.com/OpenPipe/ART).
- PROVE with Qwen3-4B, BFCL-MT +10.2 (https://arxiv.org/abs/2606.03892).

None of these was trained on a T4, and none has 100-step episodes with 5k-token observations.

---

## 4. Compute options

### 4.1 Kaggle (recommended if any GPU is used)

- **Accelerators:** `NvidiaTeslaT4` = "GPU T4 ×2, the default GPU". Also listed: `NvidiaL4`, `NvidiaL4X1`, `NvidiaTeslaA100`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`, but "some are limited to specific competition participants, and some are limited to Kaggle admins" (https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md). Assume T4×2.
- **P100:** still selectable per older docs, but broken with current PyTorch (https://github.com/Kaggle/docker-python/issues/1546).
- **Quota:** "30 hours or sometimes higher depending on demand" per week, quoted from Kaggle's GPU-tips page by secondary sources. The exact number is shown in account settings (https://fastero.com/blog/kaggle-notebooks, https://aifree.dev/offer/kaggle-free-gpu-tpu). I could not fetch Kaggle's JS-rendered docs directly, so **check your account settings**.
- **Session:** 12 h for CPU/GPU, 9 h for TPU. Idle timeout is reported as 20 min for interactive sessions, while the GPU-tips page refers to 60 min (https://fastero.com/blog/kaggle-notebooks). Committed ("Save & Run All") runs continue after you close the tab.
- **Storage:** `/kaggle/working` is about 20 GB and persists as notebook output. Everything else is scratch (same source). For persistence across runs, upload checkpoints as a Kaggle Dataset or Model and attach them via `dataset_sources`/`model_sources`.
- **Internet and phone:** `enable_internet` defaults to false in `kernel-metadata.json` (https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md). GPU and internet need a phone-verified account (secondary: https://fastero.com/blog/kaggle-notebooks). Internet is needed for `pip install` and HF downloads.
- **Programmatic control:**
  - `kaggle kernels push -p <dir> [--accelerator NvidiaTeslaT4] [-t <timeout>]` uploads and runs.
  - `kaggle kernels status <owner/slug>` polls.
  - `kaggle kernels output <owner/slug> -p out/` downloads the produced files.
  - Metadata fields include `enable_gpu`, `enable_internet`, `machine_shape` (e.g. `NvidiaTeslaT4`), `dataset_sources`, `model_sources`, `kernel_sources`.
  - Sources: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md, https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md.
  - This is enough for a scripted "push → poll → fetch trajectories.jsonl/metrics.json" loop from a laptop.
- **Cost:** $0.

### 4.2 Kamatera (console.kamatera.com)

- **No GPU instances.** I fetched kamatera.com's homepage, `/pricing/`, `/cloud-servers/` and `/free-trial/`: 0 occurrences of "gpu" on any of them. Grepping its English sitemaps (pages, services, FAQ, knowledge base, posts, blog) for gpu/nvidia/graphic found nothing [measured, 19 Nov 2026]. The pricing page configures only CPU type (A/B/T/D), vCPU, RAM, NVMe and add-ons (https://www.kamatera.com/pricing/).
  - A SourceForge listing claiming RTX 3090 at $0.15/h appeared in search results, but it is the **gpu.nz** product page (https://sourceforge.net/software/product/gpu.nz/), not Kamatera.
  - A 2026 VPS roundup lists Kamatera GPU support as "Not confirmed" (https://googiehost.com/blog/best-managed-vps-hosting-for-ai-agents/).
- **Trial terms** (official FAQ):
  - *"During the 30-day trial period, you may create one server with a configuration costing up to $100 … 1 TB of free traffic … Customers must provide a valid credit card to start the trial … if you do not cancel before the trial ends, we will begin charging for services used."* (https://www.kamatera.com/faq/answer/what-are-the-terms-of-the-free-trial/)
  - A credit card is mandatory. If the card is rejected, a refundable PayPal deposit is the alternative (https://www.kamatera.com/faq/answer/can-i-sign-up-for-the-free-trial-without-a-credit-card/).
  - Overage beyond the trial config is charged pro rata (https://www.kamatera.com/faq/answer/why-was-i-charged-during-my-30-day-free-trial-period/).
  - A $10 refundable deposit may be requested on "unusual billing activity" (https://www.kamatera.com/faq/answer/why-am-i-being-asked-to-make-a-deposit-during-my-30-day-free-trial-period/).
  - Trial page: "1 cloud server, any configuration valued up to $100 … Free trial requires valid credit card" (https://www.kamatera.com/free-trial/).
- **Honest use:** a free 30-day CPU box, e.g. 8 vCPU/16 GB, as a reproducible "official-check-like" runner or for parallel evaluation of the CPU scorer. Not for LLM training.
- **Risks:**
  - Card on file with auto-charge after day 30, so put a calendar reminder to terminate the server.
  - The case says *no registration of financial accounts* (§12). A cloud signup isn't that, but avoid mixing it into the submission.
  - It adds nothing the team's laptops can't already do for CPU work.

**Verdict:** Kaggle T4×2 for any GPU work. Kamatera only as an optional spare CPU box, and it is not needed.

CPU-only LLM inference is not a substitute. llama.cpp on 4 threads does about 290 tok/s prompt processing for a ~1B Q4 model (https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/threads-and-size/). A single 5k-token ledger page then costs about 17 s of prefill, so an LLM episode on CPU blows the 60 s/episode official limit (case.pdf §07) [estimate].

---

## 5. Cheap non-LLM learned policy (recommended optional demo)

### 5.1 Design

1. **Fixed scan policy.** Read the small tables (invoices, payments, fx, policy) fully, then the ledger, with `limit=50`. Deterministic, at most about 50 steps for 2,000 rows (step cost ≤ −0.1). This is the same as the heuristic baseline.
2. **Candidate generator (the heuristic's rules, loosened).** Emit *all plausible* candidates per error type, with the evidence set each would cite:
   - same entity, total and line items within N days → duplicate;
   - Σ payments allocated < invoice total → partial or missing allocation;
   - FX rate or date ≠ policy-selected rate → FX error;
   - timestamp vs period boundary in the manifest timezone → cutoff error.
3. **Features per candidate**, e.g.:
   - |Δamount| in base currency;
   - Δdays;
   - same `entity_id`;
   - line-item Jaccard;
   - invoice_refs count;
   - payment split count;
   - refund or credit-note present;
   - recurring-amount periodicity score (same amount on a monthly cadence);
   - FX rate deviation in bp;
   - hours from the period boundary in local time;
   - currency ≠ base;
   - number of competing candidates sharing an ID.

   These target exactly the distractors the case mandates: split payments, refunds, legitimate recurring amounts (case.pdf §02).
4. **Learner.** One `LogisticRegression` (interpretable coefficients for the design note) or `HistGradientBoostingClassifier` per error type, trained on train seeds. Labels come from the shared matcher (candidate = TP vs not). This is the dedupe pattern: field-distance features with a learned regularized logistic regression over candidate record pairs (https://docs.dedupe.io/en/latest/how-it-works/Matching-records.html).
5. **Reward-driven tuning.** Choose per-type decision thresholds, plus a global "submit nothing if max score < τ" (clean-company guard), by **maximizing mean episode return on train seeds** with a derivative-free search such as random search or ES. ES is "invariant to … delayed rewards" and "tolerant of extremely long horizons" (https://arxiv.org/abs/1703.03864). This makes the claim "a policy optimized against the env's own terminal reward" literally true, not just "a supervised classifier".
6. **Inference** is plain Python with no GPU and runs in milliseconds, so it can be a fifth policy adapter inside the one-command check without risk.

### 5.2 Pros and cons vs LLM training (for jury perception)

**Pros:**
- It directly shows that **the env is trainable**: reward separates policies, labels and evidence are learnable, and nothing leaks.
- It needs only CPU and fits the 2 CPU / 4 GB / 60 s limits.
- It is deterministic and reproducible from a seed.
- It takes hours, not days.
- Its feature weights double as an **interpretable failure analysis** ("FX-boundary interactions are where the heuristic loses F1").
- Its trajectories use the same `trajectories.jsonl` format.

**Cons:**
- It is not "an LLM that learned to audit".
- Some jurors may see it as "a tuned heuristic".
- The candidate generator still encodes domain rules.
- Gains over a good heuristic may be small or zero. Report that honestly; it still earns the "honest limitations" credit.

**Overfitting and leakage risks:**
- Fit **only** on train seeds. Tune on train via cross-validation. Touch dev **once**.
- Additionally evaluate on fresh seeds from *new parameter combinations*, mirroring the jury's "new combinations and harmless transformations" (case.pdf §06).

**Why not LLM:**
- Training earns 0 points.
- "Pretty UI and unproven training improvements earn no points" (case.pdf §12).
- The organizers advise: "do not spend the main time on model training or UI" (case.pdf §08).
- The LLM path is about 15–30 engineer-hours with a high chance of a null or negative result.

---

## 6. Honest controlled comparison protocol

1. **Splits.**
   - Train seeds: the 10 published plus any number of extra train seeds generated from *train configs*.
   - Dev: the 10 published dev episodes.
   - Held-out test: ≥ 100 fresh seeds generated **after** freezing the model, from dev/test configs including unseen parameter combinations, ≥ 25% clean companies and ≥ 25% interacting-error companies (mirrors §06).
   - Record seeds, config hashes and git SHA in `results/manifest.json`.
2. **Same conditions for every policy:**
   - same seeds, same 100-step budget, same `limit`, same env version, same hardware;
   - for the LLM, same decoding parameters and a fixed sampling seed (or greedy);
   - compared policies: heuristic baseline vs learned policy, plus random / flag-all / oracle as reference rows.
3. **Metrics per episode:**
   - TP/FP/FN, P, R, F1, terminal return (the shared reward, shown separately from any experimental reward, §05), steps, wall time;
   - for LLMs also invalid-action count, unknown-ID count, malformed-submit count and context-overflow count.
   - Aggregate with micro-F1 (summing TP/FP/FN) **and** mean per-episode return.
   - Report clean vs erroneous vs interacting strata separately.
4. **Uncertainty:**
   - Use a **paired** bootstrap over episodes (same seed for both policies) for Δreturn and ΔF1: `scipy.stats.bootstrap(..., paired=True, method='BCa', n_resamples=9999)` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
   - Report a 95% CI and the fraction of episodes won, tied or lost.
   - Optionally add a Wilcoxon signed-rank test.
   - For stochastic learners, train ≥ 3–5 seeds and report a CI across training seeds too. Point estimates over few runs mislead, so report interval estimates and robust aggregates such as IQM (https://arxiv.org/abs/2108.13264).
5. **Claim rule.** Say "improves on held-out seeds" only if the 95% CI of Δreturn excludes 0. Otherwise write "no significant difference (Δ = x, 95% CI [a, b], n = 100)". Never tune on the test seeds, and say how many configurations were tried.
6. **Paired robustness check** (mirrors §06, 20 paired checks): rerun on order-permuted and ID-renamed versions of the same test seeds. A learned policy must keep its return. This catches models that learned ID patterns or row order.

---

## RECOMMENDATION

**Train at all?** Only after the mandatory core is complete: contract and isolation, all 4 error types, 2 interacting scenarios, clean and distractor companies, matcher and attack tests, 10+10 episodes, trajectories, the one-command check, and the design note. Hard cut-off: start optional work **no later than Sat 22 Nov ~12:00 Astana**, or skip it. It earns no points directly, and the case says not to spend main time on training.

**Which option.**
1. **Primary: feature-based learned candidate scorer** with reward-tuned thresholds (§5), CPU-only, evaluated with the §6 protocol against the heuristic on dev plus ≥ 100 fresh held-out seeds. Present it as "evidence the environment is learnable and the reward is informative", not as a better auditor.
2. **Optional, only if ≥ 6 h remain afterwards: zero-shot LLM probe, no training.** Run `Qwen3-4B-Instruct-2507` (or `xLAM-2-3b-fc-r`, noting its NC license) on Kaggle T4 via vLLM, with:
   - per-step JSON-schema constrained decoding;
   - ID-enum whitelisting;
   - a sliding-window memory;
   - a pre-submit validator;
   - 10 *small* dev episodes (200–400 ledger rows).

   Report invalid-ID, malformed-JSON and overflow rates and P/R/F1/return next to the heuristic. That feeds the "measured failure analysis" criterion. It must live outside the official check (a separate `demo/llm_probe/` with its own requirements).
3. **Do not attempt SFT+GRPO of an LLM** before the deadline. Estimated 6–8 T4 GPU-h plus 15–30 engineer-hours. The cold-start reward is about −1 for almost all rollouts, so SFT on oracle trajectories is required first (TRL Wordle evidence). The context explosion (§1.2) and Echo-Trap instability (RAGEN) make a null result likely.

**Which compute.**
- Laptop CPU for the scorer.
- Kaggle T4×2 (free; about 30 GPU-h/week; 12-h sessions; scriptable with `kaggle kernels push/status/output`; P100 broken with current PyTorch) only for the optional LLM probe.
- **Not Kamatera.** It has no GPUs; the trial is one ≤ $100 CPU server for 30 days, needs a credit card and auto-charges afterwards.

**Estimated hours.**

| Item | Engineer-hours | Compute |
|---|---|---|
| Candidate features + learner + threshold search | 3–4 | < 10 CPU-min |
| Evaluation harness (paired bootstrap, strata, permutation/ID-rename check) + write-up | 1.5–2 | < 10 CPU-min for ~200 episodes |
| *Optional* LLM probe (Kaggle setup, harness, 10 episodes, failure table) | 4–6 | 1–2 T4-h |
| *Not recommended* SFT+GRPO | 15–30 | 6–8+ T4-h, high variance |

**Risks and mitigations.**
1. Training work steals time from the scored core: enforce the Saturday-noon cut-off.
2. Leakage: the learner may only read truth on train seeds, in training mode, through the matcher. Assert in a test that the policy module never imports grader or truth paths.
3. Overfitting to seeds or ID patterns: held-out fresh seeds, the permutation/ID-rename check, and a single look at test.
4. Null result: report it as such with CIs. That is "honest limitations", which is scored.
5. Kaggle risks: quota or phone verification, P100 incompatibility, pip changes. Pin versions, pick T4×2 explicitly, and keep everything optional and out of the official check.
6. If you do run the LLM probe, xLAM-2's CC-BY-NC license and Qwen3.5's nightly-stack and hybrid-kernel risk on T4 both argue for Qwen3-4B-Instruct-2507 as the default.

---

## Sources

Case: `research/case.pdf` (Genesis Hackathon FinTech case, §01–§13).

1. BFCL leaderboard (updated 2026-04-12): https://gorilla.cs.berkeley.edu/leaderboard.html, data: https://gorilla.cs.berkeley.edu/data_overall.csv
2. TinyLLM small-model agentic eval: https://arxiv.org/abs/2511.22138
3. PROVE (GRPO for multi-step tool use, Qwen3-4B): https://arxiv.org/abs/2606.03892
4. Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B
5. xLAM-2 model card: https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r
6. SmolLM3-3B card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
7. Qwen3 configs: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json, https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json
8. TRL OpenEnv / environment_factory: https://huggingface.co/docs/trl/en/openenv
9. TRL GRPOTrainer docs: https://huggingface.co/docs/trl/en/grpo_trainer
10. OpenEnv Wordle GRPO tutorial (Colab, Qwen3-1.7B config): https://huggingface.co/docs/openenv/tutorials/wordle-grpo
11. Unsloth memory-efficient RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl
12. Unsloth notebooks index (Kaggle T4 GRPO): https://unsloth.ai/docs/get-started/unsloth-notebooks
13. Unsloth fp16 vs bf16 for RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/fp16-vs-bf16-for-rl
14. OpenPipe ART: https://github.com/OpenPipe/ART
15. prime-rl: https://github.com/PrimeIntellect-ai/prime-rl
16. RAGEN / StarPO: https://arxiv.org/html/2504.20073v2
17. Agent-R1: https://github.com/0russwest0/Agent-R1
18. rLLM: https://github.com/rllm-org/rllm
19. SkyRL hardware (secondary): https://www.spheron.network/blog/deploy-skyrl-on-gpu-cloud-train-coding-agents-with-rl-on-h100/
20. vLLM GPU requirements: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
21. vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
22. XGrammar: https://arxiv.org/abs/2411.15100; XGrammar-2: https://arxiv.org/pdf/2601.04426v2, https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation
23. llama.cpp GBNF / JSON schema: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
24. Let Me Speak Freely (format restrictions): https://arxiv.org/abs/2408.02442
25. Tool hallucination, closed-world resolution: https://arxiv.org/pdf/2609.19425; Reasoning Trap summary: https://www.beri.net/article/reasoning-trap-tool-hallucination-enterprise-agents
26. NVIDIA T4 specs: https://www.nvidia.com/en-us/data-center/tesla-t4/
27. vLLM T4 throughput (secondary): https://markaicode.com/benchmarks/vllm-benchmark/
28. llama.cpp CPU throughput (Arm learning path): https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/threads-and-size/
29. FLA CI hardware: https://pypi.org/project/rwkv-fla/0.7.202506150816/
30. Kaggle API kernels: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md; metadata: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
31. Kaggle P100 / PyTorch issue: https://github.com/Kaggle/docker-python/issues/1546
32. Kaggle quota/session (secondary, quoting Kaggle docs): https://fastero.com/blog/kaggle-notebooks, https://aifree.dev/offer/kaggle-free-gpu-tpu
33. Kamatera trial terms: https://www.kamatera.com/faq/answer/what-are-the-terms-of-the-free-trial/, https://www.kamatera.com/free-trial/, https://www.kamatera.com/faq/answer/can-i-sign-up-for-the-free-trial-without-a-credit-card/, https://www.kamatera.com/faq/answer/why-was-i-charged-during-my-30-day-free-trial-period/, https://www.kamatera.com/faq/answer/why-am-i-being-asked-to-make-a-deposit-during-my-30-day-free-trial-period/
34. Kamatera pricing (CPU-only configurator): https://www.kamatera.com/pricing/
35. gpu.nz listing (source of the misattributed RTX 3090 price): https://sourceforge.net/software/product/gpu.nz/; Kamatera GPU "Not confirmed": https://googiehost.com/blog/best-managed-vps-hosting-for-ai-agents/
36. dedupe (learned logistic regression over field distances): https://docs.dedupe.io/en/latest/how-it-works/Matching-records.html
37. Evolution Strategies: https://arxiv.org/abs/1703.03864
38. SciPy bootstrap (paired, BCa): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
39. Deep RL at the Edge of the Statistical Precipice: https://arxiv.org/abs/2108.13264
40. Qwen3.5 GRPO on T4 guide (secondary): https://www.besthub.dev/articles/from-zero-to-deployment-a-complete-qwen3-5-fine-tuning-guide-869c8e1c888c
