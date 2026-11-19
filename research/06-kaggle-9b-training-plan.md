# 06 — Evaluating and fine-tuning a ≥9B open LLM on Kaggle T4×2

Date: 19 Nov 2026. GPU work can start 21 Nov. Deadline: 22 Nov 2026, 23:59 Astana.
Constraints: Kaggle free GPUs only, about 14 GPU-hours of quota left, no paid APIs, models of at least 9B parameters. Training earns 0 points, must stay out of the official offline check, and any claim of improvement needs a controlled comparison on held-out seeds (case.pdf §01, §07, §12).

Labels, as in `03-training-feasibility.md`:
- **[measured]**: I read it from a primary artifact: a config, CSV, notebook output, changelog or issue log.
- **[cited]**: someone else's number, with a URL.
- **[estimate]**: my own arithmetic, with the assumptions shown. Every [estimate] here is replaced by a number measured in session S0 before any money-equivalent quota is spent on the main runs.

---

## RECOMMENDATION

| Decision | Choice | Why, in one line |
|---|---|---|
| Model | **Qwen3-14B**: Apache-2.0, dense, standard GQA attention. Run it **non-thinking** | It is the only ≥9B model with (a) a public Kaggle-T4 QLoRA run in Unsloth's own notebooks, (b) standard attention that every T4 kernel path supports, and (c) official AWQ and GGUF quants. Its BFCL multi-turn score is mediocre, 34.75% FC, and that is expected. |
| Optional reference arm | Qwen3.5-9B, **zero-shot only**, run last if quota allows | Vendor BFCL-V4 66.1 and TAU2 79.1 are much better on paper. On T4 it **cannot be trained** in budget: it needs fp32 because fp16 produces NaNs, and a 4B model already takes about 2 h for 60 steps. Its inference on sm75 was only fixed in vLLM on 9 Nov 2026. |
| Inference stack | **vLLM 0.11.2**, the version Unsloth pins for T4. Qwen3-14B-AWQ in fp16, **one engine per GPU** (data-parallel over seeds, no tensor parallelism), prefix caching, JSON-schema structured outputs with ID enums, LoRA served at runtime | This exact vLLM pin is what Unsloth's Kaggle T4 notebooks install. Two independent engines avoid NCCL on non-NVLink T4s. Fallback: build llama.cpp for sm_75. |
| Training method | **RFT / filtered behaviour cloning**: QLoRA SFT on per-step (prompt → action) samples from the truth-free heuristic policy on train seeds, keeping only episodes whose shared terminal return is ≥ threshold | It uses env reward as the filter, as in expert iteration (STaR, RFT, ReST-EM). It is feasible in about 3 GPU-h. |
| GRPO | **Not attempted** | At 14B on a T4, GRPO is roughly 1 optimizer step per 30–40 min for our episode shape. Cold-start reward variance is about zero. See §3.3. |
| Eval | Base vs SFT on the **same 30 held-out seeds plus 10 order/ID-permuted twins**, same harness and greedy decoding, paired bootstrap | The claim rule is in §6. |
| GPU budget | **≈ 7.25 h planned + 2.5 h re-run reserve + 2 h optional Qwen3.5 arm + 2 h buffer = 13.75 h** | Table in §7. |

Expected outcome [estimate]: SFT beats base by +0.1 to +0.5 mean return. Both stay **below the heuristic**, because behaviour cloning is capped near its expert. There is a real chance, which I put at 30–50%, that the 95% CI of the difference still includes 0 with n=30. If so, we report exactly that.

---

## 1. Candidate models ≥9B on Kaggle T4×2

Hardware facts:
- T4: sm_75, 16 GB (14.74 GiB usable), no bf16, no FlashAttention-2.
- Kaggle's current image is Colab-based, with Python 3.13 and torch cu128 [measured: https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl, https://github.com/Kaggle/docker-python/issues/1546].
- P100 is retired, and `kernels push` warns that the server substitutes another GPU [measured: https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md].

BFCL v4 numbers below come from the leaderboard CSV (https://gorilla.cs.berkeley.edu/data_overall.csv), which I filtered myself [measured].

| Model | Params / arch | License | BFCL v4 overall / **multi-turn** (mode) | Other agentic evidence | T4 inference | T4 QLoRA | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen3-14B** | 14.8B dense; 40 layers, 8 KV heads × 128 → **160 KiB KV/token** fp16; 40,960 positions [measured, config.json] | Apache-2.0 | 41.03% / **34.75%** (FC); 37.77% / 26.13% (Prompt) | — | AWQ 9.99 GB, GGUF Q4_K_M 9.0 GB [measured, HF trees]. AWQ is supported on Turing in vLLM [cited, https://docs.vllm.ai/en/stable/features/quantization/] | **Proven on Kaggle T4:** Unsloth notebook loads at 11.9 GB reserved, 14.51 GB peak at seq 2048 × bs 2, 30 steps in 41.3 min [measured, notebook output: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(14B)-Reasoning-Conversational.ipynb] | **PICK** |
| Qwen3-8B (for contrast) | 8.2B | Apache-2.0 | 42.57% / 41.75% (FC) | — | easy | easy | Excluded by the ≥9B rule. It scores *higher* multi-turn than 14B, so "bigger" is not a sure win |
| **Qwen3.5-9B** (exists, Feb 2026) | ~9B **hybrid**: 24 Gated-DeltaNet + 8 full-attn layers, vision tower, 262k ctx [measured, config.json] | Apache-2.0 | not on leaderboard | Vendor: BFCL-V4 **66.1**, TAU2 **79.1** (https://huggingface.co/Qwen/Qwen3.5-9B) | Needs vLLM ≥0.17. The sm75 RMSNorm/FlashInfer-JIT crash was fixed in PR #66289, merged 9 Nov 2026 [measured: https://github.com/vllm-project/vllm/issues/66288]. A different stack (transformers + FLA) fails on T4 compiling the bf16 GDN kernel [cited: https://github.com/strands-labs/strands-decider/issues/113]. llama.cpp runs Qwen3.5 on sm_75 [cited: https://github.com/ggml-org/llama.cpp/issues/29513] | **Effectively no.** bf16 LoRA needs 22 GB, and "QLoRA not recommended" for Qwen3.5 [cited: https://unsloth.ai/docs/models/qwen3.5/fine-tune]. On T4 "Qwen3.5 gives NaNs in pure float16, so Unsloth trains it in float32": the **4B** model took **~2 h for 60 steps** [measured: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_5_(4B)-Decision.ipynb] | Optional zero-shot reference arm only |
| Qwen3.5 other sizes | 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B. **No ~12–14B** [measured, HF API]. Qwen3.6 has only 27B and 35B-A3B; Qwen3.8 has 27B (qwen3_5 arch), Flash-Next and 2.4T | | | | | Qwen3.8-27B QLoRA works only across both T4s at seq **1024** [measured: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3.8_(27B)-Conversational.ipynb] | Too big for our context |
| Gemma 3 12B | dense | Gemma terms | 30.43% / **5.75%** (Prompt) | — | ok | ok | Reject: multi-turn ≈ 0 |
| Gemma 4 12B (May 2026) | 48 layers; 40 sliding (1,024) + 8 global with **head_dim 512** [measured, config.json] | Apache-2.0 | not on leaderboard | Vendor Tau2 69.0% | **vLLM fails on Turing.** The global layers' Triton attention needs 98,304 B shared memory vs 65,536 B. The fix PR #39018 is **unmerged** [measured: https://github.com/vllm-project/vllm/issues/57712] | Unsloth's Kaggle notebook was run on an A100 in 16-bit, 22.4 GB [measured: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Gemma4_(12B)_Text.ipynb] | Reject for T4 |
| GLM-4-9B-0414 | 9B dense | MIT | not on leaderboard | none found for multi-turn | ok | probably ok | Reject: no evidence |
| Mistral NeMo 12B (2407) | 12B dense | Apache-2.0 | not on leaderboard | none | ok | Unsloth Kaggle notebook exists | Reject: old (Jul 2024), no tool-use evidence |
| Ministral-3-14B-Instruct-2512 | 14B, vision | Apache-2.0 | not on leaderboard | none found | likely ok | unknown | Unverified alternate |
| Phi-4 | 14B dense | MIT | 28.79% / **3.88%** (Prompt) | — | ok | Kaggle T4 GRPO notebook exists | Reject: multi-turn ≈ 0 |
| Llama 3.1 8B | 8B (<9B) | Llama 3 | 25.83% / 11.12% | — | ok | ok | Reject |
| xLAM-2-8b-fc-r | 8B (<9B) | **CC-BY-NC-4.0** | 46.68% / **70.00%** | — | ok | ok | Excluded by the 9B rule and the NC license. Note it as the strongest small multi-turn model |
| xLAM-2-32b-fc-r | 32B | CC-BY-NC-4.0 | 54.66% / 69.50% | — | Q4 is about 19 GB, so it needs both T4s split by layer | no | Not trainable here |
| gpt-oss-20b | 21B MoE, 3.6B active | Apache-2.0 | not on leaderboard | — | MXFP4 Marlin is **not** supported on Turing in vLLM [cited, vLLM quantization table]. llama.cpp is ok | Kaggle T4 notebook: 12.8 GB at load, seq **1024** only [measured: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-gpt-oss-(20B)-Fine-tuning.ipynb] | Reject: too little headroom for 3.5k-token samples; harmony format adds risk |

**Skeptical note on the pick.** Qwen3-14B's 34.75% BFCL multi-turn is roughly a third of tasks solved. Our task also needs exact minor-unit arithmetic and cross-page joins. Expect the **base** model to be far below the heuristic. The interesting comparison is base vs SFT, not LLM vs heuristic.

---

## 2. Inference on T4×2

### 2.1 What actually works on sm_75 (Nov 2026)

- **vLLM requirements:** compute capability ≥ 7.5 [cited: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/]. FlashAttention requires SM80+. On Turing the only auto-selected backends are `TRITON_ATTN` and `FLEX_ATTENTION` [cited: https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/triton_attn/].
- **Prefill on T4 is slow.** Since Triton 3.3, Triton emits FMA instead of tensor-core MMA on SM<80. T4 prefill is "up to 15× slower than FlashInfer" [measured: https://github.com/vllm-project/vllm/issues/60821]:
  - Qwen3-0.6B: 2,048-token prefill takes 0.68 s; 8,000 tokens take 9.5 s.
  - Qwen3-4B: an 8k prefill takes **26.2 s**.
  - The fix that restores FlashInfer on SM75 (PR #55380) is **open/unmerged**.
  - **Implication:** keep every prompt ≤ ~3.5k tokens. Long-context agent loops on T4 vLLM are not viable.
- **Quantization on Turing:** AWQ ✅, GPTQ ✅, bitsandbytes ✅, GGUF ✅. Marlin works on Turing except MXFP4 [cited: https://docs.vllm.ai/en/stable/features/quantization/].
- **Version pin.** Unsloth's Kaggle notebooks install `vllm==0.11.2` when `nvidia-smi` shows a T4. Their comment says "Keep the machine's torch: upgrading it can pull a CUDA build the driver cannot load" [measured: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Phi_4_(14B)-GRPO.ipynb, https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO-Tool_Use.ipynb]. The latest vLLM is 0.34.0 (16 Nov 2026) [measured: GitHub releases/PyPI], but nothing shows it running on Kaggle's driver. **Use 0.11.2 for Qwen3-14B.** Use 0.34.0 only for the optional Qwen3.5 arm, in its own session.
- **API change:** `guided_json` and similar fields were removed in v0.12.0 and replaced by `structured_outputs={"json": …}` [cited: https://github.com/vllm-project/vllm/blob/v0.14.0rc2/docs/features/structured_outputs.md]. On 0.11.2, code should try `StructuredOutputsParams` and fall back to `GuidedDecodingParams`, and S0 confirms which one works.
- **Tensor parallelism vs replicas.** TP=2 has been run on 2×T4 (Qwen3.5-2B in #66288), but T4s on Kaggle have no NVLink. 14B AWQ (10 GB) fits on **one** T4, so run **two independent engines**, one per GPU, each taking half of the seeds. That avoids NCCL, and for a prefill-bound workload it is about 2× the throughput of TP=2.
- **Known SM75 vLLM bugs:**
  - MRV2 full-CUDA-graph replay crashes with ≥2 concurrent sequences when speculative decoding is on [cited: https://github.com/vllm-project/vllm/issues/61233]. **Don't enable spec-decode.**
- **llama.cpp fallback:**
  - It has tensor-core flash-attention MMA kernels for sm_75 (`fattn-mma-f16`), with a known suboptimal 32-token Q-tile [cited: https://github.com/ggml-org/llama.cpp/issues/28761].
  - T4 scoreboard, Llama-2-7B Q4_0: **pp512 1,310 t/s, tg128 44 t/s** with FA [cited: https://github.com/ggml-org/llama.cpp/discussions/15013].
  - Scaling to 14B Q4_K_M gives about 550–600 t/s prefill and about 20 t/s decode per T4 [estimate]. That is likely **2–3× faster prefill than vLLM's FMA-bound Triton attention** at our prompt sizes.
  - Costs: you must build it yourself (about 10–20 min on Kaggle's 4 CPUs, `-DCMAKE_CUDA_ARCHITECTURES=75`, which also avoids the PTX-JIT issues in [https://github.com/ggml-org/llama.cpp/issues/31503] and [#29513]), and the LoRA must be converted to GGUF.
  - It is the fallback, not the default, because the vLLM pin needs no build and serves PEFT LoRA directly.

### 2.2 Memory and context, Qwen3-14B-AWQ on one T4 [estimate]

- Weights: about 10.0 GB. vLLM overhead plus activations: about 1.0–1.5 GB. KV budget at `gpu_memory_utilization=0.92`: about 2.5–3.0 GB.
- That is about **16k–19k tokens of KV**, i.e. 4 concurrent sequences of 4k at 160 KiB/token.
- `max_model_len=4096`.

### 2.3 Throughput per T4 at our prompt shape [estimate, to be replaced by S0]

Shape: about 3.4k-token prompt, of which about 1.1k is a static prefix-cache hit and about 2.3k is new; output about 150 tokens; greedy.

| Component | Estimate | Basis |
|---|---|---|
| Prefill of 2.3k new tokens | 6–9 s | Linear: 2 × 14e9 × 2.3k ≈ 64 TFLOP at ~15–20 TFLOP/s ≈ 3.5–4.5 s. Attention, FMA-only: scaled from #60821 (0.6B, 16 heads, 28 layers: 0.68 s at 2k) by 40 heads × 40 layers and a longer KV ≈ 2.5–4 s |
| Decode, 150 tokens | single stream ≈ 7–9 s (≈ 45–60 ms/token on 10 GB of weights at 320 GB/s); at 4 concurrent ≈ 2–3 s effective per sequence | Bandwidth bound |
| **Effective per env step** | **≈ 10 s per GPU at concurrency 4** (≈ 15 s single-stream) | — |
| Episode (≤ 25 LLM steps) | **≈ 4.2 min per GPU**, so ≈ 2.1 min per episode with both GPUs | — |
| Episodes per session-hour | ≈ 28 with both GPUs; ≈ 14 with one | — |

### 2.4 Structured decoding to stop invented IDs and malformed JSON

- Build a JSON schema for each step.
- The ID fields (`get_record.id`, `get_document.doc_id`, submit `affected_ids`/`evidence_ids`) are `enum`s over **IDs the harness has actually seen** in this episode.
- `table` and finding `type` are enums. `limit` is an integer from 1 to 20. `calculator.expression` is a regex `^[0-9.+\-*/() ]{1,200}$`. Use `maxItems` 30/10 and `maxLength` 300 for the explanation.
- The model then cannot emit an unseen ID or invalid JSON. Malformed-JSON and invented-ID rates should both be exactly 0, which is a useful, honest table row.
- A remaining risk is that structured outputs can hurt reasoning [cited in 03: https://arxiv.org/abs/2408.02442]. The schema therefore allows a short `note_add` field before `action`.

---

## 3. Training on T4×2 within about 3–4 GPU-hours

### 3.1 SFT with Unsloth QLoRA on Qwen3-14B: what fits

- **Anchor** [measured, Kaggle T4 notebook output above]:
  - Load reserves 11.90 GB.
  - Peak is **14.51 GB** at `max_seq_length=2048`, batch 2 (4,096 tokens per micro-step).
  - 30 optimizer steps of 8 sequences take 2,476 s, so **≈ 10.3 s per sequence** at ≤ 2,048 tokens on one T4.
- Most of the load memory is not the 4-bit body. Embedding plus LM head (151,936 × 5,120 × 2 B × 2, untied) is **3.1 GB** [estimate from config].
- **Our setting:** `max_seq_length=3584`, batch 1, so ≤ 3,584 tokens per micro-step. That is about the same tokens as the anchor's 4,096, so expect a peak of about 14.3–14.6 GB. **Tight.** S0 measures it.
- **Mitigations, in this order:**
  1. LoRA r=16 instead of 32.
  2. Cap the prompt at p95 ≤ 3,300 tokens.
  3. Use `offload_embedding=True` (Unsloth uses it in its Qwen3.8 recipe [cited: https://unsloth.ai/docs/models/qwen3.8/train]; unverified for FastLanguageModel/Qwen3).
  4. Show 15 rows per page instead of 20.
- **Speed:** ≈ 18–22 s per 3.4k-token sample on one T4 [estimate, scaled from the anchor]. **3 h ≈ 500–600 samples**, ≈ 70 optimizer steps at GA=8.
- **DDP:** Unsloth supports it via `torchrun --nproc_per_node 2` [cited: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth]. I still plan **single-GPU SFT in parallel with base eval on the other GPU**, which uses the session better and removes DDP risk.
- **LoRA size:** r=16 on all 7 linear projections × 40 layers is about 64M parameters, ≈ 128 MB fp16 [estimate].
- **Loss on actions only:** use the prompt–completion dataset format. TRL computes loss on the completion. Qwen3 templates are auto-patched for assistant-only loss [cited: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md].
- **Template parity:** prompts are rendered **once** by our own function (`tokenizer.apply_chat_template(..., enable_thinking=False)`). The same string goes to SFT and to vLLM's `generate()`, which rules out train/infer template drift.

### 3.2 What data: rejection-sampled heuristic trajectories (RFT)

1. **Expert = the truth-free heuristic policy.** Run it with the *LLM's* constraints: page limit 20, ≤ 25 steps, same observation renderer. At each step it emits `{note_add, action}`. Its notes record what it has learned from pages already observed, e.g. `"dup? INV-0412~INV-0419 same entity, total 1250000 JPY, Δ2d"`.
   - **Do not use the oracle as the expert.** It acts on hidden truth, so cloning it teaches the model to "know" errors without evidence. At test time that turns into confident false accusations.
2. **Filter (the RL part).** Keep a train-seed episode only if its **shared terminal return** is ≥ 0.5, or ≥ 0.9 for clean companies. That is filtered behaviour cloning, or one round of expert iteration, using the env's own reward.
   - Optional hindsight relabel: drop submitted findings the matcher marks FP. Report this as a separate variant if used.
3. **Split steps into samples.** Each step is one sample: (compacted prompt at step t) → (target JSON). Keep every `submit` step. Subsample navigation steps so the set reaches about 550 samples, roughly 22 episodes × 25 steps.
4. **Precedents:**
   - STaR: https://arxiv.org/abs/2203.14465
   - RFT: https://arxiv.org/abs/2308.01825
   - ReST-EM: https://arxiv.org/abs/2312.06585
   - FireAct, agent fine-tuning on a few hundred filtered trajectories: https://arxiv.org/abs/2310.05915
   - AgentTuning: https://arxiv.org/abs/2310.12823
   - TRL's Wordle example: SFT on solver games took Qwen3-1.7B from 0/30 to 10/30 wins before any GRPO (https://huggingface.co/docs/trl/en/openenv).

**Honest framing for the write-up:** "SFT on reward-filtered expert trajectories (expert iteration, 1 round)". It is not "RL", and not "learned to beat the heuristic".

### 3.3 GRPO at 9–14B on T4: infeasible in this budget

- **Unsloth's Phi-4 (14B) GRPO on a Kaggle T4** [measured, notebook log]:
  - `max_seq_length=512`, vLLM KV cache 1.24 GB, generation about 10 tok/s single-stream.
  - That covers toy single-turn GSM8K, not 25-turn episodes.
- **Our rollout:** about 25 turns × 3.4k tokens, so about 85k processed tokens per rollout.
- **One GRPO step with 4 rollouts** [estimate]:
  - Generation: about 4 × 25 × 10 s ≈ 15 min, concurrency only partly helping.
  - Backward over about 340k tokens at about 170 tok/s ≈ 33 min.
  - Total **≈ 40–50 min per optimizer step**, so about 8 steps in the whole remaining quota.
- **Group variance is near zero at cold start.** Untrained rollouts return about −1 on erroneous books, so most groups give no advantage signal, as TRL's own Wordle note says (https://huggingface.co/docs/trl/en/openenv).
- **TRL's multi-turn `environment_factory`** needs a prefix-preserving chat (no compaction) and caps the whole multi-turn conversation by `max_completion_length` [cited: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md]. Our compaction breaks that assumption.
- **Unsloth's T4 tool-use GRPO notebook does not use vLLM at all:** "vllm needs transformers 5 for tool calling, which no vllm supports on a T4" [measured: Kaggle-Qwen3_(4B)-GRPO-Tool_Use.ipynb].
- **Verdict:** no GRPO. Say so in the README limitations, citing these numbers.

---

## 4. Episode design so the LLM fits in context

All of this is **agent-side scaffolding** in `experiments/kaggle_llm/harness.py`. The env contract and observations are unchanged, and the base and SFT arms use identical scaffolding.

| Prompt block, in order (static first for prefix caching) | Tokens [estimate; prior measurement was 74 tok/row as CSV] |
|---|---|
| System: rules, action list, output contract | ≈ 700 |
| Task + manifest + policy summary (from reset obs) | ≈ 400 |
| Coverage tracker (harness bookkeeping only): e.g. `ledger rows 1–140/300 read, next cursor=…; invoices done; fx done` | ≈ 150 |
| NOTES: model-written `note_add` lines, rolling, ≤ 30 lines. **Any ID-like token not in the seen-ID set is stripped** before re-injection | ≤ 600 |
| Last observation only, rendered as CSV (≤ 20 rows), or one document | ≤ 1,550 |
| Last action + bounded env error, if any | ≈ 100 |
| **Total prompt** | **≤ ≈ 3,500** (gate: p95 ≤ 3,300) |
| Target: `{"note_add":[…≤3 lines], "action":{…}}` | ≈ 80–200 |

Rules:
- **Eval episode configs:** "small" dev/test configs, 150–300 ledger rows, 20–40 docs, 0–4 errors. A full read is about 15 ledger pages + 2 invoices + 2 payments + 1 fx ≈ 20 list steps, plus a submit.
- The env's 100-step cap is unchanged. The harness gives the LLM **≤ 25 env steps**. At step 23 it switches the schema to **submit-only**, so termination is guaranteed.
- **Submit guard:**
  1. Validate the findings locally (schema, limits, all IDs observed) *before* `env.submit`.
  2. On failure, run one repair generation, which costs no env step.
  3. If it still fails, submit `findings: []`. That gives silence, never the −1 malformed penalty.
- **Limitation, state it:** the LLM is only evaluated on small configs. The 2,000-row official sizes would need about 100 pages and are out of reach for a T4-hosted 14B within 25–100 steps.

### Episode budget [estimate, recompute after S0]

| Workload | Episodes | GPU layout | Wall time |
|---|---|---|---|
| Base eval: 30 held-out + 10 permuted twins | 40 | 1 GPU (other GPU trains) | 40 × 4.2 min ≈ **2.8 h** |
| SFT eval: same 40 | 40 | 2 GPUs | ≈ **1.4 h** |
| Held-in replay sanity (SFT on 4 train seeds) | 4 | 2 GPUs | ≈ 0.15 h |
| Teacher-forced action agreement on 300 held-out expert steps, base and SFT (prefill + ≤ 200 tokens, batched) | — | 2 GPUs | ≈ 0.2 h |

Formula for after S0: `episodes = floor(minutes_available × n_gpus / (steps_per_ep × sec_per_step / 60))`. If S0 shows ≥ 2× my estimate, drop to 20 held-out + 10 twins. **Never go below 20 held-out seeds.**

---

## 5. Kaggle mechanics (verified against the CLI repo, Nov 2026)

- **CLI:** `kaggle` 2.2.5 (PyPI, 22 Oct 2026). The repo is now `Kaggle/kaggle-cli`. `kaggle quota` exists since **2.2.1** and prints `resource,used,remaining,total,refreshAt` with `-v` [measured: https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md, https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md].
- **Auth, in the CLI's lookup order:**
  1. `KAGGLE_API_TOKEN` env var, or `~/.kaggle/access_token`.
  2. Legacy `kaggle.json` / `KAGGLE_USERNAME` + `KAGGLE_KEY`.
  3. OAuth via `kaggle auth login`.

  Generate the token at https://www.kaggle.com/settings/api with "Generate New Token". "`KAGGLE_API_TOKEN` can authenticate **without a legacy username/key pair** when the token introspection succeeds" [measured: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md, https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md].
  - The `KGAT_` prefix and the "CLI ≥ 1.8.0" threshold come only from third-party guides. 1.8.3 fixed a `KAGGLE_API_TOKEN` KeyError, so **use ≥ 2.2.5**.
  - The **username is still needed in `kernel-metadata.json` `id`** (`<user>/<slug>`). Get it from `kaggle kernels list -m`, or from the profile URL.
- **GPU and internet** require a phone-verified account [secondary, cited in 03].
- **Accelerators:** `machine_shape` / `--accelerator NvidiaTeslaT4` = **GPU T4 ×2** [measured: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md, https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md].
- **Push options:** `kaggle kernels push -t <seconds>` sets the maximum run time. **Always set it**; it is the quota circuit-breaker. `--no-run` saves without running.
- **Downloading outputs:** `kaggle kernels output <owner/slug> -p dir -o`. It has streamed, resumable downloads with a path-traversal guard since 2.2.5. `kaggle kernels status <owner/slug>` reports the status. `logs` exists (changelog mentions version handling for `logs`); I did not verify its flags.
- **Datasets:** `kaggle datasets create -p dir [-r zip]` creates a private dataset by default. `kaggle datasets version -p dir -m msg -r zip` adds a version. The default `--dir-mode` is `skip`, so **subfolders are silently dropped unless you pass `-r zip` or `tar`** [measured: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md].
- **Mount path:** inputs appear under `/kaggle/input/…`. Glob for the wheel instead of hard-coding the path.
- **Session limits** (secondary sources): 12 h per session; about 30 GPU-h per week, shown in settings; `/kaggle/working` ≈ 20 GB is persisted as output.
  - Download model weights to `/tmp`, not `/kaggle/working`, or they become 10 GB of output.
  - Chain sessions with `kernel_sources: ["<user>/<prev-kernel>"]`, which mounts the previous run's output (e.g. the adapter).
- **T4×2 quota accounting is unverified.** I found no primary source saying whether a T4×2 session consumes quota at 1× or 2×. **Measure it:** run `kaggle quota -v` before and after S0. If it is 2×, cut the plan to the "minimal" column in §7.
- **Weights:** with internet on, pull from the HF Hub (`Qwen/Qwen3-14B-AWQ`, `unsloth/Qwen3-14B-unsloth-bnb-4bit`; both public, no token). Kaggle also mirrors HF models (e.g. https://www.kaggle.com/refs/hf-model/Qwen/Qwen3.5-9B), which could go in `model_sources`, but this is unverified for these repos.

### 5.1 Files

`kaggle/pkg/dataset-metadata.json`:
```json
{"title": "genesis-audit-gym-pkg", "id": "<user>/genesis-audit-gym-pkg", "licenses": [{"name": "CC0-1.0"}]}
```
Contents:
- `genesis_audit_gym-<ver>-py3-none-any.whl` (env)
- `harness/` (renderer, schema builder, notes validator, runner)
- `seeds/heldout.json` (sealed list + sha256)
- `sft/train.jsonl`, `sft/val.jsonl`

`kaggle/k_smoke/kernel-metadata.json` (the same shape is used for `k_train_eval` and `k_sft_eval`):
```json
{
  "id": "<user>/gag-smoke",
  "title": "gag-smoke",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["<user>/genesis-audit-gym-pkg"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```
For `k_sft_eval`, set `"kernel_sources": ["<user>/gag-train-eval"]` so the adapter from S1 is mounted.

### 5.2 Laptop-side commands

```bash
pipx install 'kaggle>=2.2.5'          # or: uv tool install kaggle
export KAGGLE_API_TOKEN='<paste; never commit>'
kaggle quota -v                       # record BEFORE
kaggle datasets create  -p kaggle/pkg -r zip            # first time
kaggle datasets version -p kaggle/pkg -r zip -m "v2"    # later
kaggle kernels push -p kaggle/k_smoke --accelerator NvidiaTeslaT4 -t 4500
until kaggle kernels status <user>/gag-smoke | grep -qiE 'complete|error|cancel'; do sleep 120; done
kaggle kernels output <user>/gag-smoke -p runs/smoke -o
kaggle quota -v                       # record AFTER, to check 1x vs 2x accounting
```

### 5.3 `run.py` core (S0; S1/S2 reuse it with `pip freeze` pins copied from S0's output)

```python
import subprocess, sys, glob, os, json
def sh(c): print(c, flush=True); subprocess.run(c, shell=True, check=True)
sh("nvidia-smi; python -V; pip freeze | grep -iE 'torch|transformers|vllm|triton' || true")
# Same pins Unsloth uses on Kaggle T4 (Phi-4 GRPO notebook):
sh("pip install -q uv && uv pip install --system -q vllm==0.11.2 bitsandbytes unsloth "
   "&& uv pip install --system -q transformers==4.57.6 && uv pip install --system -q --no-deps trl==0.22.2")
sh(f"pip install -q {glob.glob('/kaggle/input/**/genesis_audit_gym-*.whl', recursive=True)[0]}")
sh("pip freeze > /kaggle/working/freeze.txt")
# One vLLM engine per GPU, half the seeds each:
for g in (0, 1):
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python harness/run_eval.py --arm base --shard {g}/2 "
                     f"--out /kaggle/working/base_g{g}", shell=True)
```

The harness engine:
```python
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams as SO; so = lambda s: dict(structured_outputs=SO(json=s))
except ImportError:
    from vllm.sampling_params import GuidedDecodingParams as GD; so = lambda s: dict(guided_decoding=GD(json=s))
llm = LLM("Qwen/Qwen3-14B-AWQ", dtype="half", max_model_len=4096, gpu_memory_utilization=0.92,
          enable_prefix_caching=True, enable_lora=(arm == "sft"), max_lora_rank=16, max_loras=1, seed=0)
# Each round: one prompt per active episode -> llm.generate(prompts, [SamplingParams(temperature=0, max_tokens=256, **so(schema_i))],
#             lora_request=LoRARequest("sft", 1, adapter_dir) if arm == "sft" else None)
```

The SFT core:
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
m, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-14B-unsloth-bnb-4bit", max_seq_length=3584, load_in_4bit=True)
m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=32, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
# dataset rows: {"prompt": <exact rendered prompt string>, "completion": <target JSON> + "<|im_end|>"}
SFTTrainer(model=m, processing_class=tok, train_dataset=ds, args=SFTConfig(
    per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=1e-4,
    warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_8bit", logging_steps=1, save_steps=20,
    max_length=3584, output_dir="/kaggle/working/sft", report_to="none", seed=3407)).train()
m.save_pretrained("/kaggle/working/adapter")   # ≈128 MB; PEFT format, loadable by vLLM LoRARequest
```
Add a wall-clock stop callback at 3.0 h that saves the adapter, so a slow run still yields an adapter.

---

## 6. Experiment protocol

**Pre-registration.** Commit `experiments/kaggle_llm/PROTOCOL.md` **before S1**, recording:
- the hypothesis: SFT mean return > base mean return on held-out seeds;
- the primary metric;
- the seed list hash;
- the decoding parameters;
- the claim rule.

**Freeze** env version + git SHA, harness/prompt version hash, `temperature=0`, `max_tokens=256`, 25-step LLM cap, page limit 20, model revision, and AWQ checkpoint revision.

**Seeds:**
- **Train:** train configs only. Used to generate expert data.
- **Validation:** 4 train-config seeds kept out of SFT. Used for the teacher-forced agreement check.
- **Held-out test:** 30 fresh seeds from dev/test configs with unseen parameter combinations: ≥ 8 clean, ≥ 8 with interacting errors, the rest single or multiple errors (mirrors §06's 5 + 5 out of 20). Generate them once, write `seeds/heldout.json` + sha256, and never look at per-seed results before S2 finishes.
- **Robustness:** 10 of the 30 re-run as order-permuted + ID-renamed twins (mirrors §06's paired checks).

**Arms** (same harness, same seeds, same hardware type):
- **R:** heuristic, both full-budget and matched-budget (limit 20, ≤ 25 steps), for reference.
- **A:** Qwen3-14B-AWQ base + scaffold.
- **B:** A + SFT LoRA.
- **C (optional):** Qwen3.5-9B zero-shot.
- Reference rows: random, flag-all, oracle.

**Metrics per episode:**
- shared terminal return (primary), TP/FP/FN, P/R/F1, steps, wall time;
- invalid-action count (bounded env errors), schema-repair count, malformed-submit count (should be 0 by construction), notes-ID strips (a hallucination proxy), forced-submit count;
- **false-accusation rate on clean companies**, which is likely the biggest driver of return.
- Aggregate as micro-F1 and mean return, reported separately for the clean / single-error / interacting strata.

**Statistics:**
- paired BCa bootstrap over the 30 held-out seeds, 9,999 resamples (`scipy.stats.bootstrap(paired=True)`);
- Wilcoxon signed-rank as secondary;
- win/tie/loss counts.
- The twin set is reported separately as the change in return under permutation.

**Claim rule:**
- Write "SFT improves on held-out seeds" **only if** the 95% CI of Δreturn (B−A) excludes 0 **and** the twin-set return of B does not drop by more than its CI.
- Otherwise write: "no significant difference (Δ = x, 95% CI [a, b], n = 30)".
- Report how many training configurations were tried (planned: 1).

**Statistical power** [estimate]: assuming the paired SD of Δreturn ≈ 0.5, n=30 gives SE ≈ 0.09, so the minimum detectable Δ is ≈ 0.18.

**Expected ranges** [estimate; derived from BFCL multi-turn ≈ 35% and the clean-company scoring asymmetry]:

| | Base (A) | SFT (B) | Heuristic (R) |
|---|---|---|---|
| Malformed JSON / invented IDs | 0 / 0 (constrained) | 0 / 0 | — |
| Bounded env errors per episode | 1–5 | 0–2 | 0 |
| Teacher-forced next-action agreement with expert | 20–40% | 70–90% | 100% by definition |
| Clean-company false accusation | 40–80% | 10–40% | low |
| Micro-F1, erroneous books | 0.05–0.30 | 0.20–0.50 | higher than B |
| Mean return | −0.7 to −0.2 | −0.3 to +0.3 | highest |

**If there is no improvement, report:**
1. The CI table.
2. Teacher-forced agreement. If SFT agreement is high but return is flat, the bottleneck is cross-page memory and arithmetic, not imitation.
3. A failure taxonomy over 10 hand-read SFT trajectories:
   - premature submit;
   - notes drift (stripped IDs);
   - wrong `claimed_amount_minor` (arithmetic, currency exponent);
   - missing evidence set;
   - over-flagging clean companies.
4. A note that BC caps performance near the matched-budget heuristic.

All of this feeds the "measured failure analysis" and "honest limitations" criteria (case.pdf §10).

---

## 7. GPU-hour budget (session-hours; must total ≤ 14)

| # | Session (when) | Contents | Cap `-t` | Planned h | Minimal plan if quota counts 2× |
|---|---|---|---|---|---|
| S0 | Calibrate (21 Nov AM) | Install + freeze. vLLM AWQ load on each GPU. 2 base dev episodes per GPU. Measure s/step and peak memory. 8 QLoRA steps at seq 3,584 to measure memory and s/sample. Confirm the structured-outputs param. Run `kaggle quota` before and after | 4,500 s | **1.25** | 1.0 |
| S1 | Train ∥ base eval (21 Nov PM, runs unattended) | GPU0: SFT ≈ 550 samples, 3.0 h wall stop. GPU1: base eval, 30 held-out + 10 twins | 14,400 s | **4.0** | 3.0 (20 + 5 eps) |
| S2 | SFT eval (22 Nov AM) | Both GPUs: SFT arm on the same 40, + 4 held-in replays + teacher-forced agreement for A and B | 7,200 s | **2.0** | 1.25 |
| — | Re-run reserve | One failed S1 or S2 restart (pip break, OOM, bug) | — | **2.5** | 1.5 |
| S3 | *Optional* Qwen3.5-9B zero-shot (22 Nov, only if the S2 gate passes and ≥ 4 h of quota remain) | vLLM 0.34.0 TP=2 `--dtype half --language-model-only`, non-thinking, 30 held-out seeds. Abort if it is not serving within 20 min | 7,200 s | **2.0** | 0 |
| — | Unallocated buffer | Quota rounding, idle overhead | — | **2.0** | 0.25 |
| | **Total** | | | **13.75** | **7.0** (= 14.0 quota-h at 2×) |

**Engineer-hours** [estimate]:
- harness, renderer, schema and notes validator, tested locally with a mock "replay the heuristic" LLM: 4–5 h;
- expert-data generation (CPU): 1 h;
- Kaggle plumbing: 1.5 h;
- analysis and write-up: 2 h.
- **Total ≈ 9–10 h**, which competes with the scored core.

---

## 8. Runbook with go/no-go gates

**G0, CPU only, before any GPU (21 Nov AM).** All must pass:
- The env is frozen and tagged. The official check is green.
- The harness replays heuristic actions through the LLM interface (mock model) and reproduces the heuristic's returns exactly.
- With the Qwen3 tokenizer, p95 prompt + target ≤ 3,300 tokens over all train-seed steps, and max ≤ 3,584.
- `sft/train.jsonl` has ≥ 400 samples from return-filtered episodes. The schema validator accepts 100% of expert targets.
- `heldout.json` has been written and hashed.

**S0 → G1.** Go only if all of these hold:
- vLLM 0.11.2 serves Qwen3-14B-AWQ on **each** GPU separately.
- Structured outputs enforce the enum.
- Effective time is ≤ 20 s per step at concurrency 4.
- QLoRA peak is ≤ 14.4 GB at 3,584 tokens.
- `kaggle quota` shows ≈ 1× accounting.

On failure:

| Failure | Response |
|---|---|
| vLLM breaks | Build llama.cpp (sm_75) as the eval stack (+0.3 h), with the LoRA converted via `convert_lora_to_gguf.py` |
| QLoRA OOM | r=16 → `offload_embedding` → 15 rows/page |
| 2× quota accounting | Switch to the minimal column |

**S1 → G2.** Requirements:
- SFT loss decreases, with final train loss < 50% of the initial value on the action tokens.
- The adapter is saved.
- Base eval produced ≥ 90% of its episodes.

If SFT fails, S2 becomes a "base-only failure analysis" and the training claim is dropped.

**S2 → G3.** First check the held-in replay: SFT should reach ≥ 70% teacher-forced agreement on validation steps. Below that, suspect template or quantization mismatch, and re-check AWQ vs the bnb base from the S0 numbers. Then apply the claim rule (§6). **Hard stop: no new GPU sessions after 22 Nov 12:00 Astana.** Use the afternoon for the write-up.

**Packaging:**
- Put everything under `experiments/kaggle_llm/` with its own `requirements-kaggle.txt`.
- The official one-command check never imports it.
- Commit trajectories (`trajectories.jsonl` per arm), `metrics.json`, `PROTOCOL.md` and `freeze.txt`.
- The adapter can stay a private Kaggle output; it is not needed to reproduce the official check.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Kaggle image or driver change breaks the `vllm==0.11.2` / torch combination | Medium | S0 first. Pin from `freeze.txt`. llama.cpp fallback |
| Prompt > 3.5k tokens, so SFT OOMs or vLLM prefill slows | Medium | G0 token gate. 15 rows/page. Notes cap |
| T4×2 counts 2× quota | Unknown | Measure in S0. Minimal plan = 7.0 session-h = 14.0 quota-h |
| SFT learns format but not the cross-page join, so return is flat | **High** | Expected. Report teacher-forced agreement and the failure taxonomy. No overclaim |
| AWQ base + adapter trained on bnb base mismatch | Low–medium | Held-in replay gate. bnb-in-vLLM fallback (Turing ✅) |
| Base model over-flags clean companies, so Δ is driven by one stratum | High | Stratified reporting |
| Leakage via expert notes | Low | Expert = truth-free heuristic. Truth is used only for return filtering on train seeds. Test asserts the harness never imports grader or truth paths |
| Time stolen from the scored core | High | G0 only after the core is green. Hard stop 22 Nov 12:00 |
| Qwen3.5-9B arm fails on T4 (GDN fp16, vLLM 0.34 freshness) | Medium–high | Optional, last, abort after 20 min |

---

## 10. What I could not verify

- Whether Kaggle counts T4×2 at 1× or 2× quota. No primary source found.
- Whether vLLM ≥ 0.12 runs on Kaggle's current driver. #66288 shows 2×T4 with driver 580 running vLLM main, but the host is not identified as Kaggle.
- The `KGAT_` token prefix (third-party only).
- The exact `kaggle kernels logs` flags.
- `offload_embedding` support for Qwen3 in `FastLanguageModel`.
- All throughput numbers marked [estimate]. S0 exists to replace them.

## Sources

Case: `research/case.pdf` §01, §04–§08, §10, §12. Prior research: `research/03-training-feasibility.md`.

Models and benchmarks:
- BFCL v4 CSV: https://gorilla.cs.berkeley.edu/data_overall.csv
- Qwen3-14B config: https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json
- Qwen3-14B AWQ: https://huggingface.co/Qwen/Qwen3-14B-AWQ
- Qwen3-14B GGUF: https://huggingface.co/Qwen/Qwen3-14B-GGUF
- Qwen3.5-9B card and config: https://huggingface.co/Qwen/Qwen3.5-9B
- Gemma 4 12B: https://huggingface.co/google/gemma-4-12B-it
- xLAM-2: https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r
- Qwen3.5 family listing: https://huggingface.co/Qwen

vLLM:
- GPU requirements: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- Quantization hardware table: https://docs.vllm.ai/en/stable/features/quantization/
- Triton attention backend: https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/triton_attn/
- Structured outputs (removal of `guided_*` in v0.12): https://github.com/vllm-project/vllm/blob/v0.14.0rc2/docs/features/structured_outputs.md
- T4 prefill regression: https://github.com/vllm-project/vllm/issues/60821
- Gemma 4 head_dim 512 on SM75: https://github.com/vllm-project/vllm/issues/57712
- Qwen3.5 on 2×T4: https://github.com/vllm-project/vllm/issues/66288 (fix: https://github.com/vllm-project/vllm/pull/66289)
- FlashInfer SM75 PR (open): https://github.com/vllm-project/vllm/pull/55380
- MRV2 SM75 spec-decode bug: https://github.com/vllm-project/vllm/issues/61233
- Releases: https://github.com/vllm-project/vllm/releases

llama.cpp:
- CUDA scoreboard: https://github.com/ggml-org/llama.cpp/discussions/15013
- sm_75 FA tile cap: https://github.com/ggml-org/llama.cpp/issues/28761
- sm_75 cold start, Qwen3.5: https://github.com/ggml-org/llama.cpp/issues/29513
- T4 CUDA13 PTX: https://github.com/ggml-org/llama.cpp/issues/31503

Qwen3.5 on T4 (non-vLLM): https://github.com/strands-labs/strands-decider/issues/113

Unsloth:
- Qwen3-14B Kaggle T4: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(14B)-Reasoning-Conversational.ipynb
- Phi-4 14B GRPO Kaggle T4: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Phi_4_(14B)-GRPO.ipynb
- Qwen3-4B GRPO tool use: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO-Tool_Use.ipynb
- Qwen3.5-4B on T4 (fp32): https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_5_(4B)-Decision.ipynb
- Qwen3.8-27B on 2×T4: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3.8_(27B)-Conversational.ipynb
- gpt-oss-20b T4: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-gpt-oss-(20B)-Fine-tuning.ipynb
- Gemma 4 12B: https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Gemma4_(12B)_Text.ipynb
- Qwen3.5 fine-tune docs: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- VRAM table: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- Multi-GPU: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth
- Qwen3.8 recipe: https://unsloth.ai/docs/models/qwen3.8/train

TRL:
- SFT tool calling and assistant-only loss: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md
- GRPO: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md
- OpenEnv / Wordle: https://huggingface.co/docs/trl/en/openenv

Kaggle:
- CLI docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
- Kernels: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- Kernel metadata: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- Datasets: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md
- Changelog: https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md
- Quota: https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md
- Auth: https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md
- PyPI: https://pypi.org/project/kaggle/
- Image: https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl
- P100 issue: https://github.com/Kaggle/docker-python/issues/1546

Self-training and agent fine-tuning:
- STaR: https://arxiv.org/abs/2203.14465
- RFT: https://arxiv.org/abs/2308.01825
- ReST-EM: https://arxiv.org/abs/2312.06585
- FireAct: https://arxiv.org/abs/2310.05915
- AgentTuning: https://arxiv.org/abs/2310.12823
