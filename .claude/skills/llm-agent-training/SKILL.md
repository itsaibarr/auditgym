---
name: llm-agent-training
description: Plan, run, evaluate or review the optional LLM agent policy for AuditGym — Kaggle T4x2 GPU sessions, Qwen3-14B QLoRA SFT/RFT, vLLM structured decoding, held-out evaluation and any ML or "training improved X" claim. Use when touching experiments/kaggle_llm/, pushing Kaggle kernels, building SFT data, or writing results about a learned policy.
---

# Optional LLM agent training (AuditGym)

Background: `research/06-kaggle-9b-training-plan.md` (the plan, with sources) and `research/03-training-feasibility.md` (why not small models or GRPO). Step-by-step commands: `references/kaggle-runbook.md`.

## Non-negotiable constraints

- **It scores 0 points.** The jury scores the env and grader (case.pdf §01, §10, §12). Start only after the official check is green and the core is frozen and tagged. **No new GPU sessions after 22 Nov 12:00 Astana.**
- **Kaggle free GPUs only** (T4x2). No paid APIs or other clouds. Kamatera has no GPUs.
- **Models >= 9B parameters.** Qwen3-8B, xLAM-2-8b and the 4B models are out.
- **Never needed for the official offline check.** Everything lives in `experiments/kaggle_llm/` with its own `requirements-kaggle.txt`; `auditgym check` never imports it. A test asserts that neither the harness nor the policy imports grader or truth modules.
- **No improvement claim without a controlled held-out comparison** (protocol below). A null result is reported as a null, honestly, with its CI.

## The chosen plan (settled; don't relitigate)

| Decision | Choice |
|---|---|
| Model | **Qwen3-14B, non-thinking** (`enable_thinking=False`). Inference via `Qwen/Qwen3-14B-AWQ` fp16; training via `unsloth/Qwen3-14B-unsloth-bnb-4bit` |
| Optional arm | Qwen3.5-9B **zero-shot only**, last, only if >= 4 h of quota remain; abort if it isn't serving within 20 min. It can't train on T4 (fp16 NaNs, needs fp32) |
| Inference | **vLLM 0.11.2** (Unsloth's T4 pin; keep Kaggle's torch). **One engine per T4** (`CUDA_VISIBLE_DEVICES=g`, seeds sharded), no tensor parallelism, no spec-decode. `max_model_len=4096`, `gpu_memory_utilization=0.92`, prefix caching on, greedy. Fallback: llama.cpp built for sm_75 |
| Structured output | A JSON schema per step. ID fields (`get_record.id`, `get_document.doc_id`, submit `affected_ids` / `evidence_ids`) are `enum`s over **IDs seen so far in this episode**; `table` and finding `type` are enums; `limit` 1-20; calculator regex `^[0-9.+\-*/() ]{1,200}$`; maxItems 30/10; explanation maxLength 300. A short `note_add` comes before `action`. On 0.11.2 try `StructuredOutputsParams`, then fall back to `GuidedDecodingParams` (`guided_*` was removed in 0.12) |
| Prompt | **<= ~3.5k tokens** (gate: p95 <= 3,300, max <= 3,584), static parts first for prefix caching: system ~700, task + manifest + policy ~400, coverage tracker ~150, NOTES <= 30 lines / ~600, **last observation only** as CSV (<= 20 rows) or one document <= 1,550, last action + error ~100. Observation compaction: notes carry the memory; **strip ID-like tokens not in the seen-ID set** before re-injecting notes |
| Episode budget | Small eval configs (150-300 ledger rows, 20-40 docs, 0-4 errors). The LLM gets <= 25 env steps; at step 23 the schema switches to submit-only. **Submit guard:** validate locally, allow one free repair generation, otherwise submit `[]` (silence, never the -1 malformed penalty) |
| Training | **RFT / filtered behaviour cloning**: QLoRA SFT on per-step (prompt -> `{note_add, action}`) samples from the **truth-free heuristic policy** run under the LLM's constraints (limit 20, <= 25 steps, same renderer) on **train seeds**. Keep an episode only if its shared return is >= 0.5 (>= 0.9 on clean books). Keep every submit step and subsample navigation to ~550 samples. r=16, alpha 32, all 7 projections, seq 3,584, bs 1, GA 8, lr 1e-4, 1 epoch, adamw_8bit, loss on completion only; stop the clock at 3.0 h and save the adapter |
| Never | **Never clone the oracle.** It acts on hidden truth, so cloning teaches confident accusations without evidence. Truth is used only to compute the return filter on train seeds |
| GRPO | **Not attempted.** At 14B on T4 one step with 4 rollouts takes ~40-50 min (about 8 steps in the whole quota); cold-start returns are ~ -1, so group variance is ~0; TRL multi-turn needs prefix-preserving chats that our compaction breaks; Unsloth's T4 tool-use GRPO can't use vLLM. Say this in README limitations |
| Template parity | Render prompts **once** with our function (`apply_chat_template(..., enable_thinking=False)`). The same string feeds SFT and vLLM `generate()` |

Framing for the write-up: "SFT on reward-filtered expert trajectories (one round of expert iteration)". Not "RL", and not "beats the heuristic". Behaviour cloning is capped near its expert.

## GPU budget (<= 13.75 h of ~14 h; check before every push)

| Session | Contents | `-t` cap | h |
|---|---|---|---|
| S0 calibrate | Install + freeze; AWQ load per GPU; 2 base episodes per GPU; s/step; 8 QLoRA steps for memory and s/sample; confirm the structured-output param | 4,500 | 1.25 |
| S1 | GPU0 SFT (3 h wall stop) in parallel with GPU1 base eval (30 held-out + 10 twins) | 14,400 | 4.0 |
| S2 | SFT eval on the same 40 + 4 held-in replays + teacher-forced agreement for base and SFT | 7,200 | 2.0 |
| Reserve | One failed S1/S2 restart | — | 2.5 |
| S3 (opt.) | Qwen3.5-9B zero-shot, vLLM 0.34.0, TP=2 | 7,200 | 2.0 |
| Buffer | Rounding, idle | — | 2.0 |

Run `kaggle quota -v` **before and after S0**. Whether T4x2 bills 1x or 2x is unverified; if 2x, switch to the minimal plan (S0 1.0, S1 3.0 with 20+5 episodes, S2 1.25, reserve 1.5, no S3). Always pass `-t` on push; it's the quota circuit-breaker.

## Evaluation protocol (pre-register before S1)

Commit `experiments/kaggle_llm/PROTOCOL.md` **before S1**: hypothesis (SFT mean return > base), primary metric, sha256 of the seed list, decoding params, claim rule, and the number of training configs planned (1).

- **Seeds:** train seeds -> expert data; 4 more train-config seeds held out of SFT for the teacher-forced check; **30 held-out seeds** from dev/test configs with unseen parameter combos (>= 8 clean, >= 8 interacting) in `seeds/heldout.json` + sha256, written once and not looked at per seed before S2 ends; **10 of them as order-permuted + ID-renamed twins**. Never go below 20 held-out.
- **Freeze:** env version + git SHA, harness/prompt hash, model and AWQ revisions, `temperature=0`, `max_tokens=256`, 25-step cap, page limit 20.
- **Arms on the same seeds, harness and hardware:** A base, B base + LoRA; reference rows: heuristic (full and matched budget), random, flag-all, oracle; optional C Qwen3.5-9B.
- **Report per arm:** P, R, F1 (micro), mean shared return, steps, wall time; **invalid-action rate** (bounded env errors), **invented-ID rate** (notes-ID strips; 0 malformed / 0 invented in actions by construction), schema repairs, forced submits, **false-accusation rate on clean companies**; stratified clean / single / interacting.
- **Statistics:** paired BCa bootstrap over the 30 seeds, 9,999 resamples (`scipy.stats.bootstrap(..., paired=True)`), 95% CI of Δreturn (B-A); Wilcoxon as secondary; win/tie/loss counts; twin-set Δ reported separately.
- **Claim rule:** write "SFT improves on held-out seeds" **only if** the 95% CI excludes 0 **and** B's twin-set return doesn't drop by more than its CI. Otherwise write: "no significant difference (Δ = x, 95% CI [a, b], n = 30)". Minimum detectable Δ is ~0.18, so a null is likely (30-50%).
- **If null:** CI table + teacher-forced agreement (high agreement with flat return = a memory/arithmetic bottleneck, not an imitation one) + a failure taxonomy over 10 hand-read SFT trajectories (premature submit, notes drift, wrong amount or exponent, missing evidence set, over-flagging clean books). This feeds the scored "measured failure analysis" and "honest limitations".

## Credentials

The Kaggle token lives in the macOS Keychain. Export it only for the command's own shell:

```bash
export KAGGLE_API_TOKEN="$(security find-generic-password -a genesis-audit-gym -s kaggle-api-token -w)"
```

Never echo it, `set -x` around it, write it to the repo, `~/.kaggle`, `.env`, logs, kernel code, kernel metadata or memory. Kernels don't need it, and HF weights are public. If a command output might contain it, redact it before showing.

## Reviewing ML/training claims (checklist)

Reject or flag any statement that lacks: pre-registered protocol committed before S1; identical seeds, harness and decoding across arms; held-out seeds disjoint from SFT data; paired CI; twin-set result; stratified clean-FP rate; the count of configs tried; `freeze.txt`, per-arm `trajectories.jsonl`, `metrics.json`. Flag oracle-derived training data, test seeds used for tuning, "RL" wording for SFT, and any comparison of the LLM against the full-budget heuristic presented as the headline.
