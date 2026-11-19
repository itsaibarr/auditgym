# Kaggle runbook: Qwen3-14B on T4x2

All commands run from the repo root. `<user>` is the Kaggle username: `kaggle config view` or `kaggle kernels list -m`. Plan details and sources: `research/06-kaggle-9b-training-plan.md` §5 and §8.

## 0. CLI setup (laptop)

```bash
pipx install 'kaggle>=2.2.5'      # or: uv tool install 'kaggle>=2.2.5'
kaggle --version                  # must be >= 2.2.5 (quota needs 2.2.1+; 1.8.3 fixed a KAGGLE_API_TOKEN KeyError)
export KAGGLE_API_TOKEN="$(security find-generic-password -a genesis-audit-gym -s kaggle-api-token -w)"
kaggle config view                # confirms auth and username; never print the token
kaggle quota -v                   # CSV: resource,used,remaining,total,refreshAt -> paste into experiments/kaggle_llm/quota.log
```

GPU and internet need a phone-verified account. If `quota` shows < 7 h remaining, use the minimal plan only.

## 1. Gate G0 (CPU only, before any GPU)

All must hold:
- Core env frozen and tagged; `auditgym check` green.
- The harness replays heuristic actions through the LLM interface (mock model) and reproduces the heuristic's returns exactly.
- Qwen3 tokenizer: p95 prompt+target <= 3,300 tokens, max <= 3,584, over all train-seed steps.
- `sft/train.jsonl` has >= 400 samples from return-filtered episodes; the schema validator accepts 100% of expert targets.
- `seeds/heldout.json` written, sha256 recorded in `PROTOCOL.md`; `PROTOCOL.md` committed.
- A test asserts the harness never imports grader or truth modules.

## 2. Dataset package

`kaggle/pkg/dataset-metadata.json`:
```json
{"title": "genesis-audit-gym-pkg", "id": "<user>/genesis-audit-gym-pkg", "licenses": [{"name": "CC0-1.0"}]}
```
Contents: `genesis_audit_gym-<ver>-py3-none-any.whl`, `harness/`, `seeds/heldout.json`, `sft/train.jsonl`, `sft/val.jsonl`.

```bash
python -m build --wheel && cp dist/genesis_audit_gym-*.whl kaggle/pkg/
kaggle datasets create  -p kaggle/pkg -r zip                  # first time; private by default
kaggle datasets version -p kaggle/pkg -r zip -m "v2: <change>"  # later versions
```
`-r zip` is required: the default dir-mode `skip` **silently drops subfolders**. Inside the kernel, inputs mount under `/kaggle/input/...`; glob for the wheel rather than hard-coding the path.

## 3. Kernel folders

`kaggle/k_smoke/kernel-metadata.json` (the same shape for `k_train_eval` and `k_sft_eval`):
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
For `k_sft_eval`, set `"kernel_sources": ["<user>/gag-train-eval"]` to mount the S1 adapter.

`run.py` must:
- print `nvidia-smi`, `python -V`, and the torch/transformers/vllm/triton versions;
- install `vllm==0.11.2 bitsandbytes unsloth`, then `transformers==4.57.6`, then `trl==0.22.2 --no-deps` (keep Kaggle's torch);
- install the wheel from the dataset;
- write `pip freeze > /kaggle/working/freeze.txt`;
- launch one process per GPU with `CUDA_VISIBLE_DEVICES=0|1`, each taking half the seeds.

Download weights to `/tmp`, **not** `/kaggle/working` (that is saved as output, ~20 GB cap). Write metrics and trajectories to `/kaggle/working/<arm>_g<gpu>/` as the run goes, so a timeout still leaves results.

## 4. Push, watch, fetch (each session)

```bash
kaggle quota -v >> experiments/kaggle_llm/quota.log                       # BEFORE
kaggle kernels push -p kaggle/k_smoke --accelerator NvidiaTeslaT4 -t 4500 # ALWAYS pass -t
kaggle kernels status <user>/gag-smoke
kaggle kernels logs   <user>/gag-smoke --follow --interval 30             # or poll status
until kaggle kernels status <user>/gag-smoke | grep -qiE 'complete|error|cancel'; do sleep 120; done
kaggle kernels output <user>/gag-smoke -p runs/kaggle/smoke -o
kaggle quota -v >> experiments/kaggle_llm/quota.log                       # AFTER: 1x or 2x billing?
```
Caps: S0 `-t 4500`, S1 `-t 14400` (`k_train_eval`), S2 `-t 7200` (`k_sft_eval`), S3 `-t 7200`. Copy S0's `freeze.txt` pins into later sessions' installs.

## 5. Go/no-go checkpoints

**G1 (after S0):** go only if all hold:
- vLLM 0.11.2 serves Qwen3-14B-AWQ on **each** GPU separately;
- structured outputs enforce the ID enum (test an unseen ID: it must be impossible);
- <= 20 s effective per step at concurrency 4;
- QLoRA peak <= 14.4 GB at seq 3,584;
- quota billing ~1x.

| S0 failure | Response |
|---|---|
| vLLM won't serve | llama.cpp built with `-DCMAKE_CUDA_ARCHITECTURES=75` (+0.3 h); convert the LoRA with `convert_lora_to_gguf.py` |
| QLoRA OOM | r=16 -> `offload_embedding=True` -> 15 rows per page |
| 2x billing | Minimal plan (SKILL.md budget) |
| Step time > 2x the estimate | 20 held-out + 10 twins (never fewer than 20 held-out) |

**G2 (after S1):** final SFT loss on action tokens is < 50% of the initial loss; the adapter is saved; base eval finished >= 90% of its episodes. If SFT failed, S2 becomes a base-only failure analysis and the training claim is dropped.

**G3 (after S2):** first check held-in teacher-forced agreement >= 70%. If it is lower, suspect template or AWQ-vs-bnb mismatch; try bnb in vLLM. Then apply the claim rule. **Hard stop: no new sessions after 22 Nov 12:00 Astana.**

## 6. Commit after each session

Commit `freeze.txt`, per-arm `trajectories.jsonl`, `metrics.json`, `quota.log` and the CI table. The adapter stays a private Kaggle output. Before committing, grep the files for `KAGGLE_API_TOKEN` and the token value; abort if either is found.

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Kaggle image or driver breaks the vllm 0.11.2/torch combination | Medium | S0 first, pin from freeze, llama.cpp fallback |
| Prompt > 3.5k tokens -> OOM or slow prefill (T4 Triton prefill is FMA-bound) | Medium | G0 token gate, 15 rows/page, notes cap |
| T4x2 billed at 2x | Unknown | Measure in S0, minimal plan |
| SFT learns the format but not cross-page joins (flat return) | High | Expected; report agreement + taxonomy, no overclaim |
| AWQ inference vs bnb-trained adapter mismatch | Low-medium | Held-in replay gate; bnb-in-vLLM fallback |
| One stratum (clean over-flagging) drives Δ | High | Stratified reporting |
| Time stolen from the scored core | High | G0 only after the core is green; 12:00 hard stop |
| Token leaks into logs or the repo | Low | Keychain + env var only; pre-commit grep |
| Qwen3.5-9B arm fails on T4 | Medium-high | Optional, last, 20-min abort |
