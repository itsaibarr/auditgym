# AuditGym training experiment: Kaggle GPU job. Shipped inside the private dataset; the kernel stub runs it with
# GAG_MODE set. Modes:
#   smoke : calibrate memory/speed (one SFT micro-step on the longest samples, generation on the longest prompts)
#   base  : greedy generation on the held-out set with the base model (both GPUs), then QLoRA SFT -> /kaggle/working/adapter
#   after : greedy generation on the held-out set with base + adapter (adapter mounted from the base kernel's output)
# Grading is not done here: training/grade.py replays every generation through the env's own grader.
import glob
import json
import math
import os
import subprocess
import sys
import time

W = "/kaggle/working"
MODE = os.environ.get("GAG_MODE", "smoke")
MODEL_REPO, MODEL_REV = "unsloth/Qwen3-14B-bnb-4bit", "b455922269e4678c84f854fe38a6993d73614a1e"
MAX_NEW_TOKENS = 1024
GEN_BATCH = int(os.environ.get("GAG_GEN_BATCH", "2"))
ATTN_CHUNK = int(os.environ.get("GAG_ATTN_CHUNK", "512"))
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
LR, GRAD_ACCUM, WARMUP = 1e-4, 8, 3
TRAIN_SECONDS = float(os.environ.get("GAG_TRAIN_SECONDS", "8400"))   # wall-clock stop; the adapter is saved first
TRAIN_SAMPLES = int(os.environ.get("GAG_TRAIN_SAMPLES", "240"))      # planned samples (cosine schedule length)
TRAIN_DEVICES = os.environ.get("GAG_TRAIN_DEVICES", "balanced")      # "single" (GPU0 only) or "balanced" (both GPUs)
# config 1 used "sample_mean": per-sample mean CE, then /accum (short answers dominate, see README); config 2: "token_sum":
# summed CE over the accumulation group / number of completion tokens in that group
LOSS_NORM = os.environ.get("GAG_LOSS_NORM", "sample_mean")


def sh(c, check=True):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=check)


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def log(**kw):
    kw["t"] = round(time.time(), 1)
    print(json.dumps(kw), flush=True)
    with open(f"{W}/log_{MODE}.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


def peak():
    import torch
    return {d: round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())}


def child(args, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args, env=env)


def setup():
    sh("nvidia-smi; python -V")
    sh("pip install -q -U bitsandbytes peft 2>&1 | tail -3", check=False)
    sh("pip freeze | grep -iE '^(torch|transformers|peft|bitsandbytes|accelerate|safetensors)==' || true")
    sh(f"pip freeze > {W}/freeze_{MODE}.txt")
    from huggingface_hub import snapshot_download
    t = time.time()
    p = snapshot_download(MODEL_REPO, revision=MODEL_REV, cache_dir="/tmp/hf")
    log(event="downloaded", s=round(time.time() - t))
    return p


def install_chunked_attention():
    """Exact causal attention computed in query chunks. On Kaggle's T4 stack (torch 2.11) the stock SDPA path
    materialises the full heads x L x L score matrix (5.6 GiB at 6k tokens); chunking caps it at heads x chunk x L."""
    import torch
    import torch.nn.functional as F
    from transformers import AttentionInterface

    def attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        L, S = query.shape[2], key.shape[2]
        rep = query.shape[1] // key.shape[1]
        if rep > 1:
            key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
        mask = attention_mask[:, :, :, :S] if attention_mask is not None and attention_mask.dim() == 4 else None
        outs = []
        for i in range(0, L, ATTN_CHUNK):
            q = query[:, :, i:i + ATTN_CHUNK]
            if mask is not None:
                m = mask[:, :, i:i + ATTN_CHUNK] if mask.shape[2] > 1 else mask
            elif L > 1:
                qpos = torch.arange(i, i + q.shape[2], device=q.device) + (S - L)
                m = torch.arange(S, device=q.device)[None, :] <= qpos[:, None]
            else:
                m = None
            outs.append(F.scaled_dot_product_attention(q, key, value, attn_mask=m, scale=scaling))
        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("sdpa", attn)


def two_gpu_map(n_layers=40, split=20):
    m = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    m.update({f"model.layers.{i}": int(i >= split) for i in range(n_layers)})
    return m


def load(path, device_map, adapter=None, trainable=False):
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    install_chunked_attention()
    kw = dict(device_map=two_gpu_map() if device_map == "balanced" else device_map, attn_implementation="sdpa")
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, **kw)
    for mod in m.modules():           # the checkpoint declares bf16 compute; T4 has no bf16, so compute in fp16
        if isinstance(mod, bnb.nn.Linear4bit):
            mod.compute_dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(path)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter, is_trainable=False)
        for n, p in m.named_parameters():   # adapter saved in fp32; infer in the model's fp16 (fp32 LoRA OOMs at 5.6k x 2)
            if "lora_" in n:
                p.data = p.data.half()
    elif trainable:
        from peft import LoraConfig, get_peft_model
        m.config.use_cache = False
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA))
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.float()   # fp32 master weights for the adapter, fp16 autocast compute
        m.print_trainable_parameters()
    m.eval()
    return m, tok


def prompt_text(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ------------------------------------------------------------------------------------------- generation (child)
def generate(spec):
    import torch
    path, rows, out_file, adapter = spec["path"], spec["rows"], spec["out"], spec["adapter"]
    max_new, batch = spec.get("max_new", MAX_NEW_TOKENS), spec.get("batch", GEN_BATCH)
    done = {json.loads(l)["seed"] for l in open(out_file)} if os.path.exists(out_file) else set()
    rows = sorted([r for r in rows if r["seed"] not in done], key=lambda r: -r["prompt_tokens"])
    if not rows:
        return
    m, tok = load(path, {"": 0}, adapter=adapter)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([prompt_text(tok, r["messages"]) for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda:0")
        t = time.time()
        with torch.inference_mode():
            out = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=eos, pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        L = enc["input_ids"].shape[1]
        with open(out_file, "a") as f:
            for j, r in enumerate(chunk):
                new = out[j, L:]
                n_new = int(sum(1 for x in new.tolist() if x not in (tok.pad_token_id,)))
                f.write(json.dumps({"seed": r["seed"], "episode_id": r["episode_id"], "profile": r["profile"],
                                    "text": tok.decode(new, skip_special_tokens=True),
                                    "prompt_tokens": int(enc["attention_mask"][j].sum()), "new_tokens": n_new,
                                    "hit_max_new_tokens": n_new >= max_new, "batch_size": len(chunk),
                                    "batch_seconds": round(dt, 1)}) + "\n")
        log(event="gen_batch", tag=spec["tag"], n=len(chunk), s=round(dt, 1), peak_gb=peak(), in_tokens=L,
            done=i + len(chunk), total=len(rows))


def generate_sharded(path, rows, tag, adapter=None, **extra):
    """One child per GPU, each takes every other episode (by seed order)."""
    import torch
    n = torch.cuda.device_count()
    rows = sorted(rows, key=lambda r: r["seed"])
    procs = []
    for g in range(n):
        spec = dict(path=path, rows=rows[g::n], out=f"{W}/gen_{tag}_g{g}.jsonl", adapter=adapter, tag=tag, **extra)
        json.dump(spec, open(f"{W}/spec_{tag}_g{g}.json", "w"))
        procs.append(child(["--gen", f"{W}/spec_{tag}_g{g}.json"], str(g)))
    codes = [p.wait() for p in procs]
    log(event="gen_done", tag=tag, codes=codes)
    return codes


# ------------------------------------------------------------------------------------------- SFT (child)
def train(spec):
    import torch
    path, rows, devices = spec["path"], spec["rows"], spec["devices"]
    n_plan, max_s, accum = spec.get("samples", TRAIN_SAMPLES), spec.get("seconds", TRAIN_SECONDS), spec.get("accum", GRAD_ACCUM)
    out_dir = spec.get("out", f"{W}/adapter")
    m, tok = load(path, {"": 0} if devices == "single" else "balanced", trainable=True)
    data = []
    for r in rows[:n_plan]:
        p = tok(prompt_text(tok, r["messages"]), add_special_tokens=False)["input_ids"]
        c = tok(r["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        data.append((p, c, r["seed"]))
    log(event="sft_data", n=len(data), max_len=max(len(p) + len(c) for p, c, _ in data), devices=devices)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = max(1, math.ceil(len(data) / accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda")
    causal = m.base_model.model           # the CausalLM with LoRA layers injected
    dev0 = causal.model.embed_tokens.weight.device
    group_tok = [sum(len(c) for _, c, _ in data[i:i + accum]) for i in range(0, len(data), accum)]
    log(event="loss_norm", norm=LOSS_NORM)
    t0, step, acc_loss, acc_tok, seen = time.time(), 0, 0.0, 0, 0
    m.train()
    for k, (p, c, seed) in enumerate(data):
        ids = torch.tensor([p + c], device=dev0)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = causal.model(input_ids=ids, use_cache=False).last_hidden_state
            h = hs[:, len(p) - 1:-1]       # only the positions that predict completion tokens reach the LM head
            logits = causal.lm_head(h.to(causal.lm_head.weight.device)).float()
        tgt = ids[:, len(p):].to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        obj = loss / accum if LOSS_NORM == "sample_mean" else loss * len(c) / group_tok[k // accum]
        scaler.scale(obj).backward()
        acc_loss += loss.item() * len(c)
        acc_tok += len(c)
        seen += 1
        if seen % accum == 0 or k == len(data) - 1:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log(event="sft_step", step=step, samples=seen, loss=round(acc_loss / acc_tok, 4), grad_norm=round(float(gn), 3),
                lr=sched.get_last_lr()[0], s=round(time.time() - t0), peak_gb=peak(), scale=scaler.get_scale())
            acc_loss, acc_tok = 0.0, 0
            if step % 5 == 0:
                m.save_pretrained(out_dir)
            if time.time() - t0 > max_s:
                log(event="sft_time_stop", samples=seen)
                break
    m.save_pretrained(out_dir)
    json.dump({"samples_trained": seen, "optimizer_steps": step, "seconds": round(time.time() - t0),
               "trained_seeds": [s for _, _, s in data[:seen]], "lora": LORA, "lr": LR, "grad_accum": accum,
               "planned_samples": len(data), "devices": devices, "loss_norm": LOSS_NORM, "model": MODEL_REPO, "revision": MODEL_REV},
              open(f"{out_dir}/train_meta.json", "w"), indent=1)
    log(event="sft_saved", samples=seen, steps=step, s=round(time.time() - t0))


# ------------------------------------------------------------------------------------------- diagnostic (child)
def diag(spec):
    """Teacher-forced log-probs of the empty answer and of the truth-free expert answer (diagnostic only)."""
    import torch
    m, tok = load(spec["path"], {"": 0}, adapter=spec["adapter"])
    causal = m.base_model.model if spec["adapter"] else m
    with open(spec["out"], "w") as f:
        for r in spec["rows"]:
            p = tok(prompt_text(tok, r["messages"]), add_special_tokens=False)["input_ids"]
            res = {"seed": r["seed"], "split": r["split"]}
            for name, text in (("empty", '{"findings": []}'), ("expert", r["target"])):
                c = tok(text + "<|im_end|>", add_special_tokens=False)["input_ids"]
                ids = torch.tensor([p + c], device="cuda:0")
                with torch.inference_mode():
                    hs = causal.model(input_ids=ids, use_cache=False).last_hidden_state[:, len(p) - 1:-1]
                    lp = torch.log_softmax(causal.lm_head(hs).float(), -1)[0]
                tl = lp.gather(1, torch.tensor(c, device="cuda:0")[:, None])[:, 0]
                first_div = next((i for i, (a, b) in enumerate(zip(c, tok('{"findings": []}<|im_end|>', add_special_tokens=False)["input_ids"])) if a != b), None)
                res[name] = {"logp_sum": round(float(tl.sum()), 3), "n_tokens": len(c),
                             "nll_per_token": round(float(-tl.mean()), 4),
                             "greedy_token_agreement": round(float((lp.argmax(-1) == torch.tensor(c, device="cuda:0")).float().mean()), 4)}
                if name == "expert":
                    res["branch_index"] = first_div
                    if first_div is not None:
                        e_tok = tok('{"findings": []}<|im_end|>', add_special_tokens=False)["input_ids"][first_div]
                        res["branch_logp_expert_token"] = round(float(lp[first_div, c[first_div]]), 3)
                        res["branch_logp_empty_token"] = round(float(lp[first_div, e_tok]), 3)
            f.write(json.dumps(res) + "\n")
            f.flush()
    log(event="diag_done", tag=spec["tag"], n=len(spec["rows"]))


def run_train(path, rows, devices, tag, **extra):
    spec = dict(path=path, rows=rows, devices=devices, **extra)
    json.dump(spec, open(f"{W}/spec_train_{tag}.json", "w"))
    code = child(["--train", f"{W}/spec_train_{tag}.json"], "0" if devices == "single" else "0,1").wait()
    log(event="train_exit", tag=tag, devices=devices, code=code)
    return code


# ------------------------------------------------------------------------------------------- modes
def main():
    if len(sys.argv) > 2:
        spec = json.load(open(sys.argv[2]))
        {"--gen": generate, "--train": train, "--diag": diag}[sys.argv[1]](spec)
        return
    path = setup()
    held = [json.loads(l) for l in open(find("heldout.jsonl"))]
    sft = [json.loads(l) for l in open(find("sft.jsonl"))]
    log(event="start", mode=MODE, n_held=len(held), n_sft=len(sft))
    if MODE == "smoke":
        longest = sorted(sft, key=lambda r: -(r["prompt_tokens"] + r["target_tokens"]))[:3]
        for devices in ("balanced", "single"):
            run_train(path, longest, devices, f"smoke_{devices}", samples=3, accum=1, seconds=1e9, out=f"/tmp/adapter_{devices}")
        generate_sharded(path, sorted(held, key=lambda r: -r["prompt_tokens"])[:2 * GEN_BATCH], "smoke")
    elif MODE == "base":
        generate_sharded(path, held, "base")
        run_train(path, sft, TRAIN_DEVICES, "main")
    elif MODE == "sft":
        run_train(path, sft, TRAIN_DEVICES, "main")
    elif MODE == "diag":
        tg = {json.loads(l)["seed"]: json.loads(l)["target"] for l in open(find("heldout_expert_targets.jsonl"))}
        rows = [{**r, "target": tg[r["seed"]], "split": "heldout"} for r in held] + \
               [{**r, "split": "train_seen"} for r in sft[:20]]
        adapter = os.path.dirname(find("adapter_config.json"))
        procs = []
        for g, (tag, ad) in enumerate((("base", None), ("after", adapter))):
            spec = dict(path=path, rows=rows, adapter=ad, tag=tag, out=f"{W}/diag_{tag}.jsonl")
            json.dump(spec, open(f"{W}/spec_diag_{tag}.json", "w"))
            procs.append(child(["--diag", f"{W}/spec_diag_{tag}.json"], str(g)))
        log(event="diag_exit", codes=[p.wait() for p in procs])
    elif MODE == "after":
        generate_sharded(path, held, "after", adapter=os.path.dirname(find("adapter_config.json")))
    log(event="end", mode=MODE)


if __name__ == "__main__":
    main()
