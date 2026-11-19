"""Build the held-out eval set and the reward-filtered SFT set (CPU only, deterministic).

    uv run --with tokenizers python training/make_data.py

Writes training/data/{heldout.jsonl,sft.jsonl,manifest.json}. Prompt length is measured with the Qwen3 tokenizer;
episodes whose prompt exceeds MAX_PROMPT_TOKENS are skipped by a rule that looks only at the public workbook.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tokenizers import Tokenizer  # noqa: E402

import agpolicy as A  # noqa: E402
from episodes import HELDOUT_MIX, HELDOUT_SEED0, TRAIN_SEED0, config, profile_cycle  # noqa: E402
from play import expert_decide, play  # noqa: E402

MODEL_REPO, MODEL_REV = "unsloth/Qwen3-14B-bnb-4bit", "b455922269e4678c84f854fe38a6993d73614a1e"
MAX_PROMPT_TOKENS = 5632      # + up to 1,024 completion tokens = 6,656 training sequence cap
MAX_TARGET_TOKENS = 1024
N_TRAIN = 400                 # SFT candidates kept after the reward filter; the kernel trains on a prefix within its time budget
REWARD_FILTER = 0.5           # keep an expert episode only if its env reward >= this (one round of filtered behaviour cloning)
TEMPLATE_OVERHEAD = 30        # chat-template tokens around system + user (checked exactly on Kaggle)


def main():
    tok = Tokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    stats = {"heldout_skipped_long": Counter(), "heldout_skipped_invalid_config": Counter(),
             "train_skipped_long": 0, "train_skipped_invalid_config": 0, "train_rejected_by_reward": 0}

    def episode(seed, profile):
        cfg = config(seed, profile)
        try:
            obs0, rows, n, g = play(seed, cfg, expert_decide)
        except ValueError:
            return None, cfg
        wb = A.render(obs0, rows)
        return dict(seed=seed, profile=profile, config=cfg, episode_id=obs0["manifest"]["episode_id"], scan_steps=n,
                    messages=A.messages(wb), prompt_tokens=ntok(A.SYSTEM) + ntok(wb) + TEMPLATE_OVERHEAD,
                    expert={k: g[k] for k in ("reward", "f1", "tp", "fp", "fn")},
                    n_issues=g["tp"] + g["fn"], _payload=None), cfg

    # held-out: fill the per-profile quota in seed order
    need, held = dict(HELDOUT_MIX), []
    seed = HELDOUT_SEED0
    while any(need.values()):
        for profile in [p for p, k in need.items() if k > 0]:
            e, _ = episode(seed, profile)
            seed += 1
            if e is None:
                stats["heldout_skipped_invalid_config"][profile] += 1
            elif e["prompt_tokens"] > MAX_PROMPT_TOKENS:
                stats["heldout_skipped_long"][profile] += 1
            else:
                e.pop("_payload")
                held.append(e)
                need[profile] -= 1
    # SFT: expert targets on train seeds, reward-filtered
    sft, seed, cyc = [], TRAIN_SEED0, profile_cycle()
    while len(sft) < N_TRAIN:
        profile = next(cyc)
        cfg = config(seed, profile)
        try:
            obs0, rows, n, g = play(seed, cfg, expert_decide)
        except ValueError:
            stats["train_skipped_invalid_config"] += 1
            seed += 1
            continue
        wb = A.render(obs0, rows)
        target = A.target_text(expert_decide(obs0, rows)["findings"])
        p = ntok(A.SYSTEM) + ntok(wb) + TEMPLATE_OVERHEAD
        if p > MAX_PROMPT_TOKENS or ntok(target) > MAX_TARGET_TOKENS:
            stats["train_skipped_long"] += 1
        elif g["reward"] < REWARD_FILTER:
            stats["train_rejected_by_reward"] += 1
        else:
            sft.append(dict(seed=seed, profile=profile, messages=A.messages(wb), target=target, prompt_tokens=p,
                            target_tokens=ntok(target), expert_reward=g["reward"], expert_f1=g["f1"]))
        seed += 1
    assert not {e["seed"] for e in held} & {e["seed"] for e in sft}

    def dump(name, rows):
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (out / name).write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    manifest = {
        "model": MODEL_REPO, "model_revision": MODEL_REV, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_target_tokens": MAX_TARGET_TOKENS, "reward_filter": REWARD_FILTER,
        "heldout_sha256": dump("heldout.jsonl", held), "sft_sha256": dump("sft.jsonl", sft),
        "heldout_seeds": [e["seed"] for e in held], "heldout_profiles": dict(Counter(e["profile"] for e in held)),
        "sft_seed_range": [min(e["seed"] for e in sft), max(e["seed"] for e in sft)], "n_sft": len(sft),
        "sft_profiles": dict(Counter(e["profile"] for e in sft)),
        "heldout_expert_mean_reward": round(sum(e["expert"]["reward"] for e in held) / len(held), 4),
        "heldout_prompt_tokens_max": max(e["prompt_tokens"] for e in held),
        "sft_prompt_tokens_max": max(e["prompt_tokens"] for e in sft),
        "sft_target_tokens_max": max(e["target_tokens"] for e in sft),
        "sft_seq_tokens_mean": round(sum(e["prompt_tokens"] + e["target_tokens"] for e in sft) / len(sft)),
        "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "heldout_seeds"}, indent=1))


if __name__ == "__main__":
    main()
