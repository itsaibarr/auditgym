"""Round 2 data (CPU only, deterministic): config-3 SFT set and a fresh held-out set B.

    uv run --with tokenizers python training/make_data3.py

- sft3.jsonl: train seeds from 100000 (same episode family and prompt as round 1). Target = the truth-free ExpertPolicy's
  findings restricted to duplicate_invoice ({"findings": []} when there are none). Balanced 50/50 between episodes whose
  target is non-empty and episodes whose target is empty, interleaved so every accumulation group of 8 has 4 of each.
- heldout_b.jsonl: 60 new held-out episodes from seed 600000, same profile quota and length rule as set A (heldout.jsonl).
  Set B is built once, before any config-3 result exists, and is used only for the pre-registered confirmatory check.
Writes training/data_manifest3.json (seeds, sha256, counts).
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
from episodes import HELDOUT_MIX, TRAIN_SEED0, config, profile_cycle  # noqa: E402
from make_data import MAX_PROMPT_TOKENS, MAX_TARGET_TOKENS, MODEL_REPO, MODEL_REV, REWARD_FILTER, TEMPLATE_OVERHEAD  # noqa: E402
from play import expert_decide, play  # noqa: E402

HELDOUT_B_SEED0 = 600_000
N_POS, N_NEG = 320, 320          # episodes with / without a duplicate_invoice target
KEEP_TYPES = ("duplicate_invoice",)


def main():
    tok = Tokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = HERE / "data"
    held_a = {json.loads(l)["seed"] for l in (out / "heldout.jsonl").read_text().splitlines()}
    stats = Counter()

    def build(seed, profile):
        cfg = config(seed, profile)
        try:
            obs0, rows, n, g = play(seed, cfg, expert_decide)
        except ValueError:
            stats["invalid_config"] += 1
            return None
        wb = A.render(obs0, rows)
        p = ntok(A.SYSTEM) + ntok(wb) + TEMPLATE_OVERHEAD
        if p > MAX_PROMPT_TOKENS:
            stats["skipped_long"] += 1
            return None
        return dict(seed=seed, profile=profile, config=cfg, episode_id=obs0["manifest"]["episode_id"], scan_steps=n,
                    messages=A.messages(wb), prompt_tokens=p, expert={k: g[k] for k in ("reward", "f1", "tp", "fp", "fn")},
                    n_issues=g["tp"] + g["fn"], _findings=expert_decide(obs0, rows)["findings"])

    # held-out set B: same quota and rules as set A, new seeds
    need, held_b, seed = dict(HELDOUT_MIX), [], HELDOUT_B_SEED0
    while any(need.values()):
        for profile in [p for p, k in need.items() if k > 0]:
            e = build(seed, profile)
            seed += 1
            if e is not None:
                e.pop("_findings")
                held_b.append(e)
                need[profile] -= 1
    # config-3 SFT set
    pos, neg, seed, cyc = [], [], TRAIN_SEED0, profile_cycle()
    while len(pos) < N_POS or len(neg) < N_NEG:
        profile = next(cyc)
        e = build(seed, profile)
        seed += 1
        if e is None:
            continue
        if e["expert"]["reward"] < REWARD_FILTER:
            stats["rejected_by_reward"] += 1
            continue
        kept = [f for f in e["_findings"] if f["type"] in KEEP_TYPES]
        target = A.target_text(kept)
        if ntok(target) > MAX_TARGET_TOKENS:
            stats["skipped_long_target"] += 1
            continue
        row = dict(seed=e["seed"], profile=profile, messages=e["messages"], target=target, prompt_tokens=e["prompt_tokens"],
                   target_tokens=ntok(target), n_target_findings=len(kept), n_expert_findings=len(e["_findings"]),
                   expert_reward=e["expert"]["reward"])
        if kept and len(pos) < N_POS:
            pos.append(row)
        elif not kept and len(neg) < N_NEG:
            neg.append(row)
        else:
            stats["surplus"] += 1
    sft = [r for pair in zip(pos, neg) for r in pair]
    seeds_b, seeds_sft = {e["seed"] for e in held_b}, {r["seed"] for r in sft}
    assert not (seeds_b & seeds_sft) and not (seeds_b & held_a) and not (held_a & seeds_sft)

    def dump(name, rows):
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (out / name).write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    manifest = {
        "model": MODEL_REPO, "model_revision": MODEL_REV, "keep_types": KEEP_TYPES,
        "heldout_b_sha256": dump("heldout_b.jsonl", held_b), "sft3_sha256": dump("sft3.jsonl", sft),
        "heldout_b_seeds": [e["seed"] for e in held_b], "heldout_b_profiles": dict(Counter(e["profile"] for e in held_b)),
        "sft3_seed_range": [min(seeds_sft), max(seeds_sft)], "n_sft3": len(sft), "n_pos": len(pos), "n_neg": len(neg),
        "sft3_neg_profiles": dict(Counter(r["profile"] for r in neg)), "sft3_pos_profiles": dict(Counter(r["profile"] for r in pos)),
        "sft3_target_findings": sum(r["n_target_findings"] for r in sft),
        "sft3_seq_tokens_mean": round(sum(r["prompt_tokens"] + r["target_tokens"] for r in sft) / len(sft)),
        "heldout_b_expert_mean_reward": round(sum(e["expert"]["reward"] for e in held_b) / len(held_b), 4),
        "stats": dict(stats),
    }
    (HERE / "data_manifest3.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "heldout_b_seeds"}, indent=1))


if __name__ == "__main__":
    main()
