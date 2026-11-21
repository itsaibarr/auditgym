"""Round 2, config 4 (disclosed follow-up): SFT targets = ExpertPolicy findings of types duplicate_invoice + period_cutoff.

    uv run --with tokenizers python training/make_data4.py

640 train samples from seed 100000 (same family, prompt and length rule), balanced in three equal groups and interleaved:
target has a period_cutoff finding / target has duplicates only / target empty. Writes data/sft4.jsonl and
data_manifest4.json. Evaluated on the same held-out sets A and B as config 3.
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
from episodes import TRAIN_SEED0, config, profile_cycle  # noqa: E402
from make_data import MAX_PROMPT_TOKENS, MAX_TARGET_TOKENS, MODEL_REPO, MODEL_REV, REWARD_FILTER, TEMPLATE_OVERHEAD  # noqa: E402
from play import expert_decide, play  # noqa: E402

KEEP_TYPES = ("duplicate_invoice", "period_cutoff")
N_EACH = {"cutoff": 214, "dup_only": 213, "empty": 213}


def main():
    tok = Tokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = HERE / "data"
    held = {json.loads(l)["seed"] for f in ("heldout.jsonl", "heldout_b.jsonl") for l in (out / f).read_text().splitlines()}
    groups, stats, seed, cyc = {k: [] for k in N_EACH}, Counter(), TRAIN_SEED0, profile_cycle()
    while any(len(groups[k]) < n for k, n in N_EACH.items()):
        profile = next(cyc)
        cfg = config(seed, profile)
        s, seed = seed, seed + 1
        try:
            obs0, rows, _, g = play(s, cfg, expert_decide)
        except ValueError:
            stats["invalid_config"] += 1
            continue
        wb = A.render(obs0, rows)
        p = ntok(A.SYSTEM) + ntok(wb) + TEMPLATE_OVERHEAD
        if p > MAX_PROMPT_TOKENS:
            stats["skipped_long"] += 1
            continue
        if g["reward"] < REWARD_FILTER:
            stats["rejected_by_reward"] += 1
            continue
        kept = [f for f in expert_decide(obs0, rows)["findings"] if f["type"] in KEEP_TYPES]
        target = A.target_text(kept)
        if ntok(target) > MAX_TARGET_TOKENS:
            stats["skipped_long_target"] += 1
            continue
        k = "cutoff" if any(f["type"] == "period_cutoff" for f in kept) else ("dup_only" if kept else "empty")
        if len(groups[k]) >= N_EACH[k]:
            stats["surplus"] += 1
            continue
        groups[k].append(dict(seed=s, profile=profile, messages=A.messages(wb), target=target, prompt_tokens=p,
                              target_tokens=ntok(target), n_target_findings=len(kept), group=k,
                              target_types=dict(Counter(f["type"] for f in kept))))
    sft = []
    for i in range(max(N_EACH.values())):
        for k in ("cutoff", "dup_only", "empty"):
            if i < len(groups[k]):
                sft.append(groups[k][i])
    assert not ({r["seed"] for r in sft} & held)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in sft)
    (out / "sft4.jsonl").write_text(text)
    tt = Counter()
    for r in sft:
        tt.update(r["target_types"])
    manifest = {"model": MODEL_REPO, "model_revision": MODEL_REV, "keep_types": KEEP_TYPES,
                "sft4_sha256": hashlib.sha256(text.encode()).hexdigest(), "n_sft4": len(sft),
                "groups": {k: len(v) for k, v in groups.items()}, "target_findings_by_type": dict(tt),
                "sft4_seed_range": [min(r["seed"] for r in sft), max(r["seed"] for r in sft)],
                "empty_group_profiles": dict(Counter(r["profile"] for r in groups["empty"])),
                "sft4_seq_tokens_mean": round(sum(r["prompt_tokens"] + r["target_tokens"] for r in sft) / len(sft)),
                "stats": dict(stats)}
    (HERE / "data_manifest4.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
