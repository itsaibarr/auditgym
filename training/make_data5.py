"""Round 3, config 5 (disclosed new round, PROTOCOL.md addendum 4): confirmation set C + SFT set sft5.

    uv run --with tokenizers python training/make_data5.py

- heldout_c.jsonl: 60 new held-out episodes from seed 700000, same profile quota and length rule as sets A and B. Built
  once, before any config-5 result exists; the primary set for config 5.
- sft5.jsonl: 640 train samples from seed 100000 (same family, prompt, length rule and reward filter). Target = the
  truth-free ExpertPolicy's findings of types duplicate_invoice + period_cutoff + payment_allocation. Four equal groups of
  160, interleaved: target has a payment_allocation finding / else has a period_cutoff finding / else duplicates only /
  empty target.
Writes training/data_manifest5.json (seeds, sha256, counts).
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

HELDOUT_C_SEED0 = 700_000
KEEP_TYPES = ("duplicate_invoice", "period_cutoff", "payment_allocation")
ORDER = ("payalloc", "cutoff", "dup_only", "empty")
N_EACH = {k: 160 for k in ORDER}


def main():
    tok = Tokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = HERE / "data"
    held_ab = {json.loads(l)["seed"] for f in ("heldout.jsonl", "heldout_b.jsonl") for l in (out / f).read_text().splitlines()}
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

    # held-out set C: same quota and rules as sets A and B, new seeds
    need, held_c, seed = dict(HELDOUT_MIX), [], HELDOUT_C_SEED0
    while any(need.values()):
        for profile in [p for p, k in need.items() if k > 0]:
            e = build(seed, profile)
            seed += 1
            if e is not None:
                e.pop("_findings")
                held_c.append(e)
                need[profile] -= 1
    # config-5 SFT set
    groups, seed, cyc = {k: [] for k in ORDER}, TRAIN_SEED0, profile_cycle()
    while any(len(groups[k]) < n for k, n in N_EACH.items()):
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
        types = {f["type"] for f in kept}
        k = ("payalloc" if "payment_allocation" in types else "cutoff" if "period_cutoff" in types
             else "dup_only" if kept else "empty")
        if len(groups[k]) >= N_EACH[k]:
            stats["surplus"] += 1
            continue
        groups[k].append(dict(seed=e["seed"], profile=profile, messages=e["messages"], target=target,
                              prompt_tokens=e["prompt_tokens"], target_tokens=ntok(target), n_target_findings=len(kept),
                              group=k, target_types=dict(Counter(f["type"] for f in kept))))
    sft = [groups[k][i] for i in range(max(N_EACH.values())) for k in ORDER if i < len(groups[k])]
    seeds_c, seeds_sft = {e["seed"] for e in held_c}, {r["seed"] for r in sft}
    assert not (seeds_c & seeds_sft) and not (seeds_c & held_ab) and not (held_ab & seeds_sft)

    def dump(name, rows):
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (out / name).write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    tt = Counter()
    for r in sft:
        tt.update(r["target_types"])
    manifest = {
        "model": MODEL_REPO, "model_revision": MODEL_REV, "keep_types": KEEP_TYPES,
        "heldout_c_sha256": dump("heldout_c.jsonl", held_c), "sft5_sha256": dump("sft5.jsonl", sft),
        "heldout_c_seeds": [e["seed"] for e in held_c], "heldout_c_profiles": dict(Counter(e["profile"] for e in held_c)),
        "heldout_c_expert_mean_reward": round(sum(e["expert"]["reward"] for e in held_c) / len(held_c), 4),
        "heldout_c_books_without_errors": sum(e["n_issues"] == 0 for e in held_c),
        "n_sft5": len(sft), "groups": {k: len(v) for k, v in groups.items()}, "target_findings_by_type": dict(tt),
        "sft5_seed_range": [min(seeds_sft), max(seeds_sft)],
        "empty_group_profiles": dict(Counter(r["profile"] for r in groups["empty"])),
        "sft5_seq_tokens_mean": round(sum(r["prompt_tokens"] + r["target_tokens"] for r in sft) / len(sft)),
        "stats": dict(stats),
    }
    (HERE / "data_manifest5.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "heldout_c_seeds"}, indent=1))


if __name__ == "__main__":
    main()
