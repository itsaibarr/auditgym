"""Grade the Kaggle generations with the env's own grader and compare arms with a paired bootstrap.

    uv run python training/grade.py            # reads training/results/kaggle/{base,after}/gen_*.jsonl

Every generation is replayed through a fresh AuditEnv in eval mode: the same scan actions the scaffold used to build
the prompt, then submit(<the model's JSON object>) . The grade (case reward, F1, validity) comes from env.result().
Writes training/results/episodes.jsonl and training/results/metrics.json.
"""
from __future__ import annotations

import json
import random
from collections import Counter
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import agpolicy as A  # noqa: E402
from play import expert_decide, play  # noqa: E402

GROUPS = {"clean": "clean", "duplicate_invoice": "single", "payment_allocation": "single", "fx_conversion": "single",
          "period_cutoff": "single", "mixed": "multi", "X1": "multi", "X2": "multi"}
B, CI_SEED = 10_000, 20261121


def load_gens(arm: str) -> dict:
    out = {}
    for f in sorted((HERE / "results" / "kaggle" / arm).glob("gen_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[r["seed"]] = r
    return out


def lenient(gens: dict) -> dict:
    """Exploratory, not pre-registered: add a blank `explanation` to findings that lack it (the grader never reads it)."""
    out = {}
    for seed, g in gens.items():
        p, ok = A.to_submission(g["text"], g["episode_id"])
        if ok and isinstance(p["findings"], list):
            p["findings"] = [{**f, "explanation": f.get("explanation", "")} if isinstance(f, dict) else f for f in p["findings"]]
            g = {**g, "text": json.dumps({"findings": p["findings"]})}
        out[seed] = g
    return out


def grade_arm(held: list[dict], gens: dict | None) -> list[dict]:
    rows = []
    for e in held:
        g = gens.get(e["seed"]) if gens is not None else None
        meta = {"seed": e["seed"], "profile": e["profile"], "group": GROUPS[e["profile"]], "n_issues": e["n_issues"]}
        if gens is None:            # reference rows
            decide = expert_decide if e.get("_ref") == "expert" else (
                lambda o, r: {"episode_id": o["manifest"]["episode_id"], "findings": []})
            parse_ok, missing, extra = True, False, {}
        elif g is None:             # pre-registered: a missing generation is graded as an empty answer
            decide, parse_ok, missing, extra = (lambda o, r: {"episode_id": o["manifest"]["episode_id"], "findings": []}), False, True, {}
        else:
            payload, parse_ok = A.to_submission(g["text"], e["episode_id"])
            decide, missing = (lambda o, r, p=payload: p), False
            extra = {"new_tokens": g["new_tokens"], "hit_max_new_tokens": g["hit_max_new_tokens"],
                     "n_findings": len(payload["findings"]) if parse_ok and isinstance(payload["findings"], list) else None}
        obs0, _, n_scan, res = play(e["seed"], e["config"], decide)
        assert obs0["manifest"]["episode_id"] == e["episode_id"]
        rows.append({**meta, "reward": res["reward"], "f1": res["f1"], "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                     "valid": res["valid"], "parse_ok": parse_ok, "missing": missing,
                     "steps": res["inspection_steps"] + 1,
                     "fp_reasons": [m["reason"] for m in res["matches"] if m["result"] == "FP"],
                     **extra})
    return rows


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    clean = [r for r in rows if r["n_issues"] == 0]
    s = {"n": n, "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
         "mean_f1": round(sum(r["f1"] for r in rows) / n, 4),
         "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None,
         "tp": tp, "fp": fp, "fn": fn,
         "schema_error_rate": round(sum(not r["valid"] for r in rows) / n, 4),
         "unparseable_rate": round(sum(not r["parse_ok"] for r in rows) / n, 4),
         "mean_steps": round(sum(r["steps"] for r in rows) / n, 2),
         "fp_reasons": dict(sorted(Counter(x for r in rows for x in r["fp_reasons"]).items())),
         "clean_books_false_accusation_rate": round(sum(r["fp"] > 0 for r in clean) / len(clean), 4) if clean else None}
    if any("new_tokens" in r for r in rows):
        s["mean_new_tokens"] = round(sum(r.get("new_tokens", 0) for r in rows) / n, 1)
        s["hit_max_new_tokens"] = sum(bool(r.get("hit_max_new_tokens")) for r in rows)
        s["missing_generations"] = sum(r["missing"] for r in rows)
    return s


def paired_bootstrap(a: list[float], b: list[float]) -> dict:
    """Percentile bootstrap over episodes of mean(b - a); 10,000 resamples, fixed seed."""
    d = [y - x for x, y in zip(a, b)]
    n, rng = len(d), random.Random(CI_SEED)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"delta_mean": round(sum(d) / n, 4), "ci95": [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B) - 1], 4)],
            "wins": sum(x > 1e-9 for x in d), "ties": sum(abs(x) <= 1e-9 for x in d), "losses": sum(x < -1e-9 for x in d),
            "n": n, "resamples": B, "seed": CI_SEED}


SETS = {  # held-out set -> (file, arms read from results/kaggle/<arm>/gen_*.jsonl, output suffix)
    "A": ("heldout.jsonl", ("base", "after", "after2", "c3", "c4"), ""),
    "B": ("heldout_b.jsonl", ("base_b", "c3_b", "c4_b"), "_b"),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    fname, arm_names, suffix = SETS[which]
    held = [json.loads(l) for l in (HERE / "data" / fname).read_text().splitlines()]
    arms = {}
    for arm in arm_names:
        gens = load_gens(arm)
        if gens:
            arms[arm] = grade_arm(held, gens)
            if arm in ("base", "after", "after2"):
                arms[f"explore_{arm}_lenient_explanation"] = grade_arm(held, lenient(gens))
    arms["ref_expert"] = grade_arm([{**e, "_ref": "expert"} for e in held], None)
    arms["ref_empty"] = grade_arm(held, None)
    metrics = {"set": which, "file": fname, "arms": {}, "by_group": {}}
    for arm, rows in arms.items():
        metrics["arms"][arm] = summary(rows)
        metrics["by_group"][arm] = {g: summary([r for r in rows if r["group"] == g]) for g in ("clean", "single", "multi")}
    base = "base" if which == "A" else "base_b"
    pairs = [(b, a) for a in arm_names if a not in ("base", "base_b") for b in (base, "ref_empty")]
    for b_arm, a_arm in pairs:
        if b_arm not in arms or a_arm not in arms:
            continue
        ra, rb = arms[b_arm], arms[a_arm]
        assert [r["seed"] for r in ra] == [r["seed"] for r in rb]
        key = f"paired_{a_arm}_minus_{b_arm}"
        metrics[key] = {k: paired_bootstrap([r[k] for r in ra], [r[k] for r in rb]) for k in ("reward", "f1")}
        metrics[key + "_by_group"] = {
            g: paired_bootstrap([r["reward"] for r in ra if r["group"] == g], [r["reward"] for r in rb if r["group"] == g])
            for g in ("clean", "single", "multi")}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"episodes{suffix}.jsonl", "w") as f:
        for arm, rows in arms.items():
            for r in rows:
                f.write(json.dumps({"arm": arm, **r}, sort_keys=True) + "\n")
    (out / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=1) + "\n")
    print("| arm | n | mean reward | mean F1 | micro-F1 | schema-error rate | unparseable | steps | clean FP rate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, s in metrics["arms"].items():
        print(f"| {arm} | {s['n']} | {s['mean_reward']} | {s['mean_f1']} | {s['micro_f1']} | {s['schema_error_rate']} | "
              f"{s['unparseable_rate']} | {s['mean_steps']} | {s['clean_books_false_accusation_rate']} |")
    for k, v in metrics.items():
        if k.startswith("paired_") and not k.endswith("_by_group"):
            print(k, {m: (x["delta_mean"], x["ci95"], (x["wins"], x["ties"], x["losses"])) for m, x in v.items()})


if __name__ == "__main__":
    main()
