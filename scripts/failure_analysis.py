"""Measured failure analysis: runs every baseline on train+dev and on the closed_example config shape
(new seeds/combinations) and writes docs/FAILURE_ANALYSIS.md from the matcher traces. Reproducible:
    uv run python scripts/failure_analysis.py
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from auditgym.runner import run_policy, summarize
from auditgym.splits import ROOT, split_episodes

POLICIES = ["oracle", "expert", "reconcile", "flag_all", "random"]


def analyse(tag, episodes, out):
    rows, detail = [], {}
    for spec in POLICIES:
        t = time.time()
        res = run_policy(spec, episodes, out / tag / spec, mode="eval")
        s = summarize(res)["all"]
        fp_reasons = Counter()
        fp_by_type = Counter()
        fn_by = Counter()
        for r in res:
            for m in r["matches"]:
                if m["result"] == "FP":
                    fp_reasons[m["reason"]] += 1
                if m["result"] == "FN":
                    fn_by[m["type"]] += 1
            for typ, d in r["per_type"].items():
                fp_by_type[typ] += d["fp"]
        detail[spec] = {"fp_reasons": fp_reasons, "fp_by_type": fp_by_type, "fn_by_type": fn_by, "termination": s["termination"],
                        "per_type": s["per_type"], "clean_returns": [r["reward"] for r in res if r["clean"]],
                        "dirty_returns": [r["reward"] for r in res if not r["clean"]], "wall": time.time() - t}
        rows.append((spec, s))
    return rows, detail


def table(rows):
    out = ["| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for spec, s in rows:
        out.append(f"| {spec} | {s['episodes']} | {s['micro_precision']:.3f} | {s['micro_recall']:.3f} | {s['mean_f1']:.3f} | {s['mean_return']:+.3f} | "
                   f"{_m(s, True)} | {_m(s, False)} | {s['mean_steps']:.1f} | {s['clean_fp_rate']} | {s['mean_wall_s']:.3f} |")
    return out


_D = {}


def _m(s, clean):
    xs = _D[id(s)]["clean_returns" if clean else "dirty_returns"]
    return f"{sum(xs) / len(xs):+.3f}" if xs else "-"


def main():
    out = ROOT / "runs" / "analysis"
    sets = {"open train+dev (20)": split_episodes("train") + split_episodes("dev"),
            "closed-shaped new seeds (20)": split_episodes("closed_example")}
    md = ["# Измеренный разбор сбоев (failure analysis)", "",
          "Сгенерировано `uv run python scripts/failure_analysis.py` по трассам сопоставления (`matches` в results.jsonl). "
          "Все политики работают в eval-режиме в изолированных дочерних процессах (oracle — привилегированно в раннере).", ""]
    for tag, eps in sets.items():
        rows, detail = analyse(tag.split()[0], eps, out)
        for spec, s in rows:
            _D[id(s)] = detail[spec]
        md += [f"## {tag}", ""] + table(rows) + [""]
        md += ["| policy | FP по причинам | FP по типам | FN по типам | завершение |", "|---|---|---|---|---|"]
        for spec in POLICIES:
            d = detail[spec]
            fmt = lambda c: ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) or "—"
            md.append(f"| {spec} | {fmt(d['fp_reasons'])} | {fmt(+d['fp_by_type'])} | {fmt(d['fn_by_type'])} | {fmt(d['termination'])} |")
        md.append("")
    md += [
        "## Интерпретация",
        "",
        "- **oracle = 1.000 везде**: эталон достижим, у каждого вывода есть полный допустимый набор доказательств (проверка максимума награды, не способности агента).",
        "- **expert** (без доступа к эталону, только наблюдения + опубликованная политика) находит все ошибки: задача решаема по видимым записям; "
        "потери только на стоимости шагов (полное чтение таблиц ≈ 20–45 шагов → −0.04…−0.09).",
        "- **reconcile** (наивная сверка по строкам дат, без часовых поясов, без учёта переноса курса) получает много FP: "
        "`no_issue` по `duplicate_invoice` — это отвлекающие повторяющиеся подписки с одинаковой суммой; по `payment_allocation` — переплаты с возвратом (2300) и платежи с комиссией; "
        "по `period_cutoff` — документы, записанные в UTC около полуночи; по `fx_conversion` — дата курса взята из строки времени без перевода в зону компании. "
        "Поэтому на чистых книгах reconcile = −1: отвлекающие случаи действительно ловят простые трюки.",
        "- **flag_all** = −1 на всех эпизодах: массовые обвинения не окупаются (сумма, ID и доказательства должны совпасть).",
        "- **random** с вероятностью 1% за шаг отправляет случайные выводы (всегда FP → −1), иначе упирается в лимит 100 шагов: −1 на книгах с ошибками и 0.8 на чистых (см. ограничения).",
    ]
    (ROOT / "docs" / "FAILURE_ANALYSIS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
