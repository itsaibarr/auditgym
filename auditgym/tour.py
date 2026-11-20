"""`auditgym tour`: the jury's whole flow in about a minute, ending with a scorecard per case criterion.

1. the official offline check (`auditgym check`)
2. a closed evaluation shaped like the organizers': fresh secret seeds and salt (never written anywhere),
   20 hidden-like episodes (5 clean, 5 interacting), the four organizer policy types + our truth-free expert
3. 20 paired checks (row permutation + ID rename) on those closed episodes
4. byte-identical replay with re-grading, run-dir integrity and leakscan of the closed runs
Writes <out>/report.md and <out>/report.json and prints copy-paste next steps for plugging in your own agent.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import secrets
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .canon import dumps
from .episode import Episode, generate
from .splits import ROOT, load_split_config

CRITERIA = [
    ("contract", "Контракт и изоляция", 25),
    ("data", "Данные и эталон", 25),
    ("reward", "Надёжность награды", 25),
    ("repro", "Воспроизводимость и обобщение", 15),
    ("eng", "Инженерия и объяснение", 10),
]
POLICIES = ["random", "reconcile", "flag_all", "oracle", "expert"]
POLICY_RU = {"random": "random", "reconcile": "reconcile", "flag_all": "flag_all", "oracle": "oracle*", "expert": "expert**"}


def _criterion(check_id: str) -> str:
    if check_id.startswith(("tests.test_data", "data.")):
        return "data"
    if check_id.startswith(("tests.test_reward", "reward.", "baseline.")):
        return "reward"
    if check_id.startswith(("tests.test_repro", "repro.")):
        return "repro"
    if check_id.startswith(("tests.test_contract", "tests.test_isolation", "tests.test_hardening", "tests.test_gymnasium",
                            "isolation.", "contract.")):
        return "contract"
    return "eng"


def _out_dir(out: str | None) -> Path:
    for cand in ([Path(out)] if out else [Path.cwd() / "auditgym-report", Path(tempfile.gettempdir()) / "auditgym-report"]):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=cand):
                pass
            return cand.resolve()
        except OSError:
            continue
    raise OSError("no writable output directory; pass --out DIR")


def _closed_episodes() -> list[tuple[str, Episode]]:
    """The organizers' shape (configs/closed_example.json: 5 clean, 5 interacting, 10 other) with NEW random seeds
    and a random salt that exists only in this process."""
    salt = secrets.token_hex(32)
    split = load_split_config(str(ROOT / "configs" / "closed_example.json"))
    base = split.get("base_config", {})
    eps = []
    for e in split["episodes"]:
        while True:
            try:
                ep = generate(f"tour-{secrets.token_hex(8)}", {**base, **e.get("config", {})}, salt=salt,
                              config_version=split.get("config_version"))
                break
            except ValueError:  # a config/seed the fail-closed generator refuses: draw another seed
                continue
        eps.append(("closed", ep))
    return sorted(eps, key=lambda x: x[1].episode_id)


def _replay_ok(run_dir: Path, eps: dict) -> tuple[int, int]:
    from .env import AuditEnv
    recs = [json.loads(l) for l in (run_dir / "trajectories.jsonl").read_text().splitlines() if l]
    results = {r["episode_id"]: r for r in (json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l)}
    by_ep: dict[str, list] = {}
    for r in recs:
        by_ep.setdefault(r["episode_id"], []).append(r)
    ok_n = 0
    for eid, steps in by_ep.items():
        env = AuditEnv("eval")
        obs, _ = env.reset(options={"episode": eps[eid]})
        ok = dumps(obs) == dumps(steps[0]["observation"])
        for r in steps[1:]:
            got = env.abort(r["info"].get("error_code") or "timeout") if r["action"] is None else env.step(r["action"])
            ok &= dumps(list(got[:4])) == dumps([r["observation"], r["reward"], r["terminated"], r["truncated"]])
        g = env.result()
        ok &= all(g[k] == results[eid][k] for k in ("tp", "fp", "fn", "reward"))
        ok_n += ok
    return ok_n, len(by_ep)


def run_tour(out: str | None = None, quick: bool = False) -> int:
    from .checks import run_check
    from .leakscan import scan_run
    from .perturb import perturb
    from .runner import code_hash, run_episode, run_policies, summarize, verify_run

    con = Console(highlight=False)
    t_all = time.time()
    outdir = _out_dir(out)
    report: dict = {"auditgym_version": __version__, "code_hash": code_hash(), "python": platform.python_version(),
                    "platform": platform.platform(), "steps": {}}
    con.print(Panel.fit(
        f"[bold]AuditGym {__version__}[/] — тур жюри: весь путь проверки за ~1 минуту\n"
        f"Python {platform.python_version()} · {platform.system()} {platform.machine()} · код {report['code_hash']}\n"
        "Сеть, GPU, ключи и платные API не нужны. Отчёт: " + str(outdir),
        border_style="cyan"))

    # 1. official check -------------------------------------------------------------------------------------------
    con.rule("[bold]1/4  Официальная проверка[/]  (auditgym check: pytest, базовые политики, порядки наград, leakscan, парные проверки, replay)")
    t0 = time.time()
    with con.status("идёт auditgym check …"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rc = run_check(quick=quick, out=outdir / "check")
    summary = json.loads((outdir / "check" / "summary.json").read_text())
    by_c = {k: [0, 0] for k, _, _ in CRITERIA}
    for it in summary["checks"]:
        c = by_c[_criterion(it["id"])]
        c[0] += it["ok"]
        c[1] += 1
    tb = Table(show_header=True, header_style="bold")
    tb.add_column("Критерий кейса")
    tb.add_column("Проверок пройдено", justify="right")
    for k, name, _ in CRITERIA:
        p, n = by_c[k]
        tb.add_row(name, f"[{'green' if p == n else 'red'}]{p}/{n}[/]")
    con.print(tb)
    n_ok, n_all = sum(i["ok"] for i in summary["checks"]), len(summary["checks"])
    con.print(f"{'[green]PASS[/]' if rc == 0 else '[red]FAIL[/]'}: {n_ok}/{n_all} проверок за {time.time() - t0:.0f} с · "
              f"подробно: {outdir / 'check' / 'summary.md'}")
    report["steps"]["check"] = {"ok": rc == 0, "passed": n_ok, "total": n_all, "by_criterion": by_c}

    # 2. closed evaluation like the organizers' ---------------------------------------------------------------------
    con.rule("[bold]2/4  Закрытая проверка как у организаторов[/]  (новые секретные сиды и соль, 20 эпизодов: 5 чистых, 5 со связанными ошибками)")
    t0 = time.time()
    with con.status("генерация 20 скрытых эпизодов и прогон 5 политик в изолированных процессах …"), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        eps = _closed_episodes()
        res = run_policies(POLICIES, eps, outdir / "closed", mode="eval", meta={"sources": "tour (random secret seeds)"})
    rows = {}
    tb = Table(show_header=True, header_style="bold")
    for h, j in (("Политика", "left"), ("P", "right"), ("R", "right"), ("F1", "right"), ("Награда", "right"),
                 ("На чистых", "right"), ("Шаги", "right"), ("с/эпизод", "right")):
        tb.add_column(h, justify=j)
    for spec in POLICIES:
        s = summarize(res[spec])["all"]
        clean = [r["reward"] for r in res[spec] if r["clean"]]
        rows[spec] = {**{k: s[k] for k in ("micro_precision", "micro_recall", "mean_f1", "mean_return", "mean_steps", "mean_wall_s")},
                      "clean_return": round(sum(clean) / len(clean), 4)}
        r = rows[spec]
        tb.add_row(POLICY_RU[spec], f"{r['micro_precision']:.3f}", f"{r['micro_recall']:.3f}", f"{r['mean_f1']:.3f}",
                   f"{r['mean_return']:+.3f}", f"{r['clean_return']:+.3f}", f"{r['mean_steps']:.1f}", f"{r['mean_wall_s']:.3f}")
    con.print(tb)
    order_ok = (all(r["reward"] == 1.0 for r in res["oracle"]) and all(r["reward"] == -1.0 for r in res["flag_all"] if r["clean"])
                and rows["expert"]["mean_return"] > rows["reconcile"]["mean_return"] > max(rows["flag_all"]["mean_return"], rows["random"]["mean_return"]) - 1e-9)
    con.print("[dim]* oracle видит эталон (только для оценщика) · ** expert — наш решатель по видимым записям, без эталона[/]")
    con.print(f"{'[green]✓[/]' if order_ok else '[red]✗[/]'} oracle = 1.0 на всех 20; flag_all = −1 на чистых; "
              f"expert > reconcile ≥ атаки · {time.time() - t0:.0f} с")
    report["steps"]["closed_eval"] = {"ok": order_ok, "policies": rows, "episodes": len(eps)}

    # 3. paired checks ---------------------------------------------------------------------------------------------
    con.rule("[bold]3/4  20 парных проверок[/]  (перестановка строк + смена всех ID на закрытых эпизодах)")
    t0 = time.time()
    bad = []
    with con.status("парные проверки …"):
        for k, (_, ep) in enumerate(eps):
            pub2, truth2, _ = perturb(ep.public, ep.truth, seed=secrets.randbits(32))
            ep2 = Episode(pub2, truth2, ep.provenance, {})
            a = run_episode("oracle", ep2, "eval").result
            b, b2 = (run_episode("expert", e, "eval", isolate=False).result for e in (ep, ep2))
            if a["reward"] != 1.0 or (b["tp"], b["fp"], b["fn"], b["reward"]) != (b2["tp"], b2["fp"], b2["fn"], b2["reward"]):
                bad.append(ep.episode_id)
    paired_ok = not bad
    con.print(f"{'[green]✓[/]' if paired_ok else '[red]✗[/]'} {len(eps) - len(bad)}/{len(eps)}: oracle = 1.0 после смены ID, "
              f"исход expert не меняется · {time.time() - t0:.0f} с")
    report["steps"]["paired"] = {"ok": paired_ok, "passed": len(eps) - len(bad), "total": len(eps)}

    # 4. replay, integrity, leakscan ---------------------------------------------------------------------------------
    con.rule("[bold]4/4  Повтор, целостность, утечки[/]  (побайтный replay с перепроверкой оценки, хеши run.json, leakscan)")
    t0 = time.time()
    epmap = {ep.episode_id: ep for _, ep in eps}
    rep_ok, rep_n = _replay_ok(outdir / "closed" / "reconcile", epmap)
    integ = [p for spec in POLICIES for p in verify_run(outdir / "closed" / spec)]
    leaks = [p for spec in POLICIES for p in scan_run(outdir / "closed" / spec)]
    tail_ok = rep_ok == rep_n and not integ and not leaks
    con.print(f"{'[green]✓[/]' if rep_ok == rep_n else '[red]✗[/]'} replay: {rep_ok}/{rep_n} эпизодов побайтно, оценка совпадает")
    con.print(f"{'[green]✓[/]' if not integ else '[red]✗[/]'} целостность: {len(POLICIES) - len({p.split(':')[0] for p in integ})}/{len(POLICIES)} папок прогона сходятся с run.json")
    con.print(f"{'[green]✓[/]' if not leaks else '[red]✗[/]'} leakscan: {len(leaks)} проблем в наблюдениях закрытых прогонов · {time.time() - t0:.0f} с")
    report["steps"]["replay_integrity_leaks"] = {"ok": tail_ok, "replay": [rep_ok, rep_n], "integrity_problems": integ[:5], "leaks": leaks[:5]}

    # scorecard -------------------------------------------------------------------------------------------------------
    all_ok = rc == 0 and order_ok and paired_ok and tail_ok
    evidence = {
        "contract": f"{by_c['contract'][0]}/{by_c['contract'][1]} проверок; утечек {len(leaks)}",
        "data": f"{by_c['data'][0]}/{by_c['data'][1]} проверок; oracle 1.0 на 20 новых",
        "reward": f"{by_c['reward'][0]}/{by_c['reward'][1]} проверок; flag_all −1",
        "repro": f"парные {len(eps) - len(bad)}/{len(eps)}; replay {rep_ok}/{rep_n}",
        "eng": "1 команда; версии и хеши; траектории; разбор сбоев",
    }
    con.rule("[bold]Итог по критериям кейса[/]")
    tb = Table(show_header=True, header_style="bold")
    tb.add_column("Критерий")
    tb.add_column("Баллы")
    tb.add_column("Статус")
    tb.add_column("Доказательство")
    for k, name, pts in CRITERIA:
        ok_k = by_c[k][0] == by_c[k][1] and (k != "repro" or (paired_ok and rep_ok == rep_n)) and (k != "contract" or not leaks)
        tb.add_row(name, str(pts), "[green]✓[/]" if ok_k else "[red]✗[/]", evidence[k])
    con.print(tb)
    con.print(f"[bold]{'[green]ВСЁ ПРОЙДЕНО[/]' if all_ok else '[red]ЕСТЬ ПРОВАЛЫ[/]'}[/] за {time.time() - t_all:.0f} с · "
              f"отчёт: {outdir / 'report.md'}")
    report.update({"ok": all_ok, "seconds": round(time.time() - t_all, 1), "evidence": evidence})
    _write_report(outdir, report, rows)

    ag = os.environ.get("AUDITGYM_LAUNCHER", "auditgym")  # how the user started us (npx / run.sh / docker / auditgym)
    ex = ROOT / "examples"
    con.print(Panel(
        f"[bold]Подключить свою политику[/] (класс Python с reset/step или любая программа, JSON-строки через stdin/stdout):\n"
        f"  {ag} run --policy my_policy:LookAround --policy-path {ex} --split dev --trust-policies\n"
        f"  {ag} run --policy \"cmd:python3 {ex / 'stdio_policy.py'}\" --split dev --trust-policies\n"
        f"[bold]Свои сиды и параметры[/] (формат: {ROOT / 'configs' / 'closed_example.json'}; split через stdin):\n"
        f"  {ag} run --policy reconcile,flag_all,random,oracle --split - --public-salt --out out < split.json\n"
        f"[bold]Сравнить[/]:  {ag} compare out/reconcile out/flag_all   ·   [bold]вручную[/]:  {ag} play   ·   "
        f"[bold]схемы для LLM[/]:  {ag} schema\n"
        f"Строгая закрытая проверка — в Docker (README, раздел «Закрытая проверка»). Документация: {ROOT / 'README.md'}",
        title="Что дальше", border_style="cyan"))
    return 0 if all_ok else 1


def _write_report(outdir: Path, report: dict, rows: dict) -> None:
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md = [f"# AuditGym {report['auditgym_version']} — отчёт тура жюри: {'PASS' if report['ok'] else 'FAIL'}", "",
          f"Python {report['python']} · {report['platform']} · код {report['code_hash']} · {report['seconds']} с", "",
          "## Итог по критериям кейса", "", "| Критерий | Баллы | Доказательство |", "|---|---|---|"]
    md += [f"| {name} | {pts} | {report['evidence'][k]} |" for k, name, pts in CRITERIA]
    md += ["", "## Закрытая проверка (20 новых эпизодов, 5 чистых, 5 связанных)", "",
           "| Политика | P | R | F1 | Награда | На чистых | Шаги | с/эпизод |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {POLICY_RU[s]} | {r['micro_precision']:.3f} | {r['micro_recall']:.3f} | {r['mean_f1']:.3f} | {r['mean_return']:+.3f} | "
           f"{r['clean_return']:+.3f} | {r['mean_steps']:.1f} | {r['mean_wall_s']:.3f} |" for s, r in rows.items()]
    st = report["steps"]
    md += ["", "## Шаги", "", f"- Официальная проверка: {st['check']['passed']}/{st['check']['total']}",
           f"- Парные проверки: {st['paired']['passed']}/{st['paired']['total']}",
           f"- Replay: {st['replay_integrity_leaks']['replay'][0]}/{st['replay_integrity_leaks']['replay'][1]}; "
           f"проблем целостности: {len(st['replay_integrity_leaks']['integrity_problems'])}; утечек: {len(st['replay_integrity_leaks']['leaks'])}",
           "", "Прогоны: `check/`, `closed/<политика>/` (trajectories.jsonl, results.jsonl, summary.json, run.json)."]
    (outdir / "report.md").write_text("\n".join(md) + "\n")
