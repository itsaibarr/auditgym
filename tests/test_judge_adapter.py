"""The organizers' published checks (genesis-exe.com/fintech-checks.html, 20.11.2026), run through judge_adapter.py:
contract tests C01-C20, matcher/reward tests 1-20, open examples M01-M12 and the paired-check transport."""
import json
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_adapter as ja  # noqa: E402

TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
CLEAN = {"errors": {t: {"on": False} for t in TYPES}}
ONE_DUP = {"errors": {**{t: {"on": False} for t in TYPES}, "duplicate_invoice": {"on": True, "count": [1, 1]}}}


def act(name, **args):
    return {"name": name, "args": args}


@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge")
    out = {}
    for key, seed, cfg in (("C", 11, CLEAN), ("E", 12, ONE_DUP)):
        ja.generate_episode(seed, cfg, str(root / key))
        out[key] = root / key
    return out


def closed(env, d):
    return env.reset(0, {"mode": "closed", "episode_dir": str(d), "team_config": {}})


def jsonable(x):
    return json.loads(json.dumps(x)) == x


# ----------------------------------------------------------------------------- block 1: contract tests C01-C20
def test_c01_c04_reset_manifest_and_step_shapes(eps):
    env = ja.make_env()
    obs, info = closed(env, eps["C"])
    assert jsonable(obs) and jsonable(info)
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
    m = obs["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts", "rounding"):
        assert k in m
    assert "seed" not in json.dumps(obs) and "truth" not in json.dumps(obs)
    assert m["counts"]["ledger"] == sum(1 for _ in open(eps["C"] / "ledger.csv")) - 1
    o, r, term, trunc, info = env.step(act("list_records", table="ledger", cursor=None, limit=50))
    assert isinstance(term, bool) and isinstance(trunc, bool) and r == 0 and jsonable(o) and jsonable(info)
    assert set(o) == {"result", "error", "remaining_steps"} and set(o["result"]) == {"records", "next_cursor"}


def test_c05_c06_pages_cover_all_rows_and_repeat(eps):
    env = ja.make_env()
    closed(env, eps["C"])
    seen, cur, pages = [], None, 0
    while True:
        o = env.step(act("list_records", table="ledger", cursor=cur, limit=50))[0]
        assert len(o["result"]["records"]) <= 50
        seen += [r["row_id"] for r in o["result"]["records"]]
        pages += 1
        cur = o["result"]["next_cursor"]
        if cur is None or pages > 60:
            break
    rows = [l.split(",")[0] for l in open(eps["C"] / "ledger.csv").read().splitlines()[1:]]
    assert pages >= 3 and seen == rows
    env2 = ja.make_env()
    closed(env2, eps["C"])
    c = env2.step(act("list_records", table="ledger", limit=50))[0]["result"]["next_cursor"]
    a = env2.step(act("list_records", table="ledger", cursor=c, limit=50))[0]["result"]["records"]
    b = env2.step(act("list_records", table="ledger", cursor=c, limit=50))[0]["result"]["records"]
    assert a == b


def test_c07_c08_c09_c10_records_documents_calculator(eps):
    env = ja.make_env()
    closed(env, eps["C"])
    r1 = env.step(act("list_records", table="ledger", limit=1))[0]["result"]["records"][0]
    assert env.step(act("get_record", table="ledger", id=r1["row_id"]))[0]["result"] == r1
    d1 = json.loads(open(eps["C"] / "invoices.jsonl").readline())
    assert env.step(act("get_document", doc_id=d1["doc_id"]))[0]["result"]["doc_id"] == d1["doc_id"]
    assert env.step(act("calculator", expression="0.1+0.2"))[0]["result"] == "0.3"
    o = env.step(act("calculator", expression="open('/etc/passwd').read()"))[0]
    assert o["result"] is None and o["error"]["code"] and "root" not in json.dumps(o)


def test_c11_c14_errors_spend_steps_and_do_not_break_access(eps):
    env = ja.make_env()
    obs, _ = closed(env, eps["C"])
    n = obs["remaining_steps"]
    for a in (act("get_record", table="ledger", id="U-not-there"), act("get_document", doc_id="U-not-there"), act("frobnicate")):
        o = env.step(a)[0]
        n -= 1
        assert o["error"] is not None and o["remaining_steps"] == n and "/" not in o["error"]["message"]
    o = env.step(act("list_records", table="ledger", limit=1))[0]
    assert o["error"] is None and o["remaining_steps"] == n - 1


def test_c15_c16_c17_submit_and_limits(eps):
    ep_id = json.loads((eps["C"] / "manifest.json").read_text())["episode_id"]
    env = ja.make_env()
    closed(env, eps["C"])
    o, r, term, trunc, _ = env.step(act("submit", findings={"episode_id": ep_id, "findings": []}))
    assert term and not trunc and o["result"] == {"accepted": True} and r == 0
    assert env.step(act("list_records", table="ledger"))[0]["error"]["code"] == "episode_over"
    env = ja.make_env()
    closed(env, eps["C"])
    for _ in range(99):
        assert env.step(act("calculator", expression="1+1"))[2:4] == (False, False)
    o, r, term, trunc, _ = env.step(act("submit", findings={"episode_id": ep_id, "findings": []}))
    assert term and not trunc and o["result"] == {"accepted": True}
    env = ja.make_env()
    closed(env, eps["C"])
    for i in range(100):
        o, r, term, trunc, _ = env.step(act("calculator", expression="1+1"))
    assert trunc and not term


def test_c18_time_limit_truncates(eps, monkeypatch):
    env = ja.make_env()
    closed(env, eps["C"])
    t = [time.monotonic()]
    env._env._clock = lambda: t[0]
    env._env._t0 = t[0]
    t[0] += 61
    o, r, term, trunc, _ = env.step(act("list_records", table="ledger"))
    assert trunc and not term and o["error"]["code"] == "time_limit"


def test_c19_closed_rewards_are_zero_and_hidden(eps):
    truth = json.loads((eps["E"] / "truth.json").read_text())
    i = truth["issues"][0]
    f = {"episode_id": truth["episode_id"], "findings": [{"type": i["type"], "affected_ids": i["affected_ids"],
         "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "x"}]}
    env = ja.make_env()
    closed(env, eps["E"])
    rewards = [env.step(act("list_records", table="ledger"))[1] for _ in range(3)]
    o, r, term, trunc, info = env.step(act("submit", findings=f))
    assert rewards == [0.0, 0.0, 0.0] and r == 0.0 and term
    assert "tp" not in json.dumps(o) + json.dumps(info) and "reward" not in json.dumps(info)
    assert ja.score_episode(str(eps["E"]), f, 3)["tp"] == 1


def test_c20_no_closed_data_reachable(eps):
    secret = {json.loads((eps[k] / "truth.json").read_text())["issues"][0]["issue_id"] for k in ("E",)}
    env = ja.make_env()
    obs, info = closed(env, eps["E"])
    seen = json.dumps(obs) + json.dumps(info)
    for t in ("ledger", "invoices", "payments", "fx"):
        cur = None
        while True:
            o, _, _, _, i = env.step(act("list_records", table=t, cursor=cur, limit=50))
            seen += json.dumps(o) + json.dumps(i)
            cur = o["result"]["next_cursor"]
            if cur is None:
                break
    for a in (act("get_record", table="../truth.json", id="x"), act("get_document", doc_id="../../truth.json"),
              act("calculator", expression="__import__('os')")):
        seen += json.dumps(env.step(a)[0])
    assert not any(s in seen for s in secret) and "provenance" not in seen and str(eps["E"]) not in seen


# ----------------------------------------------------------------------------- block 3: matcher and reward 1-20
@pytest.fixture(scope="module")
def fx(eps, tmp_path_factory):
    """The published common fixture, built on a real episode: type T, record R1, 10000 KZT, evidence {D1, D2}, tau 0."""
    d = tmp_path_factory.mktemp("fixture") / "ep"
    shutil.copytree(eps["C"], d)
    ledger = [l.split(",")[0] for l in (d / "ledger.csv").read_text().splitlines()[1:6]]
    docs = [json.loads(l)["doc_id"] for l in (d / "invoices.jsonl").read_text().splitlines()[:3]]
    ep_id = json.loads((d / "manifest.json").read_text())["episode_id"]
    return {"dir": d, "R": ledger, "D": docs, "ep": ep_id}


def issue(iid, rids, ev_sets, t="duplicate_invoice", amount=10000):
    return {"issue_id": iid, "type": t, "affected_ids": rids, "expected_amount_minor": amount, "currency": "KZT",
            "evidence_sets": ev_sets, "tolerance_minor": 0}


def finding(rids, ev, t="duplicate_invoice", amount=10000, cur="KZT", expl="short reason"):
    return {"type": t, "affected_ids": rids, "claimed_amount_minor": amount, "currency": cur, "evidence_ids": ev, "explanation": expl}


def score(fx, issues, findings, steps=10):
    (fx["dir"] / "truth.json").write_text(json.dumps({"episode_id": fx["ep"], "issues": issues}))
    return ja.score_episode(str(fx["dir"]), {"episode_id": fx["ep"], "findings": findings}, steps)


def close(a, b):
    return abs(a - float(b)) <= 1e-9


def test_matcher_published_vectors_1_to_17(fx):
    R1, R2, R3, R4, R5 = fx["R"]
    D1, D2, D3 = fx["D"]
    A = issue("A", [R1], [[D1, D2]])
    B = issue("B", [R2], [[D1, D2]], t="period_cutoff")
    F = finding([R1], [D1, D2])
    cases = [
        ([A], [F], 10, (1, 0, 0), Fraction(98, 100)),
        ([A], [], 10, (0, 0, 1), -1),
        ([], [], 10, (0, 0, 0), Fraction(98, 100)),
        ([], [F], 10, (0, 1, 0), -1),
        ([A], [F, F], 10, (1, 1, 0), Fraction(47, 150)),
        ([A], [finding([R1], [D1, D2], t="fx_conversion")], 10, (0, 1, 1), -1),
        ([A], [finding([R2], [D1, D2])], 10, (0, 1, 1), -1),
        ([A], [finding([R1], [D1, D2], amount=10001)], 10, (0, 1, 1), -1),
        ([A], [finding([R1], [D1, D2], cur="USD")], 10, (0, 1, 1), -1),
        ([A], [finding([R1], [D1])], 10, (0, 1, 1), -1),
        ([A], [finding([R1], [D1, "U-not-in-fixture"])], 10, (0, 1, 1), -1),
        ([A], [finding([r], [D1, D2], t=t) for r in (R1, R2, R3, R4, R5) for t in TYPES], 10, (1, 19, 0), Fraction(-871, 1050)),
        ([A, B], [F], 10, (1, 0, 1), Fraction(47, 150)),
        ([A, B], [F, finding([R3], [D1, D2])], 10, (1, 1, 1), Fraction(-2, 100)),
        ([A], [F], 0, (1, 0, 0), 1),
        ([A], [F], 50, (1, 0, 0), Fraction(90, 100)),
        ([A], [F], 100, (1, 0, 0), Fraction(80, 100)),
    ]
    for n, (iss, fs, steps, counts, reward) in enumerate(cases, 1):
        s = score(fx, iss, fs, steps)
        assert (s["tp"], s["fp"], s["fn"]) == counts, n
        assert close(s["return"], reward), (n, s["return"], reward)
    s = score(fx, [A], [F, F])
    assert close(s["precision"], Fraction(1, 2)) and close(s["recall"], 1) and close(s["f1"], Fraction(2, 3))


def test_matcher_published_vectors_18_to_20_structure(fx):
    R1 = fx["R"][0]
    D1, D2, _ = fx["D"]
    A = issue("A", [R1], [[D1, D2]])
    bad = [[finding([R1], [D1, D2])] * 31, [finding([R1], [f"{D1}" if i == 0 else f"X{i}" for i in range(11)])],
           [finding([R1], [D1, D2], expl="x" * 1001)]]
    for fs in bad:
        s = score(fx, [A], fs)
        assert s["return"] == -1.0 and s["tp"] is None and s["f1"] is None


def test_matcher_open_examples_m01_m12(fx):
    R1, R2, R3, _, _ = fx["R"]
    D1, D2, D3 = fx["D"]
    A = issue("A", [R1, R2], [[D1, D2]])
    ok = lambda s: (s["tp"], s["fp"], s["fn"])  # noqa: E731
    assert ok(score(fx, [A], [finding([R2, R1, R1], [D1, D2])])) == (1, 0, 0)                     # M01
    assert ok(score(fx, [A], [finding([R1, R2, R3], [D1, D2])])) == (0, 1, 1)                     # M02
    assert ok(score(fx, [A], [finding([R1, R2], [D2, D1, D3])])) == (1, 0, 0)                     # M03
    assert ok(score(fx, [A], [finding([R1, R2], [D1, D2, "U-not-in-fixture"])])) == (0, 1, 1)     # M04
    assert score(fx, [A], [finding([R1, R2], [D1] * 10 + [D2])])["return"] == -1.0                # M05
    assert ok(score(fx, [A], [finding([R1, R2], [D1, D2], cur="kzt")])) == (0, 1, 1)              # M06
    assert ok(score(fx, [A], [finding([R1, R2], [D1, D2], cur="KZT ")])) == (0, 1, 1)             # M06
    assert ok(score(fx, [A], [finding([R1, R2], [D1, D2], amount=10001)])) == (0, 1, 1)           # M07
    A1 = {**A, "tolerance_minor": 1}
    assert ok(score(fx, [A1], [finding([R1, R2], [D1, D2], amount=10001)])) == (1, 0, 0)          # M08
    assert ok(score(fx, [A1], [finding([R1, R2], [D1, D2], amount=10002)])) == (0, 1, 1)          # M08
    for bad in ("10000", True):                                                                   # M09
        assert score(fx, [A], [finding([R1, R2], [D1, D2], amount=bad)])["return"] == -1.0
    Aa, Bb = issue("A", [R1, R2], [[D1]]), issue("B", [R1, R2], [[D2]])
    assert ok(score(fx, [Aa, Bb], [finding([R1, R2], [D1, D2]), finding([R1, R2], [D1])])) == (2, 0, 0)  # M10
    for fs in ([finding([R1, R2], [D1, D2])] * 2,):                                               # M11
        assert ok(score(fx, [Aa, Bb], fs)) == (2, 0, 0)
    assert ok(score(fx, [A], [finding([R1, R2], [D1, D2])] * 2)) == (1, 1, 0)                     # M12
    assert close(score(fx, [A], [finding([R1, R2], [D1, D2])] * 2)["return"], Fraction(47, 150))


def test_m11_tie_break_is_published_rule():
    from auditgym.matcher import grade
    A = issue("A", ["R1", "R2"], [["D1"]])
    B = issue("B", ["R1", "R2"], [["D2"]])
    fs = [finding(["R1", "R2"], ["D1", "D2"])] * 2
    g = grade(fs, {"issues": [B, A]}, 10, {"R1", "R2", "D1", "D2"})
    assert [(m["issue_id"], m["finding"]) for m in g["matches"] if m["result"] == "TP"] == [("A", 0), ("B", 1)]


# ----------------------------------------------------------------------------- block 4: paired transport
def test_paired_copy_with_shuffled_rows_and_renamed_ids(eps, tmp_path):
    """The jury's own copy (rows shuffled, ids renamed everywhere incl. truth) through the same episode_dir: the
    oracle's answer with the new ids gets the same TP/FP/FN/P/R/F1/return at equal inspection steps."""
    from auditgym.episode import load_public, write_episode
    from auditgym.perturb import perturb
    from types import SimpleNamespace
    src = eps["E"]
    pub, truth = load_public(src), json.loads((src / "truth.json").read_text())
    new_pub, new_truth, m = perturb(pub, truth, seed=5)
    dst = tmp_path / "copy"
    write_episode(SimpleNamespace(public=new_pub, truth=new_truth, provenance={}), dst, with_grader=False)
    (dst / "truth.json").write_text(json.dumps(new_truth))
    oracle = lambda t: {"episode_id": t["episode_id"], "findings": [finding(i["affected_ids"], i["evidence_sets"][0], t=i["type"],  # noqa: E731
                        amount=i["expected_amount_minor"], cur=i["currency"]) for i in t["issues"]]}
    a, b = ja.score_episode(str(src), oracle(truth), 7), ja.score_episode(str(dst), oracle(new_truth), 7)
    assert a == b and a["return"] == pytest.approx(1 - 0.014, abs=1e-9) and a["tp"] == len(truth["issues"])
    env = ja.make_env()
    obs, _ = closed(env, dst)
    assert obs["manifest"]["episode_id"] == new_truth["episode_id"]
