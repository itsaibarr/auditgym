"""Contract & isolation (case §04): reset/step shapes, actions, pages, cursors, errors, limits, modes."""
import json

import pytest

from auditgym.env import AuditEnv
from helpers import noop, play_actions, submit_after


def test_reset_returns_obs_info_json(dirty):
    env = AuditEnv("eval")
    obs, info = env.reset(options={"episode": dirty})
    json.dumps(obs), json.dumps(info)
    assert {"task", "manifest", "policy", "tables", "available_actions", "limits", "steps_remaining"} <= set(obs)
    assert obs["steps_remaining"] == 100 and info == {"steps_remaining": 100, "error_code": None}
    m = obs["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts", "rounding"):
        assert k in m
    blob = json.dumps(obs)
    assert "seed" not in blob.lower() and "issues" not in blob and "truth" not in blob.lower()


def test_reset_from_seed_and_config_is_supported():
    env = AuditEnv("train")
    obs, _ = env.reset(seed=7, config={"errors": {"fx_conversion": {"on": True, "count": [1, 1]}}})
    assert obs["manifest"]["episode_id"].startswith("ep_")


def test_step_contract_five_tuple(dirty):
    env, out = play_actions(dirty, [{"name": "list_records", "arguments": {"table": "ledger"}}])
    obs, r, term, trunc, info = out[0]
    assert r == 0.0 and term is False and trunc is False
    assert set(obs) == {"result", "error", "steps_remaining"} and obs["steps_remaining"] == 99


@pytest.mark.parametrize("table", ["ledger", "invoices", "payments", "fx"])
def test_pages_max_50_and_stable_cursors_cover_table(dirty, table):
    env = AuditEnv("train")
    obs, _ = env.reset(options={"episode": dirty})
    rows, cursor, seen_pages = [], None, []
    while True:
        o = env.step({"name": "list_records", "arguments": {"table": table, "cursor": cursor}})[0]
        assert o["error"] is None
        assert len(o["result"]["rows"]) <= 50
        rows += o["result"]["rows"]
        seen_pages.append((cursor, o["result"]["rows"]))
        cursor = o["result"]["next_cursor"]
        if cursor is None:
            break
    assert len(rows) == obs["tables"][table]["count"]
    key = obs["tables"][table]["primary_key"]
    assert len({r[key] for r in rows}) == len(rows)
    # same cursor -> same page (stable)
    c, page = seen_pages[-1]
    again = env.step({"name": "list_records", "arguments": {"table": table, "cursor": c}})[0]
    assert again["result"]["rows"] == page


def test_get_record_and_document(dirty):
    env = AuditEnv("train")
    env.reset(options={"episode": dirty})
    row = dirty.public["ledger"][3]
    o = env.step({"name": "get_record", "arguments": {"table": "ledger", "id": row["row_id"]}})[0]
    assert o["result"]["record"]["row_id"] == row["row_id"]
    doc = dirty.public["invoices"][0]
    o = env.step({"name": "get_document", "arguments": {"doc_id": doc["doc_id"]}})[0]
    assert o["result"]["document"]["line_items"] == doc["line_items"]
    fx = dirty.public["fx"][0]
    fid = f"FX-{fx['date']}-{fx['source_currency']}-{fx['target_currency']}"
    o = env.step({"name": "get_record", "arguments": {"table": "fx", "id": fid}})[0]
    assert o["result"]["record"]["rate"] == fx["rate"]


BAD_ACTIONS = [
    ({"name": "drop_table", "arguments": {}}, "unknown_action"),
    ({"name": "list_records", "arguments": {"table": "secrets"}}, "unknown_table"),
    ({"name": "list_records", "arguments": {"table": "ledger", "limit": 51}}, "invalid_arguments"),
    ({"name": "list_records", "arguments": {"table": "ledger", "limit": 0}}, "invalid_arguments"),
    ({"name": "list_records", "arguments": {"table": "ledger", "limit": True}}, "invalid_arguments"),
    ({"name": "list_records", "arguments": {"table": "ledger", "cursor": "p999999"}}, "invalid_cursor"),
    ({"name": "list_records", "arguments": {"table": "ledger", "cursor": "../../truth"}}, "invalid_cursor"),
    ({"name": "list_records", "arguments": {"table": "ledger", "sql": "select"}}, "invalid_arguments"),
    ({"name": "get_record", "arguments": {"table": "ledger", "id": "L-00000-1"}}, "not_found"),
    ({"name": "get_record", "arguments": {"table": "ledger", "id": "../_grader/truth.json"}}, "not_found"),
    ({"name": "get_document", "arguments": {"doc_id": "D-0"}}, "not_found"),
    ({"name": "calculator", "arguments": {"expression": "__import__('os').system('id')"}}, "calc_error"),
    ({"name": "calculator", "arguments": {"expression": "2**100000"}}, "calc_error"),
    ({"name": "calculator", "arguments": {"expression": "1/0"}}, "calc_error"),
    ("not a dict", "invalid_action"),
    ({"arguments": {}}, "invalid_action"),
    ({"name": "submit", "arguments": {}}, "invalid_arguments"),
]


@pytest.mark.parametrize("action,code", BAD_ACTIONS)
def test_bounded_errors_consume_a_step(dirty, action, code):
    env, out = play_actions(dirty, [action])
    obs, r, term, trunc, info = out[0]
    assert obs["error"]["code"] == code and r == 0.0 and not term and not trunc
    assert obs["steps_remaining"] == 99 and info["error_code"] == code
    assert len(obs["error"]["message"]) <= 200


def test_calculator_is_decimal_exact(dirty):
    env, out = play_actions(dirty, [{"name": "calculator", "arguments": {"expression": "0.1+0.2"}}])
    assert out[0][0]["result"]["value"] == "0.3"


def test_step_limit_100_truncates_with_empty_answer(dirty):
    env, out = play_actions(dirty, [noop()] * 100)
    assert len(out) == 100
    obs, r, term, trunc, info = out[-1]
    assert trunc is True and term is False
    g = env.result()
    assert g["tp"] == 0 and g["fn"] == len(dirty.truth["issues"]) and g["inspection_steps"] == 100
    assert r == g["reward"] == -1.0
    assert all(o[1] == 0.0 for o in out[:-1])


def test_submit_on_step_100_is_allowed(clean):
    res, out = submit_after(clean, [], steps=99)
    assert out[-1][2] is True and out[-1][3] is False
    assert res["reward"] == pytest.approx(1 - 0.002 * 99)


def test_truncation_on_clean_books(clean):
    env, out = play_actions(clean, [noop()] * 100)
    assert env.result()["reward"] == pytest.approx(0.8)


def test_after_end_episode_over_no_regrade(dirty):
    from helpers import oracle_findings
    res, out = submit_after(dirty, oracle_findings(dirty))
    env, _ = play_actions(dirty, [{"name": "submit", "arguments": {"findings": {"episode_id": dirty.episode_id, "findings": []}}}])
    o = env.step({"name": "submit", "arguments": {"findings": {"episode_id": dirty.episode_id, "findings": []}}})
    assert o[0]["error"]["code"] == "episode_over" and o[1] == 0.0 and o[2] is True
    assert env.steps_used == 1


def test_eval_mode_zero_reward_minimal_info(dirty):
    from helpers import oracle_findings
    res, out = submit_after(dirty, oracle_findings(dirty), steps=3, mode="eval")
    for obs, r, term, trunc, info in out:
        assert r == 0.0 and set(info) == {"steps_remaining", "error_code"}
    assert res["reward"] == pytest.approx(1 - 0.002 * 3)


def test_train_mode_terminal_reward_only(dirty):
    from helpers import oracle_findings
    res, out = submit_after(dirty, oracle_findings(dirty), steps=3, mode="train")
    assert [o[1] for o in out[:-1]] == [0.0] * 3 and out[-1][1] == res["reward"]


def test_time_limit_truncates(dirty):
    t = [0.0]
    env = AuditEnv("train", clock=lambda: t[0])
    env.reset(options={"episode": dirty})
    env.step(noop())
    t[0] = 61.0
    obs, r, term, trunc, info = env.step(noop())
    assert trunc and obs["error"]["code"] == "time_limit" and env.result()["tp"] == 0


def test_reset_isolates_state(dirty, clean):
    env = AuditEnv("train")
    env.reset(options={"episode": dirty})
    for _ in range(5):
        env.step(noop())
    a = env.reset(options={"episode": clean})
    b = AuditEnv("train").reset(options={"episode": clean})
    assert a == b and env.steps_used == 0
