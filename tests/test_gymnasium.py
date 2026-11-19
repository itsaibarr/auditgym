import json

import pytest

gym = pytest.importorskip("gymnasium")


def test_gymnasium_wrapper_contract():
    import auditgym.contrib.gymnasium_env  # noqa: F401
    env = gym.make("AuditGym-v1", mode="train", disable_env_checker=True)
    obs, info = env.reset(seed=3)
    assert json.loads(obs)["steps_remaining"] == 100
    obs, r, term, trunc, info = env.step(json.dumps({"name": "list_records", "arguments": {"table": "fx"}}))
    assert r == 0.0 and not term and not trunc and json.loads(obs)["result"]["rows"]
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    assert a == b


def test_gymnasium_env_checker_determinism():
    from gymnasium.utils.env_checker import data_equivalence
    import auditgym.contrib.gymnasium_env as g
    e1, e2 = g.AuditGymEnv(), g.AuditGymEnv()
    o1, _ = e1.reset(seed=11)
    o2, _ = e2.reset(seed=11)
    assert data_equivalence(o1, o2)
    act = json.dumps({"name": "list_records", "arguments": {"table": "ledger"}})
    assert data_equivalence(e1.step(act)[0], e2.step(act)[0])
