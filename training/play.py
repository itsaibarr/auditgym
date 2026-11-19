"""Run one AuditGym episode with the scan-then-decide scaffold and grade it with the env's own grader."""
from __future__ import annotations

from auditgym.env import AuditEnv
from auditgym.policies import ExpertPolicy

from agpolicy import scan


def play(seed: int, cfg: dict, decide):
    """decide(obs0, rows) -> submit payload. Eval mode (reward withheld until the end), no wall-clock limit because
    the LLM answer is produced offline on a GPU and replayed here. Returns (obs0, rows, scan_steps, grade)."""
    env = AuditEnv("eval", time_limit_s=None)
    obs0, _ = env.reset(seed=seed, config=cfg)
    rows, n = scan(env, obs0)
    payload = decide(obs0, rows)
    _, _, term, _, _ = env.step({"name": "submit", "arguments": {"findings": payload}})
    assert term
    return obs0, rows, n, env.result()


def expert_decide(obs0, rows):
    """Truth-free ExpertPolicy on exactly the rows the scaffold observed (its own scan is the same actions)."""
    pub = {"manifest": obs0["manifest"], "policy": obs0["policy"], **rows}
    return {"episode_id": obs0["manifest"]["episode_id"], "findings": ExpertPolicy().decide(pub)}
