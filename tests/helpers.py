from auditgym.env import AuditEnv
from auditgym.policies import finding


def play_actions(ep, actions, mode="train"):
    env = AuditEnv(mode)
    obs, info = env.reset(options={"episode": ep})
    out = []
    for a in actions:
        out.append(env.step(a))
        if out[-1][2] or out[-1][3]:
            break
    return env, out


def noop():
    return {"name": "calculator", "arguments": {"expression": "1+1"}}


def submit_after(ep, findings, steps=0, mode="train"):
    env, out = play_actions(ep, [noop()] * steps + [{"name": "submit", "arguments": {"findings": {"episode_id": ep.episode_id, "findings": findings}}}], mode)
    return env.result(), out


def oracle_findings(ep):
    return [finding(i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"], i["evidence_sets"][0], "x") for i in ep.truth["issues"]]
