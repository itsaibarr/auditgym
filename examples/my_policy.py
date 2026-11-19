"""Minimal Python policy for AuditGym. Run it isolated in a child process:

    auditgym run --policy my_policy:LookAround --policy-path examples --split dev --trust-policies

Interface: reset(observation, info) -> action; step(observation, reward, terminated, truncated, info) -> action.
Actions are {"name": ..., "arguments": {...}}.
"""


class LookAround:
    def __init__(self, pages: int = 3):
        self.pages = pages

    def reset(self, observation, info):
        self.episode_id = observation["manifest"]["episode_id"]
        self.n = 0
        return {"name": "list_records", "arguments": {"table": "invoices", "cursor": None, "limit": 50}}

    def step(self, observation, reward, terminated, truncated, info):
        self.n += 1
        cursor = (observation.get("result") or {}).get("next_cursor")
        if cursor and self.n < self.pages:
            return {"name": "list_records", "arguments": {"table": "invoices", "cursor": cursor, "limit": 50}}
        # this toy policy never accuses anyone: perfect on clean books, -1 on books with errors
        return {"name": "submit", "arguments": {"findings": {"episode_id": self.episode_id, "findings": []}}}
