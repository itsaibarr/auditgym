#!/usr/bin/env python3
"""Language-agnostic protocol demo: any program that reads JSON lines on stdin and writes one action per
line can be a policy. Run with:  auditgym run --policy "cmd:python3 /abs/path/examples/stdio_policy.py" --split dev --trust-policies

Runner -> policy: {"type": "reset", "observation": ..., "info": ...}
                  {"type": "step", "observation": ..., "reward": ..., "terminated": ..., "truncated": ..., "info": ...}
                  {"type": "close"}
Policy -> runner: {"name": "<action>", "arguments": {...}}
"""
import json
import sys

episode_id = None
for line in sys.stdin:
    msg = json.loads(line)
    if msg["type"] == "close":
        break
    if msg["type"] == "reset":
        episode_id = msg["observation"]["manifest"]["episode_id"]
        action = {"name": "list_records", "arguments": {"table": "payments"}}
    else:
        action = {"name": "submit", "arguments": {"findings": {"episode_id": episode_id, "findings": []}}}
    print(json.dumps(action), flush=True)
