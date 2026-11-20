"""Regression test for the R2 blocker: `run`'s pre-flight (probe_policy) imports a custom policy before the
runner is hardened, so module-level policy code can read /proc/<runner>/environ (AUDITGYM_SALT) and memory.
The runner must be a fresh process with the salt in its *initial* environment (/proc/<pid>/environ shows only
that), as under `docker run -e AUDITGYM_SALT=...`."""
import json
import os
import subprocess
import sys

import pytest

PEEK = '''
import os
try:
    env = open(f"/proc/{os.getppid()}/environ", "rb").read()
except OSError:
    env = b""
if b"AUDITGYM_SALT=" in env:
    raise SystemExit("LEAK: policy code read the runner's environment")
class P:
    def reset(self, obs, info):
        return {"name": "submit", "arguments": {"findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []}}}
    def step(self, obs, reward, terminated, truncated, info):
        return self.reset(obs, info)
'''


@pytest.mark.skipif(not sys.platform.startswith("linux") or os.geteuid() == 0, reason="Linux /proc, non-root runner")
def test_preflight_cannot_read_runner_environ(tmp_path):
    (tmp_path / "peek.py").write_text(PEEK)
    split = json.dumps({"split": "t", "config_version": "1.0.0", "base_config": {}, "episodes": [{"seed": "leak-test-1"}]})
    r = subprocess.run([sys.executable, "-m", "auditgym", "run", "--policy", "peek:P", "--policy-path", str(tmp_path),
                        "--split", "-", "--out", str(tmp_path / "out"), "--trust-policies"],
                       input=split, capture_output=True, text=True, timeout=180,
                       env={**os.environ, "AUDITGYM_SALT": "s" * 64})
    assert "LEAK" not in r.stderr and r.returncode == 0, r.stderr[-1500:]
