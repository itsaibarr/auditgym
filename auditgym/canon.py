"""Canonical JSON (byte-identical to RFC 8785 for our float-free data) and fingerprints."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def assert_no_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise AssertionError(f"float in emitted data at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_floats(v, f"{path}[{i}]")


def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def loads_strict(s):
    """json.loads that rejects duplicate object keys (last-key-wins is a grading ambiguity)."""
    return json.loads(s, object_pairs_hook=_no_dup_keys)
