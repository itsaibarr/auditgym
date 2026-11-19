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
