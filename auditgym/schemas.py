"""Strict schemas (msgspec) for findings and actions + JSON Schema export (usable as LLM tool definitions)."""
from __future__ import annotations

from typing import Annotated

import msgspec
from msgspec import Meta

from .canon import loads_strict

FINDING_TYPES = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
CURRENCIES = ["KZT", "USD", "EUR", "CNY", "GBP", "JPY", "KWD"]  # money.EXPONENTS keys
# Parsing accepts any bounded string for type and currency: per the published jury conventions an unknown type or a
# currency that differs (even only in case or spaces) is a structurally valid finding that does not match (FP), not a
# malformed submit. The exported JSON Schema still lists the supported values (enum) for LLM tool use.
FindingType = Annotated[str, Meta(min_length=1, max_length=64)]
Id = Annotated[str, Meta(min_length=1, max_length=64)]
Currency = Annotated[str, Meta(min_length=1, max_length=16)]
MAX_FINDINGS, MAX_EVIDENCE, MAX_EXPLANATION = 30, 10, 1000


class Finding(msgspec.Struct, forbid_unknown_fields=True):
    type: FindingType
    affected_ids: Annotated[list[Id], Meta(min_length=1, max_length=10)]
    claimed_amount_minor: Annotated[int, Meta(ge=-10**18, le=10**18)]  # a JSON integer (never bool/string/float)
    currency: Currency
    evidence_ids: Annotated[list[Id], Meta(max_length=MAX_EVIDENCE)]
    explanation: Annotated[str, Meta(max_length=MAX_EXPLANATION)]


class Findings(msgspec.Struct, forbid_unknown_fields=True):
    episode_id: Id
    findings: Annotated[list[Finding], Meta(max_length=MAX_FINDINGS)]


def parse_findings(obj, episode_id: str) -> tuple[list[dict] | None, str | None]:
    """Return (findings, None) or (None, error). Rejects bools/floats/strings for ints, extra/missing keys."""
    try:
        if isinstance(obj, (str, bytes)) and len(obj) > 2_000_000:
            return None, "schema: payload too large"
        if isinstance(obj, (str, bytes)):
            obj = loads_strict(obj)
        parsed = msgspec.convert(obj, Findings, strict=True)
    except (msgspec.ValidationError, msgspec.DecodeError, TypeError, ValueError, RecursionError, UnicodeDecodeError) as exc:
        return None, f"schema: {str(exc)[:200]}"
    if parsed.episode_id != episode_id:
        return None, "schema: episode_id mismatch"
    return [msgspec.to_builtins(f) for f in parsed.findings], None


TABLES = ("ledger", "invoices", "payments", "fx")

ACTION_SCHEMAS = {
    "list_records": {"type": "object", "additionalProperties": False, "required": ["table"],
                     "properties": {"table": {"enum": list(TABLES)}, "cursor": {"type": ["string", "null"]},
                                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 50}}},
    "get_record": {"type": "object", "additionalProperties": False, "required": ["table", "id"],
                   "properties": {"table": {"enum": list(TABLES)}, "id": {"type": "string", "maxLength": 64}}},
    "get_document": {"type": "object", "additionalProperties": False, "required": ["doc_id"],
                     "properties": {"doc_id": {"type": "string", "maxLength": 64}}},
    "calculator": {"type": "object", "additionalProperties": False, "required": ["expression"],
                   "properties": {"expression": {"type": "string", "maxLength": 256}}},
    "submit": {"type": "object", "additionalProperties": False, "required": ["findings"],
               "properties": {"findings": {"$ref": "#/$defs/Findings"}}},
}

ACTION_DESCRIPTIONS = {
    "list_records": "List up to 50 rows of a table in its fixed order. Pass next_cursor to continue.",
    "get_record": "Fetch one row by primary key (ledger.row_id, invoices/payments.doc_id, fx id FX-<date>-<SRC>-<TGT>).",
    "get_document": "Fetch a full invoice, credit note or payment document by doc_id.",
    "calculator": "Evaluate + - * / ( ) over decimal numbers. No variables or functions.",
    "submit": "Submit findings.json and end the episode.",
}


def json_schemas() -> dict:
    findings_schema = msgspec.json.schema(Findings)
    props = findings_schema["$defs"]["Finding"]["properties"]
    props["type"]["enum"], props["currency"]["enum"] = FINDING_TYPES, CURRENCIES  # the supported values, for tool use
    tools = []
    for name, params in ACTION_SCHEMAS.items():
        p = dict(params)
        if name == "submit":
            p = {"type": "object", "additionalProperties": False, "required": ["findings"],
                 "properties": {"findings": findings_schema}}
        tools.append({"type": "function", "function": {"name": name, "description": ACTION_DESCRIPTIONS[name], "parameters": p}})
    action = {"type": "object", "additionalProperties": False, "required": ["name", "arguments"],
              "properties": {"name": {"enum": list(ACTION_SCHEMAS)}, "arguments": {"type": "object"}}}
    return {"findings.schema.json": findings_schema, "action.schema.json": action, "tools.json": tools}
