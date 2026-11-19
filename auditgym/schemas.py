"""Strict schemas (msgspec) for findings and actions + JSON Schema export (usable as LLM tool definitions)."""
from __future__ import annotations

from typing import Annotated, Literal

import msgspec
from msgspec import Meta

FindingType = Literal["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
Id = Annotated[str, Meta(min_length=1, max_length=64)]
MAX_FINDINGS, MAX_EVIDENCE, MAX_EXPLANATION = 30, 10, 1000


class Finding(msgspec.Struct, forbid_unknown_fields=True):
    type: FindingType
    affected_ids: Annotated[list[Id], Meta(min_length=1, max_length=10)]
    claimed_amount_minor: Annotated[int, Meta(ge=0, le=10**18)]
    currency: Annotated[str, Meta(pattern="^[A-Z]{3}$")]
    evidence_ids: Annotated[list[Id], Meta(max_length=MAX_EVIDENCE)]
    explanation: Annotated[str, Meta(max_length=MAX_EXPLANATION)]


class Findings(msgspec.Struct, forbid_unknown_fields=True):
    episode_id: Id
    findings: Annotated[list[Finding], Meta(max_length=MAX_FINDINGS)]


def parse_findings(obj, episode_id: str) -> tuple[list[dict] | None, str | None]:
    """Return (findings, None) or (None, error). Rejects bools/floats/strings for ints, extra/missing keys."""
    try:
        if isinstance(obj, (str, bytes)):
            parsed = msgspec.json.decode(obj, type=Findings, strict=True)
        else:
            parsed = msgspec.convert(obj, Findings, strict=True)
    except (msgspec.ValidationError, msgspec.DecodeError, TypeError, ValueError, RecursionError) as exc:
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
