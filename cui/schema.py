"""Stage 1a: build the per-request JSON Schema that constrains the LLM.

Every pointer / field the model may reference is an `enum` generated from the
catalog, so a hallucinated path cannot be emitted by a grammar-constrained
backend. Each collection gets its own list/table variant so `fields`, `sort_by`
and `filter` are scoped to that collection.
"""
from __future__ import annotations

from .catalog import NUMERIC, Catalog

MAX_NODES = 8
MAX_FIELDS = 6
MAX_LIMIT = 20
FORMATS = ["auto", "number", "percent", "duration_seconds", "duration_minutes"]  # optional; the compiler infers units


def _obj(type_name: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"type": {"enum": [type_name]}, **props},
        "required": ["type", *required],
        "additionalProperties": False,
    }


def build_plan_schema(cat: Catalog) -> dict:
    variants: list[dict] = []
    nums = [p for p, s in cat.scalars.items() if s.type in NUMERIC]
    strs = [p for p, s in cat.scalars.items() if s.type not in NUMERIC and s.type != "boolean"]
    everything = list(cat.scalars)

    # Labels, units and formats are derived by the compiler, so the model writes only choices.
    # Every token the model does not have to generate saves roughly 70-115 ms on a CPU.
    if nums:
        variants.append(_obj("metric", {"value": {"enum": nums}}, ["value"]))
    if everything:
        variants.append(_obj("text", {"value": {"enum": everything}}, ["value"]))
    if strs:
        variants.append(_obj("status", {"value": {"enum": strs}}, ["value"]))

    for ptr, col in cat.collections.items():
        names = list(col.fields)
        enum_fields = [n for n, f in col.fields.items() if f.values]
        props = {
            "source": {"enum": [ptr]},
            "fields": {"type": "array", "items": {"enum": names}, "minItems": 1, "maxItems": MAX_FIELDS},
            "sort_by": {"enum": names},
            "sort_dir": {"enum": ["asc", "desc"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
        }
        if enum_fields:
            all_values = sorted({v for n in enum_fields for v in col.fields[n].values})
            props["filter"] = {
                "type": "object",
                "properties": {"field": {"enum": enum_fields}, "equals": {"enum": all_values}},
                "required": ["field", "equals"],
                "additionalProperties": False,
            }
        for t in ("list", "table"):
            variants.append(_obj(t, props, ["source", "fields"]))

    return {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "minItems": 1, "maxItems": MAX_NODES, "items": {"anyOf": variants}},
        },
        "required": ["nodes"],
        "additionalProperties": False,
    }
