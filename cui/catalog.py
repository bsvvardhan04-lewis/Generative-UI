"""Stage 0: introspect an API response into a compact data catalog.

The LLM never sees the raw payload, only this catalog: scalar pointers with an
example value, and collections (arrays of objects) with their fields. Values
are only ever copied by the compiler, never by the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_ENUM = 6  # a string field with <= this many distinct values is treated as an enum
MAX_FIELD_DEPTH = 2

NUMERIC = ("integer", "number")


@dataclass
class Scalar:
    pointer: str
    type: str  # integer | number | string | boolean | datetime | date | url | email
    sample: object
    unit: str | None = None  # seconds | minutes | hours | percent


@dataclass
class Field:
    name: str  # dotted path relative to a collection item, e.g. "customer.name"
    type: str
    sample: object = None
    unit: str | None = None
    values: list[str] | None = None  # enum values, if low-cardinality string


@dataclass
class Collection:
    pointer: str
    count: int
    fields: dict[str, Field] = field(default_factory=dict)


@dataclass
class Catalog:
    scalars: dict[str, Scalar] = field(default_factory=dict)
    collections: dict[str, Collection] = field(default_factory=dict)


# --- pointer helpers -------------------------------------------------------

def esc(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def resolve(data, pointer: str):
    cur = data
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def get_field(item, dotted: str):
    cur = item
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --- type inference --------------------------------------------------------

def infer_type(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        if ISO_DATETIME.match(v):
            return "datetime"
        if ISO_DATE.match(v):
            return "date"
        if v.startswith(("http://", "https://")):
            return "url"
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            return "email"
        return "string"
    return "string"


def infer_unit(key: str) -> str | None:
    k = key.lower()
    if k.endswith(("seconds", "secs")):
        return "seconds"
    if k.endswith(("minutes", "mins")):
        return "minutes"
    if k.endswith("hours"):
        return "hours"
    if k.endswith(("pct", "percent")):
        return "percent"
    return None


# --- building --------------------------------------------------------------

def build_catalog(data) -> Catalog:
    cat = Catalog()
    _walk(data, "", cat)
    return cat


def _walk(node, ptr: str, cat: Catalog) -> None:
    if not isinstance(node, dict):
        return
    for key, val in node.items():
        child = f"{ptr}/{esc(key)}"
        if isinstance(val, dict):
            _walk(val, child, cat)
        elif isinstance(val, list):
            if val and all(isinstance(x, dict) for x in val):
                cat.collections[child] = _collection(child, val)
        elif val is not None:
            cat.scalars[child] = Scalar(child, infer_type(val), val, infer_unit(key))


def _flatten(item: dict, prefix: str = "", depth: int = 1):
    for k, v in item.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            if depth < MAX_FIELD_DEPTH:
                yield from _flatten(v, f"{name}.", depth + 1)
        elif not isinstance(v, list):
            yield name, v


def _collection(ptr: str, items: list[dict]) -> Collection:
    col = Collection(ptr, len(items))
    seen: dict[str, list] = {}
    for item in items:
        for name, v in _flatten(item):
            seen.setdefault(name, []).append(v)
    for name, vals in seen.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ftype = infer_type(vals[0])
        f = Field(name, ftype, vals[0], infer_unit(name.split(".")[-1]))
        if ftype == "string" and len(items) >= 3:
            distinct = sorted({str(v) for v in vals})
            if len(distinct) <= MAX_ENUM and len(distinct) < len(vals):
                f.values = distinct
        col.fields[name] = f
    return col


# --- rendering for the prompt ---------------------------------------------

def _example(v) -> str:
    s = repr(v) if not isinstance(v, str) else f'"{v}"'
    return s if len(s) <= 40 else s[:37] + '..."'


def render_catalog(cat: Catalog) -> str:
    lines = ["SCALARS (pointer | type | example)"]
    for s in cat.scalars.values():
        t = s.type + (f" ({s.unit})" if s.unit else "")
        lines.append(f"{s.pointer} | {t} | {_example(s.sample)}")
    lines.append("")
    lines.append("COLLECTIONS (arrays of items; use `fields` names below)")
    for c in cat.collections.values():
        lines.append(f"{c.pointer} | {c.count} items")
        for f in c.fields.values():
            t = f.type + (f" ({f.unit})" if f.unit else "")
            extra = f", one of [{', '.join(f.values)}]" if f.values else f", e.g. {_example(f.sample)}"
            lines.append(f"  - {f.name}: {t}{extra}")
    return "\n".join(lines)
