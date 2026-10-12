"""Stage 3: deterministic compiler. View plan + API JSON -> canonical UI DSL.

The canonical DSL ("cui/1.0") is platform-neutral and *semantic*: node types
are metric / text / status / list / table inside stack / row containers, with
values, display strings and tones already resolved. Renderers (adapters) decide
spacing, fonts and widgets. No part of this module calls an LLM.
"""
from __future__ import annotations

import re
from datetime import datetime

from .catalog import Catalog, Field, get_field, resolve

DSL_VERSION = "cui/1.0"

TONES = {
    "critical": {"high", "critical", "urgent", "escalated", "breached", "failed", "offline"},
    "warning": {"medium", "waiting on customer", "pending", "away", "on call", "at risk"},
    "positive": {"low", "resolved", "closed", "available", "on shift", "ok", "healthy"},
}


def tone_for(value) -> str | None:
    v = str(value).casefold()
    for tone, words in TONES.items():
        if v in words:
            return tone
    return None


ACRONYMS = {"csat": "CSAT", "sla": "SLA", "id": "ID", "url": "URL", "ttl": "TTL", "api": "API", "nps": "NPS", "pct": "%"}


def humanize(key: str) -> str:
    key = key.split(".")[-1]
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key).replace("_", " ")
    words = [ACRONYMS.get(w.lower(), w) for w in key.split()]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


# --- value formatting --------------------------------------------------------

def _trim(x: float, digits: int = 2) -> str:
    s = f"{x:,.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _duration_seconds(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s" if s else f"{m}m"
    return f"{s}s"


def _duration_minutes(mins: float) -> str:
    if mins >= 60:
        return _duration_seconds(mins * 60)
    return f"{_trim(mins, 1)} min"


def _parse_dt(v: str) -> datetime | None:
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def display(value, vtype: str, unit: str | None = None, fmt: str = "auto") -> str:
    if value is None:
        return "-"
    if vtype == "boolean":
        return "Yes" if value else "No"
    if vtype == "datetime":
        dt = _parse_dt(value)
        return f"{dt:%b} {dt.day}, {dt:%H:%M}" if dt else str(value)
    if vtype == "date":
        dt = _parse_dt(value)
        return f"{dt:%b} {dt.day}, {dt.year}" if dt else str(value)
    if vtype in ("integer", "number"):
        if fmt == "auto":
            fmt = {"seconds": "duration_seconds", "minutes": "duration_minutes", "percent": "percent"}.get(unit or "", "number")
        if fmt == "duration_seconds":
            return _duration_seconds(value)
        if fmt == "duration_minutes":
            return _duration_minutes(value)
        if fmt == "percent":
            return f"{_trim(value * 100 if abs(value) <= 1 else value, 1)}%"
        if unit == "hours":
            return f"{_trim(value, 1)} h"
        return _trim(value)
    return str(value)


def _cell(label: str, value, vtype: str, unit=None, enum=False) -> dict:
    cell = {"label": label, "value": value, "display": display(value, vtype, unit)}
    tone = tone_for(value) if (enum or vtype == "string") and isinstance(value, str) else None
    if tone:
        cell["tone"] = tone
    return cell


# --- node compilation --------------------------------------------------------

def compile_plan(plan: dict, data, cat: Catalog, prompt: str = "") -> dict:
    nodes = []
    for i, n in enumerate(plan["nodes"], 1):
        node = _compile_node(n, data, cat)
        node["id"] = f"n{i}"
        nodes.append(node)
    return {
        "cui": DSL_VERSION,
        "meta": {"prompt": prompt},
        "root": {"type": "stack", "children": _group_metrics(nodes)},
    }


def _group_metrics(nodes: list[dict]) -> list[dict]:
    out: list[dict] = []
    run: list[dict] = []

    def flush():
        if len(run) >= 2:
            out.append({"type": "row", "children": list(run)})
        else:
            out.extend(run)
        run.clear()

    for n in nodes:
        if n["type"] == "metric":
            run.append(n)
        else:
            flush()
            out.append(n)
    flush()
    return out


def _compile_node(n: dict, data, cat: Catalog) -> dict:
    t = n["type"]
    if t in ("metric", "text", "status"):
        s = cat.scalars[n["value"]]
        value = resolve(data, n["value"])
        label = n.get("label") or humanize(n["value"].rsplit("/", 1)[-1])  # derived unless a plan supplies one
        node = {"type": t, "label": label, "value": value, "source": n["value"]}
        if t == "metric":
            node["display"] = display(value, s.type, s.unit, n.get("format", "auto"))
        else:
            node["display"] = display(value, s.type, s.unit)
        if t == "status":
            node["tone"] = tone_for(value) or "neutral"
        return node
    return _compile_collection(n, data, cat)


def _compile_collection(n: dict, data, cat: Catalog) -> dict:
    col = cat.collections[n["source"]]
    items = list(resolve(data, n["source"]))
    total = len(items)
    flt = n.get("filter")
    label = n.get("label") or humanize(n["source"].rsplit("/", 1)[-1])
    if not n.get("label") and flt:
        label += f" ({flt['equals']})"
    node = {"type": n["type"], "label": label, "source": n["source"]}

    if flt:
        want = str(flt["equals"]).casefold()
        items = [it for it in items if str(get_field(it, flt["field"])).casefold() == want]
        node["filter"] = {"field": flt["field"], "equals": flt["equals"]}
    if n.get("sort_by"):
        key = n["sort_by"]
        present = [it for it in items if get_field(it, key) is not None]
        missing = [it for it in items if get_field(it, key) is None]
        present.sort(key=lambda it: get_field(it, key), reverse=n.get("sort_dir", "asc") == "desc")
        items = present + missing
    if n.get("limit"):
        items = items[: n["limit"]]
    node["total"] = total
    node["count"] = len(items)

    fields = [col.fields[f] for f in n["fields"]]

    def cell(item, f: Field) -> dict:
        return _cell(humanize(f.name), get_field(item, f.name), f.type, f.unit, enum=bool(f.values))

    if n["type"] == "table":
        node["columns"] = [{"key": f.name, "label": humanize(f.name), "kind": "number" if f.type in ("integer", "number") else ("datetime" if f.type in ("datetime", "date") else "text")} for f in fields]
        node["rows"] = [[cell(it, f) for f in fields] for it in items]
    else:
        out = []
        for it in items:
            cells = [cell(it, f) for f in fields]
            out.append({"title": cells[0], "subtitle": cells[1] if len(cells) > 1 else None, "meta": cells[2:]})
        node["items"] = out
    if not items:
        node["emptyText"] = "No matching items"
    return node


def default_plan(cat: Catalog) -> dict:
    """Safe fallback when the model fails validation: the first few numeric scalars."""
    nums = [s for s in cat.scalars.values() if s.type in ("integer", "number")][:3]
    return {
        "intent": "fallback: default overview",
        "nodes": [{"type": "metric", "label": humanize(s.pointer.rsplit("/", 1)[-1]), "value": s.pointer, "format": "auto"} for s in nums],
    }
