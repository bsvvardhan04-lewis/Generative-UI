"""Adapter: canonical DSL -> HXL widget JSON (the Salesforce Headless Experience Layer playground format).

Format, as documented in the HXL playground's component docs and tutorials:

* root: {"definition": "tile/widget", "dataProviders": [...], "children": [...]}
* data: an "inline-data" provider exposes the (minimal) API data as $data
* binding: attribute strings may contain expressions such as "{!$data.queue.open}"
* repeats: a tile/container with meta.forEach / meta.forItem renders once per item;
  meta.if keeps only items where the expression is true (used for filters)
* tables: tile/table with caption, columns [{key, header, align, columnType}] and rows [{key: value}]

Values are bound to $data whenever HXL can express the view: single values, and lists whose only
operation is a filter. Sorted or limited lists and all tables are written with literal values,
because HXL expressions have no sort or slice. Labels, captions and counts are literal text.
"""
from __future__ import annotations

import re

DATA_KEY = "$data"
ITEM_KEY = "$item"
TONE_TO_BADGE = {"critical": "error", "warning": "warning", "positive": "success"}
_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --- helpers -----------------------------------------------------------------------------

def _segments(pointer: str) -> list[str]:
    return [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]


def _expr(base: str, segments: list[str]) -> str | None:
    """'{!$data.queue.open}', or None when a key is not a plain identifier (spaces, dashes, digits first)."""
    if not segments or not all(_SAFE.match(s) for s in segments):
        return None
    return "{!" + base + "." + ".".join(segments) + "}"


def _quote(value) -> str | None:
    """An HXL string literal for a filter value, or None if it cannot be written safely."""
    s = str(value)
    return None if "'" in s or "}" in s else f"'{s}'"


def _text(text, variant: str | None = None, weight: str | None = None, literal: bool = True) -> dict:
    node = {"definition": "tile/text", "attributes": {"text": str(text)}}
    if variant:
        node["attributes"]["variant"] = variant
    if weight:
        node["attributes"]["weight"] = weight
    if literal and "{!" in str(text):
        node["ignoreExpressions"] = True  # data that happens to contain "{!" must not be evaluated
    return node


def _badge(label, tone: str | None, literal: bool = True) -> dict:
    node = {"definition": "tile/badge", "attributes": {"label": str(label), "variant": TONE_TO_BADGE.get(tone or "", "secondary")}}
    if literal and "{!" in str(label):
        node["ignoreExpressions"] = True
    return node


def _row(children: list, **attrs) -> dict:
    return {"definition": "tile/row", "attributes": attrs, "children": children}


def _column(children: list, **attrs) -> dict:
    return {"definition": "tile/column", "attributes": attrs, "children": children}


def _bindable(value, display) -> bool:
    """Bind when HXL would show the same thing: numbers, or strings shown as-is (not reformatted dates)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) or (isinstance(value, str) and display == value)


def _value(node: dict, bound: bool) -> tuple[str, bool]:
    """(text, is_literal) for a single-value node."""
    if bound and _bindable(node["value"], node["display"]):
        e = _expr(DATA_KEY, _segments(node["source"]))
        if e:
            return e, False
    return node["display"], True


# --- nodes -------------------------------------------------------------------------------

def _metric(n: dict, bound: bool) -> dict:
    text, literal = _value(n, bound)
    return {"definition": "tile/container", "children": [_column([_text(n["label"], "caption"), _text(text, "h2", literal=literal)], gap="xs")]}


def _status(n: dict, bound: bool) -> dict:
    text, literal = _value(n, bound)
    return _row([_text(n["label"], weight="semibold"), _badge(text, n.get("tone"), literal)], gap="sm", align="center")


def _plain(n: dict, bound: bool) -> dict:
    text, literal = _value(n, bound)
    return _row([_text(n["label"], weight="semibold"), _text(text, literal=literal)], gap="sm", align="center")


def _header(n: dict, with_count: bool) -> dict:
    kids = [_text(n["label"], "h3")]
    if with_count:
        shown = f"{n['count']} of {n['total']}" if n["count"] != n["total"] else str(n["count"])
        kids.append(_text(shown, "caption"))
    return _row(kids, justify="between", align="center")


def _bound_list(n: dict) -> dict | None:
    """A list HXL can compute itself: forEach over the source, meta.if for the filter. None if not expressible."""
    if n.get("sort") or n.get("limit"):
        return None
    source = _expr(DATA_KEY, _segments(n["source"]))
    fields = [_expr(ITEM_KEY, f.split(".")) for f in n["fields"]]
    if not source or not all(fields):
        return None
    meta = {"forEach": source, "forItem": ITEM_KEY}
    flt = n.get("filter")
    if flt:
        fexpr, quoted = _expr(ITEM_KEY, flt["field"].split(".")), _quote(flt["equals"])
        if not fexpr or not quoted:
            return None
        meta["if"] = "{!" + fexpr[2:-1] + " = " + quoted + "}"
    title = [_text(fields[0], weight="semibold", literal=False)]
    if len(fields) > 1:
        title.append(_text(fields[1], "caption", literal=False))
    kids = [_column(title, gap="xs")]
    if len(fields) > 2:
        kids.append(_row([_text(f, "caption", literal=False) for f in fields[2:]], gap="sm", align="center"))
    item = {"definition": "tile/container", "meta": meta, "children": [_row(kids, gap="md", justify="between", align="center")]}
    return _column([_header(n, with_count=False), item], gap="sm")


def _static_list(n: dict) -> dict:
    cards = []
    for it in n["items"]:
        title = [_text(it["title"]["display"], weight="semibold")]
        if it.get("subtitle"):
            title.append(_text(it["subtitle"]["display"], "caption"))
        kids = [_column(title, gap="xs")]
        meta = [_badge(m["display"], m["tone"]) if m.get("tone") else _text(m["display"], "caption") for m in it.get("meta", [])]
        if meta:
            kids.append(_row(meta, gap="sm", align="center"))
        cards.append({"definition": "tile/container", "children": [_row(kids, gap="md", justify="between", align="center")]})
    if not cards:
        cards.append(_text(n.get("emptyText", "No matching items"), "caption"))
    return _column([_header(n, with_count=True), *cards], gap="sm")


def _table(n: dict) -> dict:
    keys = {c["key"]: re.sub(r"[^A-Za-z0-9_]", "_", c["key"]) for c in n["columns"]}
    columns = []
    for c in n["columns"]:
        col = {"key": keys[c["key"]], "header": c["label"]}
        if c["kind"] == "number":
            col["align"] = "right"
            col["columnType"] = {"type": "number"}
        elif c["kind"] == "datetime":
            col["columnType"] = {"type": "date"}
        columns.append(col)
    rows = [{keys[c["key"]]: cell["value"] for c, cell in zip(n["columns"], row)} for row in n["rows"]]
    attrs = {"caption": n["label"], "columns": columns, "rows": rows}
    sort = n.get("sort")
    if sort and sort["field"] in keys:
        attrs["sort"] = {"key": keys[sort["field"]], "direction": sort["dir"]}
    return {"definition": "tile/table", "attributes": attrs}


def _node(n: dict, bound: bool) -> dict:
    t = n["type"]
    if t == "row":
        return _row([_node(c, bound) for c in n["children"]], gap="md", isWrapped=True)
    if t == "metric":
        return _metric(n, bound)
    if t == "status":
        return _status(n, bound)
    if t == "text":
        return _plain(n, bound)
    if t == "table":
        return _table(n)
    return (bound and _bound_list(n)) or _static_list(n)


# --- minimal data for the provider ---------------------------------------------------------

def _get(obj, segments):
    for s in segments:
        if isinstance(obj, list):
            obj = obj[int(s)]
        elif isinstance(obj, dict) and s in obj:
            obj = obj[s]
        else:
            return None
    return obj


def _put(obj: dict, segments: list[str], value) -> None:
    for s in segments[:-1]:
        obj = obj.setdefault(s, {})
    obj[segments[-1]] = value


def _walk(nodes):
    for n in nodes:
        if n["type"] == "row":
            yield from _walk(n["children"])
        else:
            yield n


def minimal_data(dsl: dict, data) -> dict:
    """Only what the view uses: referenced values, and referenced lists reduced to the fields shown
    (plus the filter field). This is the "data required for the request" part of the output."""
    out: dict = {}
    for n in _walk(dsl["root"]["children"]):
        segs = _segments(n["source"])
        if n["type"] in ("metric", "status", "text"):
            _put(out, segs, _get(data, segs))
            continue
        fields = list(n.get("fields", []))
        if n.get("filter"):
            fields.append(n["filter"]["field"])
        if n.get("sort"):
            fields.append(n["sort"]["field"])
        items = _get(data, segs) or []
        slim = []
        for it in items:
            row: dict = {}
            for f in dict.fromkeys(fields):
                v = _get(it, f.split("."))
                if v is not None:
                    _put(row, f.split("."), v)
            slim.append(row)
        _put(out, segs, slim)
    return out


def render(dsl: dict, data=None) -> dict:
    """HXL widget for the DSL. With `data` (the API response), values are bound to an inline-data
    provider ($data); without it, every value is written literally."""
    bound = data is not None
    widget: dict = {"definition": "tile/widget"}
    if bound:
        widget["dataProviders"] = [{"definition": "inline-data", "expressionKey": DATA_KEY, "inline": True, "value": minimal_data(dsl, data)}]
    body = [_node(n, bound) for n in dsl["root"]["children"]]
    widget["children"] = [{"definition": "tile/container", "children": [_column(body, gap="lg")]}]
    return widget
