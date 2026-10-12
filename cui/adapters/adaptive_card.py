"""Adapter: canonical DSL -> Microsoft Adaptive Card 1.5 (Teams)."""
from __future__ import annotations

COLOR = {"critical": "Attention", "warning": "Warning", "positive": "Good"}


def _tb(text, **kw) -> dict:
    return {"type": "TextBlock", "text": str(text), "wrap": True, **kw}


def _colored(c: dict, **kw) -> dict:
    extra = {"color": COLOR[c["tone"]]} if c.get("tone") in COLOR else {}
    return _tb(c["display"], **extra, **kw)


def render(dsl: dict) -> dict:
    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.5",
        "body": [_node(n) for n in dsl["root"]["children"]],
    }


def _metric(n: dict) -> dict:
    return {
        "type": "Container",
        "items": [_tb(n["label"], isSubtle=True, size="Small"), _tb(n["display"], size="ExtraLarge", weight="Bolder")],
    }


def _node(n: dict) -> dict:
    t = n["type"]
    if t == "row":
        return {"type": "ColumnSet", "columns": [{"type": "Column", "width": "stretch", "items": [_metric(c)]} for c in n["children"]]}
    if t == "metric":
        return _metric(n)
    if t == "text":
        return _tb(f"**{n['label']}:** {n['display']}")
    if t == "status":
        return {"type": "ColumnSet", "columns": [
            {"type": "Column", "width": "auto", "items": [_tb(n["label"] + ":", isSubtle=True)]},
            {"type": "Column", "width": "stretch", "items": [_colored(n, weight="Bolder")]},
        ]}
    head = _tb(f"{n['label']} ({n['count']})", weight="Bolder", size="Medium")
    if n.get("emptyText"):
        return {"type": "Container", "items": [head, _tb(n["emptyText"], isSubtle=True)]}
    if t == "list":
        items = [head]
        for it in n["items"]:
            sub = [_colored(it["title"], weight="Bolder")]
            if it["subtitle"]:
                sub.append(_colored(it["subtitle"], isSubtle=True))
            if it["meta"]:
                sub.append(_tb(" · ".join(f"{m['label']}: {m['display']}" for m in it["meta"]), size="Small", isSubtle=True))
            items.append({"type": "Container", "separator": True, "spacing": "Small", "items": sub})
        return {"type": "Container", "items": items}
    return {"type": "Container", "items": [head, {
        "type": "Table",
        "firstRowAsHeaders": True,
        "columns": [{"width": 1} for _ in n["columns"]],
        "rows": [{"type": "TableRow", "cells": [{"type": "TableCell", "items": [_tb(c["label"], weight="Bolder")]} for c in n["columns"]]}]
        + [{"type": "TableRow", "cells": [{"type": "TableCell", "items": [_colored(c)]} for c in row]} for row in n["rows"]],
    }]}
