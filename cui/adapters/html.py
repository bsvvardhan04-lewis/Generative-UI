"""Adapter: canonical DSL -> standalone HTML preview (stand-in for a React/web renderer)."""
from __future__ import annotations

from html import escape as e

CSS = """
:root{--bg:#fff;--fg:#1c2024;--muted:#5f6b76;--line:#e1e5ea;--card:#f6f8fa;--critical:#c0392b;--warning:#b9770e;--positive:#1e8449}
@media(prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8ecef;--muted:#9aa6b2;--line:#2b3238;--card:#1c2126;--critical:#ff7b6b;--warning:#f0b04a;--positive:#5fd08a}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:16px}
.stack{display:flex;flex-direction:column;gap:16px;max-width:760px}
.row{display:flex;flex-wrap:wrap;gap:12px}.metric{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:130px}
.metric .v{font-size:28px;font-weight:650}.lbl{color:var(--muted);font-size:13px}
h3{margin:0 0 6px;font-size:15px}.count{color:var(--muted);font-weight:400}
.item{padding:8px 0;border-top:1px solid var(--line)}.item .sub{color:var(--muted)}.meta{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}td.num,th.num{text-align:right}
.critical{color:var(--critical)}.warning{color:var(--warning)}.positive{color:var(--positive)}
"""


def _c(c: dict) -> str:
    cls = f' class="{c["tone"]}"' if c.get("tone") in ("critical", "warning", "positive") else ""
    return f"<span{cls}>{e(c['display'])}</span>"


def _metric(n: dict) -> str:
    return f'<div class="metric"><div class="lbl">{e(n["label"])}</div><div class="v">{e(n["display"])}</div></div>'


def _node(n: dict) -> str:
    t = n["type"]
    if t == "row":
        return '<div class="row">' + "".join(_metric(c) for c in n["children"]) + "</div>"
    if t == "metric":
        return _metric(n)
    if t in ("text", "status"):
        return f'<div><span class="lbl">{e(n["label"])}:</span> {_c(n)}</div>'
    shown = str(n["count"]) if n["count"] == n["total"] else f"{n['count']} of {n['total']}"
    head = f'<h3>{e(n["label"])} <span class="count">({shown})</span></h3>'
    if n.get("emptyText"):
        return f'<section>{head}<div class="lbl">{e(n["emptyText"])}</div></section>'
    if t == "list":
        body = ""
        for it in n["items"]:
            sub = f' <span class="sub">- {_c(it["subtitle"])}</span>' if it["subtitle"] else ""
            meta = "".join(f'<span>{e(m["label"])}: {_c(m)}</span> &nbsp; ' for m in it["meta"])
            body += f'<div class="item"><b>{_c(it["title"])}</b>{sub}<div class="meta">{meta}</div></div>'
        return f"<section>{head}{body}</section>"
    th = "".join(f'<th class="{"num" if c["kind"] == "number" else ""}">{e(c["label"])}</th>' for c in n["columns"])
    trs = "".join(
        "<tr>" + "".join(f'<td class="{"num" if col["kind"] == "number" else ""}">{_c(cell)}</td>' for col, cell in zip(n["columns"], row)) + "</tr>"
        for row in n["rows"]
    )
    return f"<section>{head}<table><tr>{th}</tr>{trs}</table></section>"


def render(dsl: dict) -> str:
    body = "".join(_node(n) for n in dsl["root"]["children"])
    title = e(dsl["meta"].get("prompt") or "Agent Console")
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style><body><div class="stack">{body}</div></body></html>'
