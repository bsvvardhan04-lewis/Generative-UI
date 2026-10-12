"""Adapter: canonical DSL -> Markdown (chat surfaces: ChatGPT, Claude, Slack-like text)."""
from __future__ import annotations

ICON = {"critical": "🔴", "warning": "🟠", "positive": "🟢", "neutral": "⚪"}


def _cell(c: dict) -> str:
    icon = ICON.get(c.get("tone"), "")
    return f"{icon} {c['display']}".strip()


def render(dsl: dict) -> str:
    return "\n\n".join(b for b in (_node(n) for n in dsl["root"]["children"]) if b)


def _node(n: dict) -> str:
    t = n["type"]
    if t == "row":
        return "  ·  ".join(f"**{c['label']}:** {c['display']}" for c in n["children"])
    if t == "metric":
        return f"**{n['label']}:** {n['display']}"
    if t == "text":
        return f"**{n['label']}:** {n['display']}"
    if t == "status":
        return f"**{n['label']}:** {_cell(n)}"
    head = f"**{n['label']}** ({n['count']} of {n['total']})" if n["count"] != n["total"] else f"**{n['label']}** ({n['count']})"
    if n.get("emptyText"):
        return f"{head}\n_{n['emptyText']}_"
    if t == "list":
        rows = []
        for it in n["items"]:
            line = f"- **{_cell(it['title'])}**"
            if it["subtitle"]:
                line += f" - {_cell(it['subtitle'])}"
            if it["meta"]:
                line += "  _(" + " · ".join(f"{m['label']}: {_cell(m)}" for m in it["meta"]) + ")_"
            rows.append(line)
        return head + "\n" + "\n".join(rows)
    if t == "table":
        cols = n["columns"]
        out = ["| " + " | ".join(c["label"] for c in cols) + " |", "|" + "|".join("---:" if c["kind"] == "number" else "---" for c in cols) + "|"]
        out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in n["rows"]]
        return head + "\n" + "\n".join(out)
    return ""
