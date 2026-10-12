"""Select mode: the model only *picks* catalog items; code fills in everything else.

On a CPU, generation time is roughly proportional to the number of tokens the model writes. A full
view plan costs 32-65 tokens even after removing labels; a pick such as {"pick":["/team"]} costs about 12.
Filters, sort order, limits and columns are almost always stated literally in the prompt ("high priority",
"3 oldest", "with customer name"), so the router's word matching adds them after the model has chosen
*which* data to show, which is the part that needs language understanding.

s1 used item numbers; small models mapped words to the wrong number (llama3.2:3b picked the same list for
most questions), so s2 picks by path, and drops extra picks that no word in the prompt points to.
"""
from __future__ import annotations

from .catalog import Catalog, _example
from .router import evidence, fill_collection, scalar_node

SELECT_VERSION = "s2"  # bump when SYSTEM_SELECT, the catalog format or the pick rules change (part of the cache key)
MAX_PICKS = 3

SYSTEM_SELECT = """You choose which parts of an API response answer a user's request.
You see a DATA CATALOG of VALUE and LIST paths. Reply with ONE JSON object {"pick":[paths]} using paths from the catalog.
- Usually pick 1 path; never more than 3. Pick only what the request asks for.
- Pick a VALUE for a single number or fact (a count, an average, a rate, a status).
- Pick a LIST when the user wants several items or rows, a table, or items that match a condition.
  Filters, sorting, limits and columns are added automatically from the request; just pick the list.
Example: {"pick":["/stats/open"]}"""


def select_options(cat: Catalog) -> list[tuple[str, str]]:
    """Choices: every scalar value, then every collection, as (kind, path)."""
    return [("value", p) for p in cat.scalars] + [("list", p) for p in cat.collections]


def render_options(cat: Catalog, opts: list[tuple[str, str]]) -> str:
    lines = []
    for kind, ptr in opts:
        if kind == "value":
            s = cat.scalars[ptr]
            t = s.type + (f" ({s.unit})" if s.unit else "")
            lines.append(f"VALUE {ptr} | {t} | e.g. {_example(s.sample)}")
        else:
            c = cat.collections[ptr]
            fields = ", ".join(f"{n} ({'/'.join(f.values)})" if f.values else n for n, f in c.fields.items())
            lines.append(f"LIST {ptr} | {c.count} items | fields: {fields}")
    return "\n".join(lines)


def select_schema(opts: list[tuple[str, str]]) -> dict:
    return {
        "type": "object",
        "properties": {"pick": {"type": "array", "items": {"enum": [p for _, p in opts]}, "minItems": 1, "maxItems": MAX_PICKS}},
        "required": ["pick"],
        "additionalProperties": False,
    }


def select_messages(prompt: str, cat: Catalog, opts: list[tuple[str, str]]) -> list[dict]:
    # Instructions and catalog first, request last, so the prompt cache covers everything but the request.
    user = f"DATA CATALOG\n{render_options(cat, opts)}\n\nUSER REQUEST: {prompt}"
    return [{"role": "system", "content": SYSTEM_SELECT}, {"role": "user", "content": user}]


def picks_to_plan(out, prompt: str, cat: Catalog, opts: list[tuple[str, str]], aliases=None) -> tuple[dict | None, list[str]]:
    kinds = {p: k for k, p in opts}
    picks = out.get("pick") if isinstance(out, dict) else None
    if not isinstance(picks, list) or not picks:
        return None, [f'reply must be {{"pick":[paths]}} with 1 to {MAX_PICKS} paths from the catalog']
    bad = [p for p in picks if not isinstance(p, str) or p not in kinds]
    if bad:
        return None, [f"pick {bad[0]!r} is not a path in the catalog"]
    picks = list(dict.fromkeys(picks))[:MAX_PICKS]  # keep order, drop repeats
    if len(picks) > 1:
        # An extra pick that no word of the request points to is usually noise (seen with llama3.1:8b adding
        # /agent/email to unrelated questions). Keep only supported picks, unless none has support at all
        # (e.g. "summarize my day"), where the model's judgement is all there is.
        scores = [evidence(prompt, cat, kinds[p], p, aliases) for p in picks]
        if max(scores) > 0:
            picks = [p for p, s in zip(picks, scores) if s > 0]
    nodes = [scalar_node(cat, p) if kinds[p] == "value" else fill_collection(prompt, cat, p, aliases) for p in picks]
    return {"nodes": nodes}, []
