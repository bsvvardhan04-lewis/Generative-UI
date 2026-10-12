"""Fast path: a deterministic prompt -> view-plan router. No model, answers in milliseconds.

It is deliberately conservative. It answers only when *every* content word of the prompt is
explained by the data's own vocabulary (scalar names, collection names, field names, enum values)
or by a small set of generic words (show, top, oldest, table, ...). Anything else, and anything
ambiguous, returns None so the caller falls back to the language model. A wrong confident answer
is worse than a slow correct one, so the rules lean toward declining.

The router uses no knowledge of any particular API: its vocabulary comes from the catalog.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import NUMERIC, Catalog
from .validate import validate_plan

# Words that carry no request content. Kept generic; nothing here is specific to one dataset.
STOP = set("""a an the of in on at for to is are was were be been being do does did have has had i me my we our us you your
it its this that these those there here what whats which who whom whose how when where why and or with from by any all some
show display list give tell get see view find fetch please can could would should will now currently current as about than
then them they their each every per just only also into over under out up down""".split())
# Generic nouns / measure words that name nothing by themselves.
FILLER = set("""time score value level amount long information info details data item record entry row result update one ones
thing things""".split())
COUNT_WORDS = {"many", "much", "number", "count"}
ABBREV = {"avg": "average", "pct": "percent", "num": "number", "qty": "quantity", "max": "maximum", "min": "minimum"}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
OLDEST, NEWEST = {"oldest", "earliest"}, {"newest", "latest", "recent", "newer"}
HIGH, LOW = {"highest", "most", "largest", "biggest", "best", "top"}, {"lowest", "least", "smallest", "worst"}
MAX_LIMIT = 20
DEFAULT_FIELDS = 4


def stem(w: str) -> str:
    w = w.lower()
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", "")):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)] + rep
            break
    else:
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
    if len(w) > 5 and w.endswith("ly"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


FILLER_STEMS = {stem(w) for w in FILLER}


def name_words(name: str) -> list[str]:
    """camelCase / dotted / snake_case name -> lower-case words, with abbreviations expanded."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [ABBREV.get(w.lower(), w.lower()) for w in re.findall(r"[A-Za-z]+|\d+", spaced)]


def name_stems(name: str) -> list[str]:
    return [stem(w) for w in name_words(name)]


def _num(raw: str):
    if raw.isdigit():
        return int(raw)
    return NUMBER_WORDS.get(raw)


@dataclass
class Route:
    plan: dict | None
    reason: str = ""  # why it declined (for logs / the UI), or what it matched


def apply_aliases(prompt: str, aliases: dict[str, str] | None) -> str:
    """Alias search: rewrite everyday phrases to the data's own words before matching
    ("tickets" -> "cases", "customer satisfaction" -> "csat"). Longest phrases win."""
    text = prompt.lower()
    table = {k.lower(): v.lower() for k, v in (aliases or {}).items() if k.strip()}
    if not table:
        return text
    # One pass, longest phrase first: replaced text is never re-matched by a shorter alias.
    pattern = "|".join(re.escape(p) for p in sorted(table, key=len, reverse=True))
    return re.sub(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", lambda m: f" {table[m.group(0)]} ", text)




@dataclass
class Parsed:
    """The prompt as words, plus the generic modifiers found in it (limit / sort / table)."""
    raws: list[str]
    stems: list[str]
    used: list[bool]          # True once a word is explained (stop word, filler, or matched to the data)
    count_intent: bool
    conjunction: bool
    mod_idx: set[int]         # positions of limit / sort / table words
    limit: int | None
    sort_kind: str | None     # oldest | newest | high | low
    want_table: bool


def parse(prompt: str, aliases: dict[str, str] | None = None) -> Parsed | None:
    raws = [ABBREV.get(w, w) for w in re.findall(r"[a-z0-9]+", apply_aliases(prompt, aliases))]
    raws = [w for w in raws if len(w) > 1 or w.isdigit()]
    if not raws:
        return None
    n = len(raws)
    used = [r in STOP or stem(r) in FILLER_STEMS or r in COUNT_WORDS for r in raws]  # stems, so plurals ("updates") count too
    mod_idx: set[int] = set()
    limit = None
    for i, r in enumerate(raws):
        if r in OLDEST | NEWEST | {"top", "first", "last"} and i + 1 < n and _num(raws[i + 1]) is not None:
            limit = _num(raws[i + 1]); mod_idx |= {i, i + 1}
        elif _num(r) is not None and i + 1 < n and raws[i + 1] in OLDEST | NEWEST | {"top"}:
            limit = _num(r); mod_idx |= {i, i + 1}
    sort_kind = None
    for i, r in enumerate(raws):
        if r in OLDEST:
            sort_kind = "oldest"; mod_idx.add(i)
        elif r in NEWEST:
            sort_kind = "newest"; mod_idx.add(i)
        elif r in HIGH - {"top"}:
            sort_kind = "high"; mod_idx.add(i)
        elif r in LOW:
            sort_kind = "low"; mod_idx.add(i)
    # "last 5 updates" means the most recent five: a limit alone would return the first five instead
    if sort_kind is None and any(r == "last" and i + 1 < n and _num(raws[i + 1]) is not None for i, r in enumerate(raws)):
        sort_kind = "newest"
    want_table = "table" in raws
    if want_table:
        mod_idx.add(raws.index("table"))
    return Parsed(raws, [stem(w) for w in raws], used, any(r in COUNT_WORDS for r in raws),
                  any(r in ("and", "also") for r in raws), mod_idx, limit, sort_kind, want_table)


def value_hits(cat: Catalog, stems: list[str]) -> list[tuple]:
    """Every enum value of every collection field that appears in the prompt: (collection, field, value, start, end)."""
    out = []
    for ptr, col in cat.collections.items():
        for fname, f in col.fields.items():
            for v in f.values or []:
                vs = [stem(w) for w in re.findall(r"[a-z0-9]+", v.lower())]
                for s in range(0, len(stems) - len(vs) + 1):
                    if stems[s:s + len(vs)] == vs:
                        out.append((ptr, fname, v, s, s + len(vs)))
    return out


def collection_node(cat: Catalog, ptr: str, p: Parsed, used: list[bool], hits: list[tuple], strict: bool) -> tuple[dict | None, str]:
    """Build the list/table node for one collection from the prompt's words: filter, sort, limit, columns.
    strict=True (router): return None on anything doubtful. strict=False (after the model chose the
    collection): skip the doubtful part instead and keep the rest."""
    col = cat.collections[ptr]
    names = list(col.fields)
    stems, n = p.stems, len(p.stems)
    node: dict = {"type": "table" if p.want_table else "list", "source": ptr}

    if len({(h[1], h[2]) for h in hits}) > 1:
        if strict:
            return None, "several filter values"
        hits = sorted(hits, key=lambda h: (h[3] - h[4], h[3]))[:1]  # lenient: the longest (most specific) value
    if hits:
        _, ffield, fvalue, s, e = hits[0]
        node["filter"] = {"field": ffield, "equals": fvalue}
        for i in range(s, e):
            used[i] = True
        qualifier = set(name_stems(ffield))
        for j in (s - 1, e):  # "high priority", "Gold customers": the word naming the filtered field is not a column
            if 0 <= j < n and stems[j] in qualifier:
                used[j] = True
    for i in p.mod_idx:
        used[i] = True
    if p.limit is not None:
        if 1 <= p.limit <= MAX_LIMIT:
            node["limit"] = p.limit
        elif strict:
            return None, f"limit {p.limit} out of range"

    if p.sort_kind in ("oldest", "newest"):
        age = next((f for f in names if col.fields[f].type in NUMERIC and "age" in name_stems(f)), None)
        times = [f for f in names if col.fields[f].type in ("datetime", "date")]
        if p.sort_kind == "oldest" and age:
            node["sort_by"], node["sort_dir"] = age, "desc"
        elif times:
            node["sort_by"], node["sort_dir"] = times[0], ("asc" if p.sort_kind == "oldest" else "desc")
        elif strict:
            return None, "no date field to sort by"
    elif p.sort_kind in ("high", "low"):
        best = None
        for f in names:
            if col.fields[f].type not in NUMERIC:
                continue
            ts = list(dict.fromkeys(name_stems(f)))
            hit = [t for t in ts if t in stems and not used[stems.index(t)]]
            if len(hit) == len(ts) and (best is None or len(hit) > len(best[1])):
                best = (f, hit)
        if best is not None:
            node["sort_by"], node["sort_dir"] = best[0], "desc" if p.sort_kind == "high" else "asc"
            for t in best[1]:
                used[stems.index(t)] = True
        elif strict:
            return None, "no numeric field named for the sort"

    mentioned = []
    for f in names:
        ts = list(dict.fromkeys(name_stems(f)))
        pos = [stems.index(t) for t in ts if t in stems and not used[stems.index(t)]]
        if len(pos) == len(ts):
            mentioned.append((min(pos), f))
            for q in pos:
                used[q] = True
    fields = [f for _, f in sorted(mentioned)]  # columns in the order the user named them
    if not fields:
        fields = names[:DEFAULT_FIELDS]
        if node.get("sort_by") and node["sort_by"] not in fields:
            fields = fields[:DEFAULT_FIELDS - 1] + [node["sort_by"]]
    node["fields"] = fields
    return node, "ok"


def scalar_node(cat: Catalog, ptr: str) -> dict:
    sc = cat.scalars[ptr]
    kind = "metric" if sc.type in NUMERIC else ("status" if "status" in name_stems(ptr.rsplit("/", 1)[-1]) else "text")
    return {"type": kind, "value": ptr}


def evidence(prompt: str, cat: Catalog, kind: str, ptr: str, aliases: dict[str, str] | None = None) -> int:
    """How many content words of the prompt point at this value or list (its name, a parent name, one of its
    field names, or one of its enum values). 0 means nothing in the request refers to it."""
    p = parse(prompt, aliases)
    if p is None:
        return 0
    words = {s for s, u in zip(p.stems, p.used) if not u and len(s) > 2}
    if kind == "value":
        return sum(1 for t in {t for seg in ptr.split("/")[1:] for t in name_stems(seg)} if t in words)
    col = cat.collections[ptr]
    score = sum(1 for t in set(name_stems(ptr.rsplit("/", 1)[-1])) if t in words)
    score += sum(1 for f in col.fields for t in set(name_stems(f)) if len(t) > 2 and t in words)
    score += sum(1 for h in value_hits(cat, p.stems) if h[0] == ptr)
    return score


def fill_collection(prompt: str, cat: Catalog, ptr: str, aliases: dict[str, str] | None = None) -> dict:
    """Lenient node for a collection the model already chose: the model picks *what*, this adds the
    filter / sort / limit / columns the prompt states literally."""
    p = parse(prompt, aliases)
    if p is None:
        return {"type": "list", "source": ptr, "fields": list(cat.collections[ptr].fields)[:DEFAULT_FIELDS]}
    hits = [h for h in value_hits(cat, p.stems) if h[0] == ptr]
    node, _ = collection_node(cat, ptr, p, list(p.used), hits, strict=False)
    return node


def route(prompt: str, cat: Catalog, aliases: dict[str, str] | None = None) -> Route:
    p = parse(prompt, aliases)
    if p is None:
        return Route(None, "empty prompt")
    raws, stems, used, n = p.raws, p.stems, list(p.used), len(p.raws)
    if p.limit is not None and not 1 <= p.limit <= MAX_LIMIT:
        return Route(None, f"limit {p.limit} out of range")

    # ---- evidence: collections (by name, or by a multi-word enum value) ---------------------
    name_hits: dict[str, int] = {}
    for ptr in cat.collections:
        last = name_stems(ptr.rsplit("/", 1)[-1])[-1]
        if last in stems:
            name_hits[ptr] = stems.index(last)
    hits_all = value_hits(cat, stems)
    evidence = set(name_hits)
    for ptr, _f, _v, s, e in hits_all:
        if e - s > 1 or ptr in name_hits:  # a single common word ("new", "low") only counts next to the collection's name
            evidence.add(ptr)
    if len(evidence) > 1:
        return Route(None, "ambiguous: matches several data sources")

    # ---- scalar candidates ---------------------------------------------------------------------
    cands = []
    for ptr, sc in cat.scalars.items():
        segs = ptr.split("/")[1:]
        last = name_stems(segs[-1])
        parents = {t for seg in segs[:-1] for t in name_stems(seg)}
        matched = [t for t in dict.fromkeys(last) if t in stems]
        if matched:
            cands.append({"ptr": ptr, "cov": len(matched) / len(set(last)), "matched": matched,
                          "parents": [t for t in parents if t in stems]})
    cands.sort(key=lambda c: (-c["cov"], -len(c["matched"]), -len(c["parents"])))
    picked: list[dict] = []
    for c in cands:
        if c["cov"] < 0.5:
            continue
        if any(set(c["matched"]) & set(q["matched"]) for q in picked):
            if len(picked) == 1 and cands[0] is not c and (c["cov"], len(c["matched"])) == (picked[0]["cov"], len(picked[0]["matched"])) \
                    and set(c["matched"]) == set(picked[0]["matched"]):
                return Route(None, "ambiguous: several values match the same words")
            continue
        picked.append(c)

    def finish(plan: dict) -> Route:
        left = [raws[i] for i in range(n) if not used[i]]
        if left:
            return Route(None, f"unrecognized words: {', '.join(left[:4])}")
        errs = validate_plan(plan, cat)
        return Route(None, "invalid plan: " + errs[0]) if errs else Route(plan, "matched")

    # ---- scalar-only requests ------------------------------------------------------------
    if not evidence or (p.count_intent and picked):
        if not picked:
            return Route(None, "no matching value or data source")
        if p.count_intent and p.conjunction:
            return Route(None, "count question combined with another request")
        if evidence and not p.count_intent:
            return Route(None, "values and a data source both matched")
        for idx in name_hits.values():  # the collection noun in "how many cases ..."
            used[idx] = True
        if p.mod_idx:
            return Route(None, "sort/limit words without a list")
        for c in picked:
            for t in c["matched"] + c["parents"]:
                used[stems.index(t)] = True
        return finish({"nodes": [scalar_node(cat, c["ptr"]) for c in picked]})

    # ---- one collection ---------------------------------------------------------------------
    ptr = next(iter(evidence))
    if ptr in name_hits:
        used[name_hits[ptr]] = True
    hits = [h for h in hits_all if h[0] == ptr and (h[4] - h[3] > 1 or ptr in name_hits)]
    node, why = collection_node(cat, ptr, p, used, hits, strict=True)
    if node is None:
        return Route(None, why)
    if p.count_intent and p.conjunction:
        return Route(None, "count question combined with another request")
    return finish({"nodes": [node]})
