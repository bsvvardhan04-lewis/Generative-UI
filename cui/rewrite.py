"""Query rewrite: level-wise enrichment of the user's request before routing, caching and filling.

Each level is cheap (well under a millisecond) and only ever rewrites a word into the data's own vocabulary,
so more requests can be answered by the cache or the router instead of the model:

  L0 normalize  lower case, contractions ("who's" -> "who is"), "&" -> "and", punctuation -> spaces
  L1 spelling   a word of 6+ letters that is not in the vocabulary -> the single closest vocabulary word
                (edit distance 1, or 2 for words of 8+ letters; same first letter; unique best match)
  L2 concepts   generic synonyms, data-aware: "client" -> "customer" only if the data says "customer"
                (and does not itself use "client"); a group with several matches in the data is left alone
  L3 aliases    data/aliases.json phrases ("tickets" -> "cases"), longest phrase first

The model still sees the user's original wording; the rewritten text drives the router, the code that adds
filters/sort/limit/columns, and the cache key, so "show me escalted tickets" and "escalated cases" share one entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import Catalog
from .router import (COUNT_WORDS, FILLER, HIGH, LOW, NEWEST, NUMBER_WORDS, OLDEST, STOP, apply_aliases,
                     name_words, stem)

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "who's": "who is", "how's": "how is", "where's": "where is",
    "when's": "when is", "there's": "there is", "it's": "it is", "that's": "that is", "i'm": "i am",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "won't": "will not", "let's": "let us", "i've": "i have",
}

# Generic, domain-independent synonym groups. A word is only rewritten to a group member that the data
# actually uses, so these never invent a field: on a dataset without "customer", "client" stays "client".
CONCEPTS = [
    {"customer", "client", "buyer", "shopper", "consumer", "purchaser"},
    {"case", "ticket", "issue", "incident"},
    {"team", "colleague", "coworker", "teammate", "crew"},
    {"employee", "staff", "worker", "personnel"},
    {"resolved", "solved", "closed", "fixed", "completed", "finished"},
    {"open", "unresolved", "outstanding", "active"},
    {"priority", "urgency", "severity", "importance"},
    {"status", "state"},
    {"average", "mean", "typical"},
    {"satisfaction", "csat", "happiness"},
    {"order", "purchase"},
    {"product", "sku", "merchandise"},
    {"price", "cost", "fee"},
    {"revenue", "sales", "income", "earnings", "turnover"},
    {"created", "opened", "submitted", "filed", "raised", "logged"},
    {"updated", "modified", "changed", "edited"},
    {"email", "mail"},
    {"phone", "telephone"},
    {"region", "area", "zone", "territory"},
    {"company", "organization", "organisation", "firm", "business"},
    {"invoice", "bill"},
    {"shipment", "delivery", "shipping", "dispatch"},
    {"inventory", "stock"},
    {"device", "machine", "equipment"},
    {"error", "failure", "fault", "exception"},
    {"activity", "event", "history", "log", "timeline"},
    {"response", "reply", "answer"},
    {"note", "comment", "remark"},
    {"rating", "stars", "review"},
    {"salary", "pay", "compensation", "wage"},
    {"department", "dept", "division"},
    {"manager", "supervisor", "boss"},
    {"wait", "delay", "hold"},
]
GENERIC = (STOP | FILLER | COUNT_WORDS | OLDEST | NEWEST | HIGH | LOW | set(NUMBER_WORDS)
           | {"top", "first", "last", "table", "list", "via", "through", "per", "vs", "versus", "today", "week", "month", "year"})
GENERIC_STEMS = {stem(w) for w in GENERIC}  # so plurals of generic words ("updates") are never remapped


@dataclass
class Rewrite:
    original: str
    text: str
    steps: list[dict] = field(default_factory=list)  # {"level", "from", "to"} for every change

    @property
    def changed(self) -> bool:
        return bool(self.steps)


def data_words(cat: Catalog) -> set[str]:
    """Every word the data uses: path segments, field names and enum values, split and lower-cased."""
    words: set[str] = set()
    for ptr in list(cat.scalars) + list(cat.collections):
        for seg in ptr.split("/")[1:]:
            words.update(name_words(seg))
    for col in cat.collections.values():
        for f in col.fields.values():
            words.update(name_words(f.name))
            for v in f.values or []:
                words.update(re.findall(r"[a-z0-9]+", v.lower()))
    return {w for w in words if not w.isdigit()}


def _distance(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance (Damerau-Levenshtein with adjacent swaps), early exit above cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def _spell(word: str, vocab: set[str]) -> str | None:
    # Short words have too many one-letter neighbours ("stats" -> "status"), so only 6+ letter words are corrected.
    if len(word) < 6 or not word.isalpha():
        return None
    cap = 2 if len(word) >= 8 else 1
    best, best_d, tie = None, cap + 1, False
    for v in vocab:
        if v[0] != word[0] or abs(len(v) - len(word)) > cap:
            continue
        d = _distance(word, v, cap)
        if d < best_d:
            best, best_d, tie = v, d, False
        elif d == best_d:
            tie = True
    return best if best is not None and not tie and best_d <= cap else None


def rewrite(prompt: str, cat: Catalog, aliases: dict[str, str] | None = None) -> Rewrite:
    steps: list[dict] = []
    aliases = aliases or {}

    # L0 normalize
    text = prompt.lower().replace("’", "'").replace("&", " and ")
    for c, full in CONTRACTIONS.items():
        if re.search(rf"(?<![a-z']){re.escape(c)}(?![a-z'])", text):
            text = re.sub(rf"(?<![a-z']){re.escape(c)}(?![a-z'])", full, text)
            if c != "whats":
                steps.append({"level": "normalize", "from": c, "to": full})
    text = " ".join(re.findall(r"[a-z0-9]+", text))

    words_in_data = data_words(cat)
    data_stems = {stem(w) for w in words_in_data}
    alias_words = {w for phrase in aliases for w in re.findall(r"[a-z0-9]+", phrase.lower())}
    concept_words = set().union(*CONCEPTS)
    targets = words_in_data | alias_words | GENERIC  # a typo is only ever corrected to one of these
    known = targets | concept_words                   # ...and a synonym-group word is valid as typed, not a typo

    # L1 spelling
    known_stems = {stem(w) for w in known}
    out = []
    for w in text.split():
        if w in known or stem(w) in known_stems or w.isdigit():
            out.append(w)
            continue
        fixed = _spell(w, targets)
        if fixed:
            steps.append({"level": "spelling", "from": w, "to": fixed})
            out.append(fixed)
        else:
            out.append(w)
    text = " ".join(out)

    # L2 concepts (data-aware synonyms). Before aliases, so "client satisfaction" can still meet the
    # "customer satisfaction" alias. Words the aliases know are left for the aliases.
    out = []
    for w in text.split():
        s = stem(w)
        if s in data_stems or w in GENERIC or s in GENERIC_STEMS or w in alias_words:
            out.append(w)
            continue
        # exact word or simple plural only: stems are too loose here ("stats" and "state" share a stem)
        forms = {w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w}
        found = {m for g in CONCEPTS if forms & g for m in g if stem(m) in data_stems}
        if len(found) == 1:
            to = next(iter(found))
            steps.append({"level": "concept", "from": w, "to": to})
            out.append(to)
        else:
            out.append(w)
    text = " ".join(out)

    # L3 aliases
    if aliases:
        table = {k.lower(): v.lower() for k, v in aliases.items() if k.strip()}
        pattern = "|".join(re.escape(p) for p in sorted(table, key=len, reverse=True))
        for m in re.finditer(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", text):
            steps.append({"level": "alias", "from": m.group(0), "to": table[m.group(0)]})
        text = " ".join(apply_aliases(text, table).split())

    # collapse a word repeated by a rewrite ("first response time" -> "first first response")
    words = text.split()
    text = " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])
    return Rewrite(prompt, text, steps)
