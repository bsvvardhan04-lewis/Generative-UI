"""L4 semantic level: match the request to the data by meaning, with a small local embedding model.

The router (and query rewrite) need the request to share words with the data. This level does not:
every value and list in the catalog is described in a short sentence, embedded once per data shape,
and the request is embedded and compared by cosine similarity. "how quickly do I get back to people"
can land on /today/firstResponseMinutes without sharing a word with it.

It answers only when the best match is both strong (score >= threshold) and clearly ahead of the
runner-up (gap >= margin); otherwise the request goes on to the language model. A confident wrong
answer skips the model, so the thresholds are set conservatively from calibration sets
(eval_semantic.py) and the result is checked on a separate set written before this module.

Measured result (see DEFAULT_THRESHOLD): with nomic-embed-text it is never wrong at the calibrated setting
but answers few requests, so it is off by default; all-minilm was not reliable at all.

Cost with nomic-embed-text (274 MB) via Ollama on the CPU-only test laptop: about 40-75 ms per request
(one query embedding); the catalog embeddings are computed once per data shape and cached.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass

from .catalog import Catalog
from .llm import _post
from .router import name_words
from .selector import picks_to_plan, select_options

UNIT_WORDS = {"seconds": "a duration of time, how long", "minutes": "a duration of time, how long",
              "hours": "a duration of time, how long", "percent": "a percentage rate, a share or proportion"}
TYPE_WORDS = {"integer": "a number", "number": "a number", "datetime": "a date and time", "date": "a date",
              "boolean": "yes or no", "email": "an email address", "url": "a web link", "string": "a text value"}

# Calibrated on the dev / held-out / held-out 2 / paraphrase sets (see eval_semantic.py); not on semantic_cases.json.
# all-minilm: correct and unrelated requests score alike; the only no-wrong setting (0.60 / 0.03) answers 1 of 28,
#   and looser settings are wrong on the fresh set (0.45 / 0.06: 2 answered there, 1 wrong).
# nomic-embed-text (used here): unrelated requests stay below 0.50, but close options often tie. With a 0.10 gap
#   it is never wrong on either set (0.55 / 0.10: 4 of 28 answered in calibration, 1 of 18 on the fresh set).
# Safe, but it answers too little to be worth ~50 ms on every request the router declines, so the server still
# keeps this level off unless asked (server.py --semantic). Re-run eval_semantic.py before changing these.
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_THRESHOLD = 0.55
DEFAULT_MARGIN = 0.10


def _related(words: list[str]) -> list[str]:
    """Generic synonyms of the name's words (the same groups the query rewrite uses), to widen what a
    description can mean: 'csat' also says 'satisfaction, happiness'."""
    from .rewrite import CONCEPTS
    out = []
    for w in words:
        for g in CONCEPTS:
            if w in g or w.rstrip("s") in g:
                out += sorted(g - {w, w.rstrip("s")})
    return list(dict.fromkeys(out))


def describe(cat: Catalog, kind: str, ptr: str) -> str:
    """A short natural-language description of one catalog option, for embedding."""
    segs = [s for s in ptr.split("/")[1:] if s]
    name = [w for s in segs for w in name_words(s)]
    related = _related(name)
    also = f" (also: {', '.join(related)})" if related else ""
    words = " ".join(name)
    if kind == "value":
        s = cat.scalars[ptr]
        what = UNIT_WORDS.get(s.unit or "", TYPE_WORDS.get(s.type, "a value"))
        return f"{words}: {what}{also}"
    c = cat.collections[ptr]
    fields = ", ".join(" ".join(name_words(f)) for f in list(c.fields)[:12])
    values = sorted({v for f in c.fields.values() for v in (f.values or [])})[:12]
    text = f"list of {words}{also}: {c.count} items, each with {fields}"
    return text + (f"; values such as {', '.join(values)}" if values else "")


# Some embedding models are trained with a task prefix on each text and score poorly without it.
PREFIXES = {"nomic-embed-text": {"query": "search_query: ", "document": "search_document: "}}


class OllamaEmbedder:
    def __init__(self, model: str = DEFAULT_EMBED_MODEL, base_url: str = "http://127.0.0.1:11434", timeout: float = 30):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.name = f"ollama:{model}"
        self.prefixes = PREFIXES.get(model.split(":")[0], {})

    def embed(self, texts: list[str], kind: str = "query") -> list[list[float]]:
        pre = self.prefixes.get(kind, "")
        r = _post(f"{self.base_url}/api/embed", {"model": self.model, "input": [pre + t for t in texts], "keep_alive": "30m"},
                  self.timeout)
        return [_unit(v) for v in r["embeddings"]]


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Match:
    plan: dict | None
    reason: str
    score: float = 0.0
    margin: float = 0.0
    best: str = ""
    second: str = ""


class SemanticIndex:
    """Embeds each data shape's options once (cached by structure fingerprint) and ranks requests against them."""

    def __init__(self, embedder, threshold: float = DEFAULT_THRESHOLD, margin: float = DEFAULT_MARGIN):
        self.embedder, self.threshold, self.margin = embedder, threshold, margin
        self._cache: dict[str, tuple[list, list]] = {}
        self._lock = threading.Lock()

    def _options(self, cat: Catalog):
        from .pipeline import structure_fingerprint  # local import: pipeline imports this module
        key = structure_fingerprint(cat)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        opts = select_options(cat)
        vecs = self.embedder.embed([describe(cat, k, p) for k, p in opts], kind="document")
        with self._lock:
            self._cache[key] = (opts, vecs)
        return opts, vecs

    def warm(self, cat: Catalog) -> None:
        self._options(cat)

    def rank(self, query: str, cat: Catalog) -> list[tuple[float, str, str]]:
        opts, vecs = self._options(cat)
        q = self.embedder.embed([query])[0]
        return sorted(((_dot(q, v), k, p) for (k, p), v in zip(opts, vecs)), reverse=True)

    def match(self, query: str, cat: Catalog) -> Match:
        ranked = self.rank(query, cat)
        if not ranked:
            return Match(None, "no options")
        (s1, k1, p1), (s2, _, p2) = ranked[0], ranked[1] if len(ranked) > 1 else (0.0, "", "")
        info = dict(score=round(s1, 3), margin=round(s1 - s2, 3), best=p1, second=p2)
        if s1 < self.threshold:
            return Match(None, f"best match {p1} too weak ({s1:.2f} < {self.threshold})", **info)
        if s1 - s2 < self.margin:
            return Match(None, f"{p1} and {p2} too close ({s1 - s2:.2f} < {self.margin})", **info)
        plan, errors = picks_to_plan({"pick": [p1]}, query, cat, select_options(cat))
        return Match(plan if not errors else None, "matched" if not errors else errors[0], **info)


def to_dict(m: Match) -> dict:
    return json.loads(json.dumps({"score": m.score, "margin": m.margin, "best": m.best, "second": m.second, "reason": m.reason}))
