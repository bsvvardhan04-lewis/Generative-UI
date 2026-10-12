"""Orchestrates the stages: catalog -> constrained LLM plan -> validate/repair -> compile."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog, build_catalog, render_catalog
from .compiler import compile_plan, default_plan
from .rewrite import rewrite
from .router import STOP, apply_aliases, route, stem
from .schema import MAX_NODES, build_plan_schema
from .selector import SELECT_VERSION, picks_to_plan, select_messages, select_options, select_schema
from .validate import validate_plan

# Bump when SYSTEM changes so stored eval results stay comparable. v1 had a filter + limit in the example.
PROMPT_VERSION = "v3"  # v3: model output has no label / intent / format (derived by the compiler) to cut generated tokens

SYSTEM = """You turn a user request into a minimal UI plan over an API response.
You only see a DATA CATALOG (paths and example values), never the full data. Reply with ONE JSON object matching the schema.

Rules:
- Include only what the request asks for. Fewer nodes is better (usually 1-4).
- Use ONLY pointers and field names that appear in the catalog.
- metric: one numeric value. text: one plain value. status: one state/label value.
- list: a short collection shown as title/subtitle rows. table: use only when the user asks for a table or wants to compare several columns.
- `fields` order matters: first field is the row title, second the subtitle.
- `filter`, `sort_by`/`sort_dir` and `limit` are OPTIONAL. Leave them out unless the request asks for them:
  filter only when the user names a specific value (e.g. "high priority" -> field priority, equals High);
  sort_by/sort_dir for words like "oldest", "latest", "highest"; limit only for an explicit count ("top 3").
  Never add a filter just to make the answer look more specific.

Example (illustrative paths, neutral request):
{"nodes":[{"type":"metric","value":"/stats/open"},{"type":"list","source":"/tickets","fields":["title","owner"]}]}"""


@dataclass
class Result:
    prompt: str
    plan: dict
    dsl: dict
    attempts: list[dict] = field(default_factory=list)
    fallback: bool = False
    errors: list[str] = field(default_factory=list)
    catalog_chars: int = 0
    total_seconds: float = 0.0
    catalog_text: str = ""
    stages: dict = field(default_factory=dict)
    rewrite: dict = field(default_factory=dict)  # {"text": rewritten request, "steps": [{"level","from","to"}]}


def build_messages(prompt: str, cat: Catalog) -> list[dict]:
    # Instructions and catalog come first and the request last, so a backend with a prompt cache
    # (Ollama, llama.cpp) only has to process the few new request tokens on repeat calls.
    user = f"DATA CATALOG\n{render_catalog(cat)}\n\nUSER REQUEST: {prompt}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prefix_messages(cat: Catalog, mode: str = "plan") -> list[dict]:
    """The shared part of every request for this data, used to pre-fill the prompt cache."""
    if mode == "select":
        return select_messages("", cat, select_options(cat))
    return build_messages("", cat)


def structure_fingerprint(cat: Catalog) -> str:
    """Hash of the data's shape (pointers, field names/types, enum values) but not of its values,
    so cached plans survive refreshes where only the numbers changed."""
    shape = {
        "s": sorted((s.pointer, s.type, s.unit) for s in cat.scalars.values()),
        "c": sorted((c.pointer, sorted((f.name, f.type, tuple(f.values or ())) for f in c.fields.values())) for c in cat.collections.values()),
    }
    return hashlib.sha1(json.dumps(shape, sort_keys=True, default=list).encode()).hexdigest()[:16]


_KEEP = {"and", "or", "not"}  # filler words that can change the meaning stay in the cache key


def normalize_prompt(prompt: str, aliases: dict[str, str] | None = None) -> str:
    """Order- and filler-insensitive key: "Show me the HIGH priority cases!" and "cases, high priority"
    are the same request. Aliases are applied first so "tickets" and "cases" share an entry."""
    words = re.findall(r"[a-z0-9]+", apply_aliases(prompt, aliases))
    return " ".join(sorted(stem(w) for w in words if w not in STOP or w in _KEEP))


class PlanCache:
    """Thread-safe LRU of validated plans. A hit skips the model entirely. With `path` set, entries are
    saved to disk after each change and reloaded on start, so seen prompts stay instant across restarts."""

    def __init__(self, size: int = 512, path=None):
        self.size, self._d, self._lock, self.path = size, OrderedDict(), threading.Lock(), path
        self.hits = self.misses = 0
        if path is not None:
            self._load()

    @staticmethod
    def key(backend_name: str, cat: Catalog, prompt: str, aliases: dict[str, str] | None = None, version: str = PROMPT_VERSION) -> tuple:
        return (backend_name, version, structure_fingerprint(cat), normalize_prompt(prompt, aliases))

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._d[key])
            self.misses += 1
            return None

    def put(self, key, plan) -> None:
        with self._lock:
            self._d[key] = copy.deepcopy(plan)
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)
            self._save()

    def __len__(self) -> int:
        return len(self._d)

    def _load(self) -> None:
        try:
            for k, plan in json.loads(Path(self.path).read_text(encoding="utf-8")):
                self._d[tuple(k)] = plan
        except (OSError, ValueError, TypeError):
            self._d.clear()  # unreadable cache: start empty rather than fail

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps([[list(k), v] for k, v in self._d.items()], ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass  # a read-only disk must never break a request


def run(prompt: str, data, backend, max_retries: int = 2, on_event=None, cache: PlanCache | None = None,
        fast_path: bool = False, aliases: dict[str, str] | None = None, mode: str = "plan", semantic=None) -> Result:
    """Run the pipeline. `on_event(name, data)` is called as each stage starts/finishes (used by the web UI).
    mode="plan": the model writes the whole view plan. mode="select": the model only picks numbered catalog
    items and code adds filter/sort/limit/columns (far fewer generated tokens, see selector.py)."""
    if mode not in ("plan", "select"):
        raise ValueError(f"unknown mode {mode!r}")
    emit = on_event or (lambda name, payload=None: None)
    t0 = time.perf_counter()
    stages: dict = {}

    t = time.perf_counter()
    cat = build_catalog(data)
    catalog_text = render_catalog(cat)
    stages["catalog"] = {
        "seconds": round(time.perf_counter() - t, 4),
        "scalars": len(cat.scalars),
        "collections": len(cat.collections),
        "catalog_chars": len(catalog_text),
        "raw_chars": len(json.dumps(data)),
    }
    emit("catalog", stages["catalog"])

    # Query rewrite (L0 normalize, L1 spelling, L2 concepts, L3 aliases): the rewritten text drives the cache key,
    # the router and the code that adds filters/sort/limit/columns. The model keeps the user's own wording.
    t = time.perf_counter()
    rw = rewrite(prompt, cat, aliases)
    query = rw.text
    stages["rewrite"] = {"seconds": round(time.perf_counter() - t, 4), "text": query, "steps": rw.steps}
    if rw.changed:
        emit("rewrite", stages["rewrite"])

    t = time.perf_counter()
    if mode == "select":
        opts = select_options(cat)
        schema = select_schema(opts)
        messages = select_messages(prompt, cat, opts)
        variants, version = len(opts), SELECT_VERSION
    else:
        schema = build_plan_schema(cat)
        messages = build_messages(prompt, cat)
        variants, version = len(schema["properties"]["nodes"]["items"]["anyOf"]), PROMPT_VERSION
    stages["schema"] = {"seconds": round(time.perf_counter() - t, 4), "variants": variants, "schema_chars": len(json.dumps(schema)), "mode": mode}
    emit("schema", stages["schema"])

    if hasattr(backend, "bind"):
        backend.bind(cat, prompt)

    attempts: list[dict] = []
    plan, errors = None, ["not run"]
    cache_key = PlanCache.key(getattr(backend, "name", "model"), cat, query, None, version) if cache is not None else None
    cached = cache.get(cache_key) if cache is not None else None
    routed = None
    if cached is not None:
        plan, errors = cached, []
        attempts.append({"cached": True, "seconds": 0.0, "errors": []})
        emit("cache_hit", {"backend": getattr(backend, "name", "model")})
    elif fast_path:
        t = time.perf_counter()
        routed = route(query, cat)
        took = round(time.perf_counter() - t, 4)
        if routed.plan is not None:
            plan, errors = routed.plan, []
            attempts.append({"fast_path": True, "seconds": took, "errors": []})
            emit("fast_path", {"seconds": took})
        else:
            emit("fast_path_declined", {"reason": routed.reason, "seconds": took})
    matched = None
    if cached is None and (routed is None or routed.plan is None) and semantic is not None:
        # L4: match by meaning with a small embedding model; answers only on a strong, unambiguous match
        t = time.perf_counter()
        try:
            matched = semantic.match(query, cat)
        except RuntimeError as e:  # embedding model missing or Ollama down: just skip this level
            emit("semantic_unavailable", {"message": str(e)})
        else:
            took = round(time.perf_counter() - t, 4)
            info = {"seconds": took, "score": matched.score, "margin": matched.margin, "best": matched.best, "reason": matched.reason}
            if matched.plan is not None:
                plan, errors = matched.plan, []
                attempts.append({"semantic": True, "seconds": took, "errors": [], "score": matched.score, "best": matched.best})
                emit("semantic", info)
            else:
                emit("semantic_declined", info)
    model_needed = cached is None and (routed is None or routed.plan is None) and (matched is None or matched.plan is None)
    for attempt in range(1, (max_retries + 2) if model_needed else 1):
        emit("model_start", {"attempt": attempt, "max_attempts": max_retries + 1, "backend": getattr(backend, "name", "model")})
        t = time.perf_counter()
        try:
            text, meta = backend.generate(messages, schema)
        except RuntimeError as e:  # transport/model error: do not loop on it
            attempts.append({"error": str(e)})
            errors = [str(e)]
            plan = None
            emit("model_error", {"attempt": attempt, "message": str(e)})
            break
        seconds = round(time.perf_counter() - t, 2)
        emit("model_done", {"attempt": attempt, "seconds": seconds, **meta})
        t = time.perf_counter()
        try:
            out = json.loads(text)
            if mode == "select":
                plan, errors = picks_to_plan(out, query, cat, opts)
                errors = errors or validate_plan(plan, cat)
            else:
                plan, errors = out, validate_plan(out, cat)
        except json.JSONDecodeError as e:
            plan, errors = None, [f"output was not valid JSON ({e.msg} at char {e.pos})"]
        attempts.append({"seconds": seconds, "errors": errors, **meta})
        emit("validate", {"attempt": attempt, "errors": errors, "will_retry": bool(errors) and attempt < max_retries + 1, "seconds": round(time.perf_counter() - t, 4)})
        if not errors:
            break
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your plan has errors. Fix them and reply with the corrected JSON only:\n- " + "\n- ".join(errors[:8])},
        ]

    fallback = bool(errors)
    if not fallback and cache is not None and model_needed:  # only model answers are cached; the router is already faster
        cache.put(cache_key, plan)
    if fallback:
        plan = default_plan(cat)
    plan["nodes"] = plan["nodes"][:MAX_NODES]
    t = time.perf_counter()
    dsl = compile_plan(plan, data, cat, prompt)
    stages["compile"] = {"seconds": round(time.perf_counter() - t, 4), "nodes": len(plan["nodes"]), "fallback": fallback}
    emit("compile", stages["compile"])
    return Result(
        prompt, plan, dsl, attempts, fallback, errors if fallback else [],
        len(catalog_text), round(time.perf_counter() - t0, 4), catalog_text, stages,
        {"text": query, "steps": rw.steps},
    )
