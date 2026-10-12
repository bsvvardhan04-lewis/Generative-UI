"""Model backends. All implement `generate(messages, schema) -> (text, meta)`.

* OllamaBackend       - local Ollama; schema goes in `format` (grammar-constrained).
* OpenAICompatBackend - llama.cpp `llama-server`, vLLM, LM Studio, ... via
                        /v1/chat/completions + response_format=json_schema.
* HeuristicBackend    - NOT an LLM. Keyword matcher used as a no-model baseline
                        and to exercise the pipeline in tests/CI.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .catalog import Catalog


def prefer_ipv4(url: str) -> str:
    """'localhost' -> '127.0.0.1'. On Windows, 'localhost' resolves to ::1 first; Ollama listens on IPv4 only, so
    every request waited ~2 s for the IPv6 attempt to fail (measured: 2,070 ms vs 16-57 ms for the same call)."""
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url)


def _post(url: str, body: dict, timeout: float) -> dict:
    url = prefer_ipv4(url)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url}: {e.reason}") from None


class OllamaBackend:
    """num_ctx stays fixed per model: changing it makes Ollama reload the runner.
    keep_alive keeps the weights in RAM between requests (the default is 5 minutes).
    num_predict caps a runaway generation (small models sometimes repeat nodes for minutes)."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", num_ctx: int = 4096, timeout: float = 600,
                 keep_alive: str = "30m", num_predict: int = 256):
        self.model, self.base_url, self.num_ctx, self.timeout = model, base_url.rstrip("/"), num_ctx, timeout
        self.keep_alive, self.num_predict = keep_alive, num_predict
        self.name = f"ollama:{model}"

    def _options(self, num_predict: int) -> dict:
        return {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": num_predict}

    def generate(self, messages, schema):
        r = _post(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": self._options(self.num_predict),
            },
            self.timeout,
        )
        return r["message"]["content"], {
            "prompt_tokens": r.get("prompt_eval_count"),
            "output_tokens": r.get("eval_count"),
            "model_seconds": (r.get("total_duration") or 0) / 1e9,
            "load_seconds": (r.get("load_duration") or 0) / 1e9,
        }

    def warm(self, messages=None) -> dict:
        """Load the model, page its weights in, and (when given the real prompt prefix) fill the
        prompt cache so the next request only has to process the user's request tokens."""
        r = _post(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages or [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": self._options(1),
            },
            self.timeout,
        )
        return {"load_seconds": (r.get("load_duration") or 0) / 1e9, "prompt_tokens": r.get("prompt_eval_count"), "total_seconds": (r.get("total_duration") or 0) / 1e9}


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8080/v1", api_key: str = "", timeout: float = 600):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.name = f"openai-compat:{model}"

    def generate(self, messages, schema):
        r = _post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": {"name": "view_plan", "strict": True, "schema": schema}},
            },
            self.timeout,
        )
        u = r.get("usage", {})
        return r["choices"][0]["message"]["content"], {"prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}


# --- heuristic baseline ------------------------------------------------------

def _tokens(text: str) -> set[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    out = set()
    for w in re.findall(r"[a-zA-Z]+", text.lower()):
        out.add(w)
        if w.endswith("ies"):
            out.add(w[:-3] + "y")
        elif w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
    return out


class HeuristicBackend:
    """Keyword overlap between the prompt and catalog names. Needs the catalog, so
    the pipeline hands it over via `bind`."""

    name = "heuristic-baseline"

    def __init__(self):
        self.cat: Catalog | None = None
        self.prompt = ""

    def bind(self, cat: Catalog, prompt: str):
        self.cat, self.prompt = cat, prompt

    def generate(self, messages, schema):
        cat, q = self.cat, _tokens(self.prompt)
        q_raw = self.prompt.lower()
        nodes = []
        scored = []
        for s in cat.scalars.values():
            if s.type in ("url", "email"):
                continue
            segs = s.pointer.split("/")[1:]
            score = len(q & _tokens(" ".join(segs[-1:]))) * 2 + len(q & _tokens(" ".join(segs[:-1])))
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        best = scored[0][0] if scored else 0
        for score, s in scored:
            if score < best or len(nodes) >= 4:
                continue
            if s.type in ("integer", "number"):
                nodes.append({"type": "metric", "value": s.pointer})
            elif s.type == "boolean":
                nodes.append({"type": "text", "value": s.pointer})
            else:
                nodes.append({"type": "status", "value": s.pointer})

        for col in cat.collections.values():
            cname = _tokens(col.pointer)
            fname = set().union(*[_tokens(n) for n in col.fields]) if col.fields else set()
            if not (q & cname):
                if not (q & fname) or nodes:
                    continue
            names = list(col.fields)
            picked = [n for n in names if q & _tokens(n)] or names[:3]
            node = {"type": "table" if "table" in q_raw else "list", "source": col.pointer, "fields": picked[:4]}
            for fname_, f in col.fields.items():
                for v in f.values or []:
                    if v.lower() in q_raw:
                        node["filter"] = {"field": fname_, "equals": v}
            m = re.search(r"\b(?:top|first|latest|last|oldest|newest)\s+(\d+)\b", q_raw)
            if m:
                node["limit"] = min(int(m.group(1)), 20)
            if "oldest" in q_raw:
                node["sort_by"], node["sort_dir"] = (next((n for n in names if "age" in n.lower()), names[0]), "desc")
            elif "latest" in q_raw or "recent" in q_raw:
                tcol = next((n for n, f in col.fields.items() if f.type == "datetime"), None)
                if tcol:
                    node["sort_by"], node["sort_dir"] = tcol, "desc"
            nodes.append(node)

        if not nodes:
            nums = [s for s in cat.scalars.values() if s.type in ("integer", "number")][:3]
            nodes = [{"type": "metric", "value": s.pointer} for s in nums]
        return json.dumps({"nodes": nodes}), {"prompt_tokens": 0, "output_tokens": 0}
