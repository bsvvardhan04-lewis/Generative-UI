"""Local web server: static UI (web/) + JSON / NDJSON API around the pipeline.

    python server.py            # http://127.0.0.1:8765

Binds to loopback only. POSTs must be application/json and the Host header must be
a loopback name, which blocks cross-site form posts and DNS-rebinding requests.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import eval as evaluation  # eval.py: gold-case scoring
from cui import run
from cui.adapters import RENDERERS
from cui.compiler import DSL_VERSION
from cui.llm import HeuristicBackend, OllamaBackend, OpenAICompatBackend
from cui.catalog import build_catalog
from cui.pipeline import PROMPT_VERSION, PlanCache, prefix_messages

HERE = Path(__file__).parent
PLAN_CACHE = PlanCache(path=HERE / "cache" / "plans.json")
# "select": the model picks catalog paths (~12 tokens) instead of writing a plan (32-65 tokens). On the fresh
# held-out set (data/heldout2_cases.json) it matched or beat "plan" accuracy at ~40% lower median latency.
DEFAULT_MODE = "select"


def load_aliases() -> dict[str, str]:
    """Alias search: everyday phrase -> the data's own word. Edit data/aliases.json; read on every request."""
    try:
        raw = json.loads((HERE / "data" / "aliases.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}

WEB = HERE / "web"
DATA = HERE / "data"
RESULTS = HERE / "results"
DEFAULT_OLLAMA = "http://localhost:11434"
MAX_BODY = 2_000_000
MAX_PROMPT = 500
LOOPBACK = {"localhost", "127.0.0.1", "[::1]"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def clean_base_url(url: str | None, default: str) -> str:
    url = (url or default).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ApiError(400, "Server URL must start with http:// or https://")
    return url


def http_get_json(url: str, timeout: float = 2.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def list_models(base_url: str) -> dict:
    try:
        tags = http_get_json(base_url + "/api/tags")
        try:
            version = http_get_json(base_url + "/api/version").get("version")
        except (urllib.error.URLError, OSError, ValueError):
            version = None
        models = [
            {"name": m["name"], "size": m.get("size", 0), "modified": m.get("modified_at"), "family": (m.get("details") or {}).get("family")}
            for m in tags.get("models", [])
        ]
        models.sort(key=lambda m: m["name"])
        return {"reachable": True, "url": base_url, "version": version, "models": models}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"reachable": False, "url": base_url, "version": None, "models": [], "error": str(getattr(e, "reason", e))}


NOT_DATASETS = {"eval_cases", "heldout_cases", "heldout2_cases", "aliases"}  # configuration and test files that live in data/


def list_datasets() -> list[dict]:
    out = []
    for p in sorted(DATA.glob("*.json")):
        if p.stem in NOT_DATASETS:
            continue
        out.append({"id": p.stem, "name": p.stem.replace("_", " "), "bytes": p.stat().st_size})
    return out


def load_dataset(dataset_id: str):
    if not dataset_id.replace("_", "").replace("-", "").isalnum():
        raise ApiError(400, "Invalid dataset id")
    path = DATA / f"{dataset_id}.json"
    if not path.is_file() or path.stem in NOT_DATASETS:
        raise ApiError(404, f"Unknown dataset '{dataset_id}'")
    return json.loads(path.read_text(encoding="utf-8"))


def make_backend(body: dict):
    kind = body.get("backend", "heuristic")
    model = (body.get("model") or "").strip()
    if kind == "heuristic":
        return HeuristicBackend()
    if kind == "ollama":
        if not model:
            raise ApiError(400, "Choose a model")
        return OllamaBackend(model, clean_base_url(body.get("baseUrl"), DEFAULT_OLLAMA))
    if kind == "openai":
        if not model:
            raise ApiError(400, "Choose a model")
        return OpenAICompatBackend(model, clean_base_url(body.get("baseUrl"), "http://localhost:8080/v1"))
    raise ApiError(400, f"Unknown backend '{kind}'")


def parse_run_request(body: dict):
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ApiError(400, "Enter a request first")
    if len(prompt) > MAX_PROMPT:
        raise ApiError(400, f"Request is too long (max {MAX_PROMPT} characters)")
    if "data" in body:
        data = body["data"]
    else:
        data = load_dataset(str(body.get("dataset") or "sample_api"))
    if not isinstance(data, dict) or not data:
        raise ApiError(400, "API JSON must be a non-empty JSON object")
    retries = body.get("retries", 2)
    if not isinstance(retries, int) or not 0 <= retries <= 3:
        raise ApiError(400, "Repair attempts must be between 0 and 3")
    return prompt, data, make_backend(body), retries, model_mode(body)


def model_mode(body: dict) -> str:
    """How the model is asked: "select" (picks catalog items; few tokens, fast) or "plan" (writes the whole plan)."""
    mode = body.get("mode", DEFAULT_MODE)
    if mode not in ("select", "plan"):
        raise ApiError(400, 'mode must be "select" or "plan"')
    return mode


def run_payload(res, backend) -> dict:
    renders = {}
    for name, render in RENDERERS.items():
        out = render(res.dsl)
        renders[name] = out
    return {
        "prompt": res.prompt,
        "backend": backend.name,
        "promptVersion": PROMPT_VERSION,
        "dslVersion": DSL_VERSION,
        "plan": res.plan,
        "dsl": res.dsl,
        "renders": renders,
        "attempts": res.attempts,
        "cacheHit": any(a.get("cached") for a in res.attempts),
        "source": "cache" if any(a.get("cached") for a in res.attempts) else "fast_path" if any(a.get("fast_path") for a in res.attempts) else "model",
        "fallback": res.fallback,
        "errors": res.errors,
        "stages": res.stages,
        "catalogText": res.catalog_text,
        "totalSeconds": res.total_seconds,
    }


def warm_model(body: dict) -> dict:
    """Load the model into RAM and pre-fill its prompt cache with this data's catalog, so the first real
    request does not pay for model loading (30-50 s on a CPU) or for processing the catalog."""
    backend = make_backend(body)
    if not hasattr(backend, "warm"):
        return {"warmed": False, "reason": "backend needs no warm-up"}
    messages = None
    data = body.get("data")
    if isinstance(data, dict) and data:
        messages = prefix_messages(build_catalog(data), model_mode(body))
    try:
        info = backend.warm(messages)
    except RuntimeError as e:
        return {"warmed": False, "error": str(e)}
    return {"warmed": True, **info}


def load_eval() -> dict:
    cases = json.loads((DATA / "eval_cases.json").read_text(encoding="utf-8"))
    by_prompt = {c["prompt"]: c for c in cases}
    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = []
        for r in doc.get("rows", []):
            case = by_prompt.get(r.get("prompt"))
            if case is None:
                continue
            # Rescore from the stored plan so every run is judged by the current scorer.
            res = SimpleNamespace(plan=r["plan"], fallback=r["fallback"], attempts=[{}] * (1 if r.get("first_try_valid") else 2), total_seconds=r["seconds"])
            s = evaluation.score(case, res)
            s.update(prompt=r["prompt"], plan=r["plan"], errors=r.get("errors", []))
            rows.append(s)
        if not rows:
            continue
        n = len(rows)
        runs.append({
            "id": path.stem,
            "backend": doc.get("backend"),
            "promptVersion": doc.get("prompt_version"),
            "date": doc.get("date"),
            "summary": {
                "n": n,
                "passed": sum(r["pass"] for r in rows),
                "recall": sum(r["recall"] for r in rows) / n,
                "precision": sum(r["precision"] for r in rows) / n,
                "firstTry": sum(r["first_try_valid"] for r in rows),
                "fallbacks": sum(r["fallback"] for r in rows),
                "strayFilters": sum(r["unexpected_filter"] for r in rows),
                "secondsPerRequest": sum(r["seconds"] for r in rows) / n,
            },
            "rows": rows,
        })
    runs.sort(key=lambda r: r["date"] or "", reverse=True)
    return {"cases": [{"prompt": c["prompt"]} for c in cases], "runs": runs, "scorer": "strict: a filter the request did not ask for fails the case"}


class Handler(BaseHTTPRequestHandler):
    server_version = "UIPlanner/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers --------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = self.headers.get("Host") or ""
        name = host[: host.index("]") + 1] if host.startswith("[") and "]" in host else host.split(":")[0]
        return name in LOOPBACK

    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, e: ApiError):
        self._send_json({"error": e.message}, e.status)

    # -- routing --------------------------------------------------------------
    def do_GET(self):
        if not self._host_ok():
            return self._fail(ApiError(403, "Forbidden host"))
        url = urlparse(self.path)
        try:
            if url.path == "/api/health":
                q = parse_qs(url.query)
                base = clean_base_url((q.get("ollamaUrl") or [None])[0], DEFAULT_OLLAMA)
                info = list_models(base)
                return self._send_json({"ok": True, "promptVersion": PROMPT_VERSION, "dslVersion": DSL_VERSION, "ollama": {k: info[k] for k in ("reachable", "url", "version")}})
            if url.path == "/api/models":
                q = parse_qs(url.query)
                return self._send_json(list_models(clean_base_url((q.get("ollamaUrl") or [None])[0], DEFAULT_OLLAMA)))
            if url.path == "/api/datasets":
                return self._send_json(list_datasets())
            if url.path.startswith("/api/datasets/"):
                return self._send_json(load_dataset(url.path.rsplit("/", 1)[1]))
            if url.path == "/api/examples":
                cases = json.loads((DATA / "eval_cases.json").read_text(encoding="utf-8"))
                return self._send_json([c["prompt"] for c in cases])
            if url.path == "/api/eval":
                return self._send_json(load_eval())
            if url.path.startswith("/api/"):
                raise ApiError(404, "Not found")
            return self._static(url.path)
        except ApiError as e:
            self._fail(e)

    def do_POST(self):
        if not self._host_ok():
            return self._fail(ApiError(403, "Forbidden host"))
        path = urlparse(self.path).path
        if path not in ("/api/run", "/api/warm"):
            return self._fail(ApiError(404, "Not found"))
        try:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ApiError(415, "Content-Type must be application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ApiError(413 if length else 400, "Request body is empty or too large (max 2 MB)")
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                raise ApiError(400, "Request body is not valid JSON") from None
            if not isinstance(body, dict):
                raise ApiError(400, "Request body must be a JSON object")
            if path == "/api/warm":
                return self._send_json(warm_model(body))
            prompt, data, backend, retries, mode = parse_run_request(body)
        except ApiError as e:
            return self._fail(e)

        # Stream newline-delimited JSON events; the connection closes when the run ends.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(name, payload=None):
            line = json.dumps({"event": name, "data": payload}, ensure_ascii=False) + "\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        try:
            emit("start", {"backend": backend.name, "retries": retries})
            use_fast = body.get("fastPath", True) is not False and backend.name != "heuristic-baseline"
            if backend.name == "heuristic-baseline":
                mode = "plan"  # the keyword baseline writes plans, it cannot "pick"
            res = run(prompt, data, backend, retries, on_event=emit, cache=PLAN_CACHE, fast_path=use_fast,
                      aliases=load_aliases(), mode=mode)
            emit("result", run_payload(res, backend))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client cancelled
        except Exception as e:  # noqa: BLE001 - surface any pipeline bug to the UI instead of a dead stream
            try:
                emit("error", {"message": f"{type(e).__name__}: {e}"})
            except OSError:
                pass

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB / rel).resolve()
        if WEB.resolve() not in target.parents and target != WEB.resolve():
            return self._fail(ApiError(404, "Not found"))
        if not target.is_file():
            target = WEB / "index.html"  # single-page app fallback
        body = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text/", "application/javascript")) else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    srv = make_server(port)
    print(f"UI Planner running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
