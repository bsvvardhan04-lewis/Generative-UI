"""CLI: python run.py "show high priority cases" --backend ollama --model llama3.2:3b"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cui import run
from cui.adapters import RENDERERS
from cui.llm import HeuristicBackend, OllamaBackend, OpenAICompatBackend

HERE = Path(__file__).parent


def make_backend(kind: str, model: str, base_url: str | None):
    if kind == "heuristic":
        return HeuristicBackend()
    if kind == "ollama":
        return OllamaBackend(model, base_url or "http://localhost:11434")
    if kind == "openai":
        return OpenAICompatBackend(model, base_url or "http://localhost:8080/v1")
    raise SystemExit(f"unknown backend {kind}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--data", default=str(HERE / "data" / "sample_api.json"))
    ap.add_argument("--backend", choices=["heuristic", "ollama", "openai"], default="heuristic")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--base-url")
    ap.add_argument("--render", choices=["plan", "dsl", *RENDERERS], default="markdown")
    ap.add_argument("--out", help="write the rendered output to this file instead of stdout")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    backend = make_backend(a.backend, a.model, a.base_url)
    res = run(a.prompt, data, backend, a.retries)

    if a.render == "plan":
        out = json.dumps(res.plan, indent=2)
    elif a.render == "dsl":
        out = json.dumps(res.dsl, indent=2, ensure_ascii=False)
    else:
        r = RENDERERS[a.render](res.dsl)
        out = r if isinstance(r, str) else json.dumps(r, indent=2, ensure_ascii=False)

    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
    else:
        print(out)
    toks = [x.get("output_tokens") for x in res.attempts if x.get("output_tokens") is not None]
    print(
        f"\n[{backend.name}] attempts={len(res.attempts)} fallback={res.fallback} nodes={len(res.plan['nodes'])} "
        f"catalog={res.catalog_chars} chars, plan tokens={toks} total={res.total_seconds}s",
        file=sys.stderr,
    )
    if res.errors:
        print("validation errors:", *res.errors, sep="\n  - ", file=sys.stderr)


if __name__ == "__main__":
    main()
