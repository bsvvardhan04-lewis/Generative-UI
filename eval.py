"""Score backends/models on gold cases.

python eval.py heuristic ollama:llama3.2:3b ollama:llama3.1:8b
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cui import run
from cui.catalog import build_catalog
from cui.pipeline import PROMPT_VERSION, prefix_messages
from cui.selector import SELECT_VERSION
from run import make_backend

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def run_id(spec: str, prompt_version: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec) + "__" + prompt_version


def save_run(spec: str, rows: list[dict], prompt_version: str = PROMPT_VERSION, date: str | None = None) -> Path:
    """One file per (backend, prompt version) so later runs never overwrite earlier ones."""
    path = RESULTS / f"{run_id(spec, prompt_version)}.json"
    payload = {
        "backend": spec,
        "prompt_version": prompt_version,
        "date": date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def refs(plan: dict) -> set[str]:
    return {n.get("value") or n.get("source") for n in plan["nodes"]}


def score(case: dict, res) -> dict:
    must, may = set(case["must"]), set(case["may"])
    got = refs(res.plan)
    nodes = res.plan["nodes"]
    coll = [n for n in nodes if "source" in n]
    s = {
        "first_try_valid": not res.fallback and len(res.attempts) == 1,
        "fallback": res.fallback,
        "recall": len(must & got) / len(must),
        "precision": len(got & (must | may)) / len(got) if got else 0.0,
        "nodes": len(nodes),
        "seconds": res.total_seconds,
    }
    def is_expected_filter(n: dict) -> bool:
        f, want = n.get("filter"), case.get("filter")
        return bool(f and want and f["field"] == want["field"] and f["equals"].casefold() == want["equals"].casefold())

    checks = []
    if "type" in case:
        checks.append(any(n["type"] == case["type"] for n in coll))
    if "filter" in case:
        checks.append(any(is_expected_filter(n) for n in coll))
    if "fields_include" in case:
        checks.append(any(set(case["fields_include"]) <= set(n["fields"]) for n in coll))
    if "limit" in case:
        checks.append(any(n.get("limit") == case["limit"] for n in coll))
    if "sort_any" in case:  # list of [field, direction]; a missing sort_dir means asc
        ok = {tuple(p) for p in case["sort_any"]}
        checks.append(any((n.get("sort_by"), n.get("sort_dir", "asc")) in ok for n in coll))
    # A filter the user did not ask for silently hides data, so it fails the case.
    s["unexpected_filter"] = any(n.get("filter") and not is_expected_filter(n) for n in coll)
    checks.append(not s["unexpected_filter"])
    s["constraints_ok"] = all(checks)
    s["pass"] = not res.fallback and s["recall"] == 1 and s["precision"] == 1 and s["constraints_ok"]
    return s


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


def main() -> None:
    """python eval.py [--set dev|heldout|all] SPEC...   SPEC = [fast+][sel+]heuristic|ollama:MODEL
    `fast+` adds the router (and data/aliases.json) in front of the model, i.e. the production path.
    `sel+` uses select mode: the model only picks catalog items (see cui/selector.py)."""
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    which = "dev"
    warm = "--warm" in args  # pre-load model + prompt prefix first, as the server does when a model is selected
    if warm:
        args.remove("--warm")
    if "--set" in args:
        i = args.index("--set")
        which = args[i + 1]
        del args[i:i + 2]
    specs = args or ["heuristic"]
    data = json.loads((HERE / "data" / "sample_api.json").read_text(encoding="utf-8"))
    aliases = json.loads((HERE / "data" / "aliases.json").read_text(encoding="utf-8"))
    dev = json.loads((HERE / "data" / "eval_cases.json").read_text(encoding="utf-8"))
    held = json.loads((HERE / "data" / "heldout_cases.json").read_text(encoding="utf-8"))
    held2 = json.loads((HERE / "data" / "heldout2_cases.json").read_text(encoding="utf-8"))
    cases = {"dev": dev, "heldout": held, "heldout2": held2, "all": dev + held}[which]
    summary = []
    RESULTS.mkdir(exist_ok=True)
    for spec in specs:
        *flags, base = spec.split("+")  # e.g. fast+sel+ollama:llama3.2:3b
        fast, mode = "fast" in flags, ("select" if "sel" in flags else "plan")
        kind, _, model = base.partition(":")
        backend = make_backend(kind, model or "llama3.2:3b", None)
        if warm and hasattr(backend, "warm"):
            t = time.perf_counter()
            backend.warm(prefix_messages(build_catalog(data), mode))
            print(f"  warm-up {spec}: {time.perf_counter() - t:.1f}s (not counted below)", flush=True)
        rows = []
        for c in cases:
            res = run(c["prompt"], data, backend, fast_path=fast, aliases=aliases if (fast or mode == "select") else None, mode=mode)
            sc = score(c, res)
            sc["prompt"] = c["prompt"]
            sc["plan"] = res.plan
            sc["vague"] = bool(c.get("vague"))
            sc["source"] = "fast_path" if any(a.get("fast_path") for a in res.attempts) else "model"
            sc["errors"] = [a.get("errors") or a.get("error") for a in res.attempts]
            rows.append(sc)
            tag = "FAST" if sc["source"] == "fast_path" else "    "
            print(f"  {'PASS' if sc['pass'] else 'FAIL'} {tag} [{spec}] {c['prompt']}  ({sc['seconds']}s, {sc['nodes']} nodes{', vague' if sc['vague'] else ''})", flush=True)
        scored = [r for r in rows if not r["vague"]]  # vague prompts have no single right answer; they only test routing
        n = len(scored)
        secs = [r["seconds"] for r in rows]
        summary.append({
            "spec": spec, "n": n, "passed": sum(r["pass"] for r in scored),
            "fast": sum(r["source"] == "fast_path" for r in rows), "total": len(rows),
            "fast_wrong": sum(r["source"] == "fast_path" and not r["pass"] for r in rows),
            "p50": percentile(secs, 0.5), "p90": percentile(secs, 0.9), "max": max(secs), "mean": sum(secs) / len(secs),
        })
        version = SELECT_VERSION if mode == "select" else PROMPT_VERSION
        save_run(spec if which == "dev" else f"{spec}@{which}", rows, version)

    print(f"\n{'backend':30} {'passed':>8} {'no-model':>9} {'wrong w/o model':>16} {'p50':>7} {'p90':>7} {'max':>7} {'mean':>7}   (set: {which}, vague prompts excluded from 'passed')")
    for s in summary:
        print(f"{s['spec']:30} {s['passed']:>3}/{s['n']:<4} {s['fast']:>4}/{s['total']:<4} {s['fast_wrong']:>16} {s['p50']:>6.2f}s {s['p90']:>6.2f}s {s['max']:>6.1f}s {s['mean']:>6.2f}s")
    print(f"\nresults saved in {RESULTS.name}/ (one file per backend + prompt version)")


if __name__ == "__main__":
    main()
