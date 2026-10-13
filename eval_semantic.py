"""Calibrate and measure the L4 semantic level (needs Ollama with the embedding model, default nomic-embed-text).

python eval_semantic.py [--model nomic-embed-text] [--threshold T --margin M]

1. Calibration: on the prompts the router declines in the dev / held-out / held-out 2 / paraphrase sets,
   sweep (threshold, margin) and show coverage vs. wrong answers.
2. Check: run the full fast chain (rewrite -> router -> semantic) on data/semantic_cases.json, which was
   written before cui/semantic.py and is not used for calibration, at the thresholds in cui/semantic.py.
A semantic answer counts as wrong if it fails the strict scorer, or if the prompt is vague.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cui.catalog import build_catalog
from cui.rewrite import rewrite
from cui.router import route
from cui.selector import picks_to_plan, select_options
from cui.semantic import DEFAULT_EMBED_MODEL, DEFAULT_MARGIN, DEFAULT_THRESHOLD, OllamaEmbedder, SemanticIndex
from eval import score

HERE = Path(__file__).parent
CALIBRATION = ["eval_cases.json", "heldout_cases.json", "heldout2_cases.json", "paraphrase_cases.json"]


def load(name):
    return json.loads((HERE / "data" / name).read_text(encoding="utf-8"))


def correct(case, plan) -> bool:
    if case.get("vague"):
        return False
    return score(case, SimpleNamespace(plan=plan, fallback=False, attempts=[{}], total_seconds=0))["pass"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = ap.parse_args()
    cat = build_catalog(load("sample_api.json"))
    aliases = load("aliases.json")
    index = SemanticIndex(OllamaEmbedder(args.model), args.threshold, args.margin)
    print(f"embedding model: {args.model}")
    opts = select_options(cat)
    index.warm(cat)

    rows = []  # (score, margin, would_be_correct, prompt, best)
    for f in CALIBRATION:
        for c in load(f):
            q = rewrite(c["prompt"], cat, aliases).text
            if route(q, cat).plan is not None:
                continue  # already answered without the model
            ranked = index.rank(q, cat)
            (s1, _, p1), (s2, _, _) = ranked[0], ranked[1]
            plan, _ = picks_to_plan({"pick": [p1]}, q, cat, opts)
            rows.append((s1, s1 - s2, correct(c, plan), c["prompt"], p1))

    print(f"calibration: {len(rows)} prompts the router declines")
    for s, m, ok, p, best in sorted(rows, reverse=True):
        print(f"  {'ok   ' if ok else 'WRONG'} score {s:.3f} gap {m:.3f}  {p[:60]!r} -> {best}")
    print("\n  threshold  margin   answered  wrong")
    for t in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        for m in (0.0, 0.03, 0.06, 0.1):
            ans = [r for r in rows if r[0] >= t and r[1] >= m]
            print(f"  {t:9.2f} {m:7.2f} {len(ans):9d} {sum(not r[2] for r in ans):6d}")

    print(f"\ncheck on semantic_cases.json (never used for calibration), threshold {args.threshold}, margin {args.margin}:")
    stats = {"router": 0, "semantic": 0, "semantic_wrong": 0, "model": 0}
    for c in load("semantic_cases.json"):
        q = rewrite(c["prompt"], cat, aliases).text
        if route(q, cat).plan is not None:
            stats["router"] += 1
            print(f"  router    {c['prompt']!r}")
            continue
        t0 = time.perf_counter()
        m = index.match(q, cat)
        ms = 1000 * (time.perf_counter() - t0)
        if m.plan is None:
            stats["model"] += 1
            print(f"  -> model  {c['prompt']!r} ({m.reason}; {ms:.0f} ms)")
        else:
            ok = correct(c, m.plan)
            stats["semantic"] += 1
            stats["semantic_wrong"] += not ok
            print(f"  {'semantic' if ok else 'WRONG   '}  {c['prompt']!r} -> {m.best} (score {m.score}, gap {m.margin}; {ms:.0f} ms)")
    print(f"\n  {stats}")


if __name__ == "__main__":
    main()
