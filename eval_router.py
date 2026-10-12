"""Measure the fast-path router on its own (no model): coverage and accuracy.

python eval_router.py            # dev cases (data/eval_cases.json) and held-out cases (data/heldout_cases.json)

* answered   : the router produced a plan itself
* correct    : the plan passes the same strict scorer used for model evaluations
* wrong      : answered but incorrect, the costly case, since it skips the model
* declined   : handed to the model (not an error, only slower)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cui.catalog import build_catalog
from cui.router import route
from eval import score

HERE = Path(__file__).parent


def load(name):
    return json.loads((HERE / "data" / name).read_text(encoding="utf-8"))


def evaluate(cases, cat, aliases, verbose=True):
    answered = correct = 0
    wrong, declined, seconds = [], [], 0.0
    for c in cases:
        t = time.perf_counter()
        r = route(c["prompt"], cat, aliases)
        seconds += time.perf_counter() - t
        if r.plan is None:
            declined.append((c["prompt"], r.reason))
            continue
        answered += 1
        if c.get("vague"):
            wrong.append((c["prompt"], "answered a vague prompt", r.plan["nodes"]))
            continue
        s = score(c, SimpleNamespace(plan=r.plan, fallback=False, attempts=[{}], total_seconds=0))
        if s["pass"]:
            correct += 1
        else:
            wrong.append((c["prompt"], f"recall {s['recall']:.2f} precision {s['precision']:.2f} constraints_ok={s['constraints_ok']}", r.plan["nodes"]))
    n = len(cases)
    print(f"  answered {answered}/{n} ({answered / n:.0%})   correct {correct}/{answered}   wrong {len(wrong)}   declined {len(declined)}   avg {1000 * seconds / n:.2f} ms/prompt")
    if verbose:
        for p, why, nodes in wrong:
            print(f"    WRONG    {p!r}: {why}\n             {json.dumps(nodes)}")
        for p, why in declined:
            print(f"    declined {p!r}: {why}")
    return answered, correct, wrong, declined


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cat = build_catalog(load("sample_api.json"))
    aliases = load("aliases.json")
    for label, file in (("dev (12 cases used while building the router)", "eval_cases.json"), ("held-out (written before the router)", "heldout_cases.json")):
        if len(sys.argv) > 1 and sys.argv[1] not in file and sys.argv[1] != "all":
            continue
        print(label)
        evaluate(load(file), cat, aliases)


if __name__ == "__main__":
    main()
