# UI Planner: prompt + API JSON -> minimal canonical UI tree

Pure Python 3.11 (standard library only) plus a dependency-free web UI. Runs fully offline with a local model (Ollama).

```
prompt + API JSON
  -> pipeline.py   plan cache: a prompt seen before skips everything below (about 5-20 ms)
  -> router.py     fast path: deterministic matcher, answers clear prompts without a model (about 1-10 ms)
  -> catalog.py    introspect payload into a compact catalog (pointers, types, enums)
  -> selector.py   select mode (default): the model only *picks* catalog paths, {"pick":["/team"]};
                   code adds filter / sort / limit / columns from the prompt's own words
     schema.py     plan mode: the model writes the whole view plan (more tokens, slower)
  -> llm.py        local model (Ollama / llama.cpp / vLLM), output constrained by a per-request JSON Schema
  -> validate.py   semantic checks; errors go back to the model for repair (default 2 retries, max 3)
  -> compiler.py   deterministic: plan + data -> canonical DSL (cui/1.0), values copied from the API
  -> adapters/     markdown | Adaptive Card (Teams) | html
```

The model only chooses what to show. It never writes values, so numbers cannot be hallucinated. Labels, units
and formats are derived by the compiler, which also keeps the model's output short: on a CPU, generation time is
roughly proportional to the number of tokens the model writes.

## Web UI

```bash
python server.py            # http://127.0.0.1:8765
```

One page, laid out like the HXL playground:

* **Prompt** bar on top, with a model picker (installed Ollama models, or the keyword baseline) and **Generate**.
* **Preview** (left): the result rendered as a card. Switch between **Teams** (Adaptive Card 1.5, drawn visually),
  **Web** (HTML) and **Chat** (Markdown). Each has a Copy button for its raw output.
* **Structure** (right): the generated canonical UI JSON with line numbers. The **Data** tab holds the input API
  response (editable; paste your own, or Reset to sample).

A status line shows what the pipeline is doing, then how the answer was produced and how long it took
("Answered by the fast path in 3 ms", "Seen before: served from the plan cache", or "Done in 3.4 s (model 3.3 s)").
Selecting a model pre-loads it and pre-fills its prompt cache for the current data, so the first request does not pay
the model's cold start. Warnings (default view after failed validation) and errors appear above the preview.

The server binds to 127.0.0.1 only, requires `Content-Type: application/json` on POST and a loopback `Host`
header (blocks cross-site and DNS-rebinding requests), and caps request bodies at 2 MB. It only runs the pipeline
and lists models; it never downloads models. `POST /api/run` accepts `"mode": "select"` (default) or `"plan"`.

## Latency

Four code-level measures keep the language model out of the request path, or make it short:

| measure | when it applies | latency |
|---|---|---|
| **Plan cache** | a request seen before; word order, case, punctuation and filler words are ignored; survives restarts (`cache/plans.json`) | ~5-20 ms |
| **Fast-path router** (`cui/router.py`) | a clear prompt: every content word is explained by the data's own names, values, or generic words (top, oldest, table, ...) | ~1-10 ms |
| **Select mode** (`cui/selector.py`) | everything else: the model picks paths (~12 generated tokens instead of 32-65) | ~3-5 s on this laptop |
| **Warm-up** | selecting a model loads it and pre-fills the prompt prefix | removes a 20-70 s first-request cost |

The router is deliberately conservative: it declines (and the model answers) when any word is unrecognized, when
several data sources match, when a count question is combined with another request, or when the prompt is vague.
**Alias search** (`data/aliases.json`) maps everyday phrases to the data's own words ("tickets" -> "cases",
"customer satisfaction" -> "csat") and is also applied to the cache key. Edit that file for your own data; it is
re-read on every request. In select mode, extra picks that no word of the request points to are dropped.

Measured on a CPU-only laptop (i5-12500H, 2x8 GB DDR4-3200, no GPU). Token generation is bound by memory bandwidth
(roughly bandwidth / model size), not by core count: of 4/6/8/12/16 threads, 12 was fastest (10.8 tokens/s with the
3B model) and more threads were slower, so the code leaves Ollama's thread setting alone.
A GPU or a high-bandwidth machine (Apple Silicon, many-channel server memory) shortens the model path further.
I have not measured other hardware; run `python eval.py --warm --set heldout2 fast+sel+ollama:<model>` on the target.

**Select vs plan mode**, fresh held-out set (`data/heldout2_cases.json`, 17 prompts written after the other sets,
before the select-mode fixes; the router answers 4 of them), models pre-warmed:

| configuration | passed | median | p90 | mean |
|---|---|---|---|---|
| router + select + llama3.1:8b | **15/17** | 4.7 s | 5.7 s | 3.9 s |
| router + plan + llama3.1:8b | 14/17 | 7.4 s | 8.9 s | 6.3 s |
| router + select + llama3.2:3b | **13/17** | 3.5 s | 4.1 s | 2.8 s |
| router + plan + llama3.2:3b | 13/17 | 6.0 s | 8.6 s | 5.5 s |
| router + select (s1, picked by number) + llama3.2:3b | 9/17 | 3.1 s | 3.9 s | 2.7 s |

Select mode first picked catalog items by number (s1); small models mapped words to the wrong number, so s2 picks
by path. Both remaining 8B misses: "How many escalations are there?" ("escalations" does not match the value
"Escalated", so no filter is added) and "Customer satisfaction for the week across the team".

**Router + model on the 40 dev/held-out prompts** (`--set all`; 4 vague prompts excluded from "passed"):

| configuration | passed | answered without the model | wrong without the model | p50 | p90 |
|---|---|---|---|---|---|
| router + plan + llama3.1:8b (not warmed) | 33/36 | 24/40 | 0 | ~2 ms | 11.3 s |
| router + plan + llama3.2:3b (warmed) | 30/36 | 24/40 | 0 | ~2 ms | 8.2 s |
| router + select s1 + llama3.1:8b (warmed) | 29/36 | 24/40 | 0 | ~2 ms | 5.2 s |
| router + select s1 + llama3.2:3b (warmed) | 28/36 | 24/40 | 0 | ~2 ms | 3.0 s |

These 40 prompts were used to find the s1 failures, so select s2 was evaluated on the fresh set above instead.
`python eval_router.py all` measures the router alone: 7/12 dev and 17/28 held-out prompts answered, all correct
(its first held-out run was 16/17; the one bug, "last 2 items" returning no sort, was then fixed). Coverage on other
datasets will differ; the router needs the same style of precise vocabulary the sample has.

## Command line

```bash
python -m unittest discover -s tests -t .          # 79 tests, no model needed
python run.py "show high priority cases" --render markdown          # keyword baseline, no LLM
python run.py "show high priority cases" --backend ollama --model llama3.1:8b --render plan
python run.py "show escalated cases as a table" --backend ollama --model llama3.1:8b --render card
python run.py "summarize my day" --backend ollama --model llama3.1:8b --render html --out out.html
python eval.py heuristic ollama:llama3.2:3b ollama:llama3.1:8b     # model only, plan mode, dev cases
python eval.py --warm --set heldout2 fast+sel+ollama:llama3.1:8b   # production path: router + select mode
python eval_router.py all                                          # router only: coverage and accuracy
```

`eval.py` spec flags: `fast+` puts the router (and aliases) in front, `sel+` uses select mode; `--set` is
`dev`, `heldout`, `heldout2` or `all`; `--warm` pre-loads the model before timing. `run.py` uses plan mode.
`--render` choices: `plan` (what the model produced), `dsl` (canonical output), `markdown`, `card`, `html`.
`--backend openai --base-url http://localhost:8080/v1` targets llama.cpp `llama-server`, vLLM or LM Studio.

## Evaluation

Gold cases: `data/eval_cases.json` (12 dev), `data/heldout_cases.json` (28, 4 vague), `data/heldout2_cases.json`
(17, fresh). A case passes only if the right data is chosen, nothing extra is added, and any filter, sort or limit
the request asked for is present. A filter the request did *not* ask for fails the case, because it silently hides
data. Each run is saved as `results/<spec>__<version>.json` (plan mode: `PROMPT_VERSION` in `cui/pipeline.py`;
select mode: `SELECT_VERSION` in `cui/selector.py`) so runs never overwrite each other.

Model only, plan mode, 12 dev cases, strict scorer:

| backend | prompt | passed | recall | precision | stray filters | s/request |
|---|---|---|---|---|---|---|
| llama3.1:8b | v2 | 11/12 | 1.00 | 1.00 | 0 | 29.4 |
| llama3.2:3b | v3 | 8/12 | 0.92 | 0.92 | not recorded | 10.8 |
| llama3.2:3b | v2 | 8/12 | 0.83 | 0.86 | 1 | 14.2 |
| llama3.2:3b | v1 | 8/12 | 0.92 | 0.89 | 3 | 22.4 |
| keyword baseline | n/a | 6/12 | 0.88 | 0.72 | 0 | <0.1 |
| llama3.2:1b | v1 | 3/12 | 0.58 | 0.44 | 2 | 34.1 |

Prompt v3 removed labels, intent and format from the model's output (the compiler derives them): same accuracy
for the 3B model, about 25% faster. With 12-17 cases per set, a one-case difference is within noise.
Dates for the earliest runs are approximate.

## Canonical DSL (cui/1.0)

Semantic, platform-neutral. Containers: `stack`, `row` (consecutive metrics are grouped).
Leaves: `metric`, `text`, `status` (with `tone`), `list`, `table`. Every value carries the
raw `value`, a formatted `display`, and a `source` JSON pointer for traceability.

## Extending

* New component: add a variant in `schema.py`, rules in `validate.py`, output in `compiler.py`, and a case per adapter.
* New renderer: write `render(dsl)` in `adapters/` and register it in `adapters/__init__.py`. Adapters should degrade (table -> list) when a platform lacks a widget.
* Different API: paste it into the UI, or add a JSON file to `data/`. Arrays of objects become collections; enums are auto-detected. Add phrases to `data/aliases.json` for its vocabulary.

## Known limits

* Filters are equality on low-cardinality string fields only (one per list); no numeric ranges or free-text search yet.
* In select mode, filters/sort/limit/columns come from the prompt's literal words; a value phrased differently
  ("escalations" for "Escalated") is missed unless an alias maps it. Plan mode lets the model set them instead.
* Layout is a flat ordered list; nested groups would need a parent/id scheme.
* Collections are detected only as arrays of objects; arrays of scalars are ignored.
* Tone mapping (High -> critical, ...) is a small built-in dictionary in `compiler.py`.
* The router is English-only and rule-based; it declines rather than guesses, so coverage depends on how closely
  prompts use the data's own vocabulary (aliases help).
* The Teams preview is a lightweight renderer for the elements our adapter emits (text, container, column set, table);
  confirm real appearance in the Adaptive Cards designer or Teams. The UI is tested in Chromium only.
