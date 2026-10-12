# UI Planner: prompt + API JSON -> minimal canonical UI tree

Pure Python 3.11 (standard library only) plus a dependency-free web UI. Runs fully offline with a local model (Ollama).

```
prompt + API JSON
  -> pipeline.py   plan cache: a prompt seen before skips everything below (about 5 ms)
  -> router.py     fast path: deterministic matcher, answers clear prompts without a model (about 1 ms)
  -> catalog.py    introspect payload into a compact catalog (pointers, types, enums)
  -> schema.py     per-request JSON Schema; every pointer/field is an enum from the catalog
  -> llm.py        local model (Ollama / llama.cpp / vLLM) generates a small *view plan*
  -> validate.py   semantic checks; errors go back to the model for repair (default 2 retries, max 3)
  -> compiler.py   deterministic: plan + data -> canonical DSL (cui/1.0), values copied from the API
  -> adapters/     markdown | Adaptive Card (Teams) | html
```

The model only chooses what to show (component types, pointers, fields, filter, sort, limit). It never writes
values, so numbers cannot be hallucinated. Labels, units and formats are derived by the compiler, which also keeps
the model's output short (generation time is what dominates latency on CPU).

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
("Answered by the fast path in 3 ms", "Seen before: served from the plan cache", or "Done in 7.4 s (model 7.3 s)").
Selecting a model pre-loads it and pre-fills its prompt cache for the current data, so the first request does not pay
the model's cold start. Warnings (default view after failed validation) and errors appear above the preview.

The server binds to 127.0.0.1 only, requires `Content-Type: application/json` on POST and a loopback `Host`
header (blocks cross-site and DNS-rebinding requests), and caps request bodies at 2 MB.
It only runs the pipeline and lists models; it never downloads models.

## Latency

Three layers keep the language model out of the request path whenever possible:

| layer | when it applies | latency |
|---|---|---|
| **Plan cache** | a request seen before; word order, case, punctuation and filler words are ignored; survives restarts (`cache/plans.json`) | ~5 ms |
| **Fast-path router** (`cui/router.py`) | a clear prompt: every content word is explained by the data's own names, values, or generic words (top, oldest, table, ...) | ~1 ms |
| **Model** | everything else | seconds, set by the hardware (below) |

The router is deliberately conservative: it declines (and the model answers) when any word is unrecognized, when
several data sources match, when a count question is combined with another request, or when the prompt is vague.
**Alias search** (`data/aliases.json`) maps everyday phrases to the data's own words ("tickets" -> "cases",
"customer satisfaction" -> "csat") and is also applied to the cache key. Edit that file for your own data; it is
re-read on every request.

Measured on a CPU-only laptop (i5-12500H, 2x8 GB DDR4-3200, no GPU), `llama3.2:3b`: about 6-11 s for a request the
model has to answer, of which generation is almost all (about 5-9 tokens/s). Token generation is bound by memory
bandwidth (roughly bandwidth / model size), not by core count, so a GPU or a high-bandwidth machine (Apple Silicon,
many-channel server memory) is what shortens the model path. I have not measured other hardware; use
`python eval.py --set all fast+ollama:<model>` on the target machine.

Hybrid results, 40 prompts (12 dev + 28 held-out, 4 of them deliberately vague), same laptop, no plan cache,
model not pre-warmed (so "worst" includes a cold start):

| configuration | passed (36 non-vague) | answered without the model | wrong without the model | median | p90 | worst | mean |
|---|---|---|---|---|---|---|---|
| fast path + llama3.2:3b | 30 | 24/40 | 0 | ~2 ms | 8.3 s | 42.9 s | 3.8 s |
| fast path + llama3.1:8b | 33 | 24/40 | 0 | ~2 ms | 11.3 s | 101 s | 7.0 s |

`python eval_router.py all` measures the router alone (no model): 17/28 held-out and 7/12 dev prompts answered,
all answers correct. Caveats: the 12 dev cases guided the router's design; the 28 held-out cases were written before
the router, and its first run on them was 16/17 correct (one bug, "last 2 items" returned no sort, was then fixed;
the 17/17 above includes that fix). Coverage on other datasets will differ; the router needs the same style of
precise vocabulary the sample has.

## Command line

```bash
python -m unittest discover -s tests -t .          # 64 tests, no model needed
python run.py "show high priority cases" --render markdown          # keyword baseline, no LLM
python run.py "show high priority cases" --backend ollama --model llama3.1:8b --render plan
python run.py "show escalated cases as a table" --backend ollama --model llama3.1:8b --render card
python run.py "summarize my day" --backend ollama --model llama3.1:8b --render html --out out.html
python eval.py heuristic ollama:llama3.2:3b ollama:llama3.1:8b     # score backends on data/eval_cases.json
python eval.py --set all fast+ollama:llama3.1:8b                   # production path (router + model) on dev + held-out
python eval_router.py all                                          # router only: coverage and accuracy
```

`--render` choices: `plan` (what the model produced), `dsl` (canonical output), `markdown`, `card`, `html`.
`--backend openai --base-url http://localhost:8080/v1` targets llama.cpp `llama-server`, vLLM or LM Studio.

## Evaluation

`eval.py` scores gold cases (`data/eval_cases.json`: 12 dev cases; `data/heldout_cases.json`: 28 more, 4 vague).
A case passes only if the right data is chosen, nothing extra is added, and any filter, sort or limit the request
asked for is present. A filter the request did *not* ask for fails the case, because it silently hides data.
Each run is saved as `results/<backend>__<prompt version>.json` (`PROMPT_VERSION` in `cui/pipeline.py`) so runs never
overwrite each other.

Model only, 12 dev cases, CPU only, strict scorer:

| backend | prompt | passed | recall | precision | stray filters | s/request |
|---|---|---|---|---|---|---|
| llama3.1:8b | v2 | 11/12 | 1.00 | 1.00 | 0 | 29.4 |
| llama3.2:3b | v3 | 8/12 | 0.92 | 0.92 | not recorded | 10.8 |
| llama3.2:3b | v2 | 8/12 | 0.83 | 0.86 | 1 | 14.2 |
| llama3.2:3b | v1 | 8/12 | 0.92 | 0.89 | 3 | 22.4 |
| keyword baseline | n/a | 6/12 | 0.88 | 0.72 | 0 | <0.1 |
| llama3.2:1b | v1 | 3/12 | 0.58 | 0.44 | 2 | 34.1 |

Prompt v3 removed labels, intent and format from the model's output (the compiler derives them): same accuracy
for the 3B model, about 25% faster. The 8B model was not re-run on v3 alone (only inside the hybrid above).
With only 12 dev cases, a one-case difference is within noise. Dates for the earliest runs are approximate.

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
* Layout is a flat ordered list; nested groups would need a parent/id scheme.
* Collections are detected only as arrays of objects; arrays of scalars are ignored.
* Tone mapping (High -> critical, ...) is a small built-in dictionary in `compiler.py`.
* The router is English-only and rule-based; it declines rather than guesses, so coverage depends on how closely
  prompts use the data's own vocabulary (aliases help).
* The Teams preview is a lightweight renderer for the elements our adapter emits (text, container, column set, table);
  confirm real appearance in the Adaptive Cards designer or Teams. The UI is tested in Chromium only.
