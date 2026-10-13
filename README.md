# UI Planner: prompt + API JSON -> minimal canonical UI tree

Pure Python 3.11 (standard library only) plus a dependency-free web UI. Runs fully offline with a local model (Ollama).

```
prompt + API JSON
  -> rewrite.py    query rewrite, level by level (~1 ms): L0 normalize, L1 spelling, L2 concepts, L3 aliases
  -> pipeline.py   plan cache: a prompt seen before skips everything below (about 5-20 ms)
  -> router.py     fast path: deterministic matcher, answers clear prompts without a model (about 1-10 ms)
  -> semantic.py   L4 meaning match with a small embedding model (off by default, see "Semantic level")
  -> catalog.py    introspect payload into a compact catalog (pointers, types, enums)
  -> selector.py   select mode (default): the model only *picks* catalog paths, {"pick":["/team"]};
                   code adds filter / sort / limit / columns from the prompt's own words
     schema.py     plan mode: the model writes the whole view plan (more tokens, slower)
  -> llm.py        local model (Ollama / llama.cpp / vLLM), output constrained by a per-request JSON Schema
  -> validate.py   semantic checks; errors go back to the model for repair (default 2 retries, max 3)
  -> compiler.py   deterministic: plan + data -> canonical DSL (cui/1.0), values copied from the API
  -> adapters/     HXL widget JSON | markdown | Adaptive Card (Teams) | html
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
* **Structure** (right): the generated **HXL widget JSON** (same format as the Salesforce HXL playground) with line
  numbers and a Copy button. The **Data** tab holds the input API response (editable; paste your own, or Reset to sample).

A status line shows what the pipeline is doing, then how the answer was produced and how long it took
("Answered by the fast path in 3 ms", "Seen before: served from the plan cache", or "Done in 3.4 s (model 3.3 s)").
Selecting a model pre-loads it and pre-fills its prompt cache for the current data, so the first request does not pay
the model's cold start. Warnings (default view after failed validation) and errors appear above the preview.

The server binds to 127.0.0.1 only, requires `Content-Type: application/json` on POST and a loopback `Host`
header (blocks cross-site and DNS-rebinding requests), and caps request bodies at 2 MB. It only runs the pipeline
and lists models; it never downloads models. `POST /api/run` accepts `"mode": "select"` (default) or `"plan"`.

## Latency

The idea: answer each request at the cheapest level that can answer it correctly, and make the cheap levels
cover as many requests as possible.

| level | cost | answers |
|---|---|---|
| query rewrite | ~1 ms, always | nothing by itself; turns wording into the data's vocabulary so the levels below match |
| plan cache | ~5-20 ms | anything asked before, in any wording that rewrites to the same words |
| fast-path router | ~1-10 ms | clear requests: every word explained by the data |
| semantic match (L4, off by default) | ~40-75 ms | strong, unambiguous meaning matches; see below |
| model (select mode) | ~1.4 s median with llama3.2:3b, ~3.1 s with llama3.1:8b on this laptop | everything else |

**Connection fix (the largest single win):** on Windows, `localhost` resolves to the IPv6 address `::1` first,
while Ollama listens on IPv4 only, so every model call waited about 2 s before falling back (measured: 2,070 ms
through `localhost` vs 16-57 ms through `127.0.0.1` for the same request). All defaults now use `127.0.0.1`, and
`cui/llm.py prefer_ipv4()` rewrites a user-supplied `localhost` URL. On the 17-prompt set below, select mode went
from a 3.5 s to a 1.4 s median with llama3.2:3b (15/17 passed), and from 4.7 s to 3.1 s with llama3.1:8b (16/17);
those runs also include the query rewrite and new aliases, which answer 6 of the 17 without the model.

### Semantic level (L4, `cui/semantic.py`), off by default

Each value and list is described in a sentence (its name, unit meaning, fields, enum values and generic synonyms),
embedded once per data shape with a local embedding model (`ollama pull nomic-embed-text`, 274 MB; the
`search_query:` / `search_document:` prefixes it was trained with are added automatically), and the request is
matched by cosine similarity; it answers only on a strong match that is clearly ahead of the runner-up.

Measured with `python eval_semantic.py --model <name>`: calibrated on the 28 prompts the router declines in the
dev / held-out / paraphrase sets, then checked on a fresh meaning-only set (`data/semantic_cases.json`, 18 prompts
written before the module and never used for calibration).

| embedding model | setting (score / gap) | calibration: answered, wrong | fresh set: answered, wrong | per request |
|---|---|---|---|---|
| all-minilm (45 MB) | 0.60 / 0.03 | 1 of 28, 0 | 0 of 18, 0 | ~15-60 ms |
| all-minilm | 0.45 / 0.06 | 10 of 28, 1 | 2 of 18, 1 | |
| **nomic-embed-text** (default) | **0.55 / 0.10** | 4 of 28, 0 | 1 of 18, 0 | ~40-75 ms |
| nomic-embed-text | 0.60 / 0.03 | 13 of 28, 2 | 2 of 18, 0 | |

all-minilm scores correct matches (0.23-0.49) in the same range as "tell me a joke", so it cannot be trusted.
nomic-embed-text keeps unrelated requests below 0.50 and correct ones up to 0.74, but neighbouring options often
tie (average vs. oldest wait, resolved vs. reopened), so a safe setting needs a 0.10 gap and then answers few
requests. It is never wrong there, but it saves the model call on only ~5-15% of declined requests while adding
~50 ms to all of them, so the server keeps it off. Turn it on with `python server.py --semantic` or
`"semantic": true` per request.

### Query rewrite (`cui/rewrite.py`)

Each level only ever rewrites a word into the data's own vocabulary:

* **L0 normalize**: lower case, contractions ("who's" -> "who is"), punctuation.
* **L1 spelling**: a 6+ letter word that is not in the vocabulary -> the single closest data, alias or generic word
  (edit distance 1, or 2 for 8+ letters; same first letter; unique best). "escalted" -> "escalated",
  "tikets" -> "tickets", "priorty" -> "priority". Short words are never corrected ("stats" stays "stats").
* **L2 concepts**: generic synonym groups, data-aware: "client" -> "customer" only if the data uses "customer" and
  not "client"; "solved" -> "resolved". Nothing is invented on data that lacks the target word.
* **L3 aliases**: `data/aliases.json` phrases.

The rewritten text drives the router, the code that adds filters/sort/limit/columns, and the cache key; the model
still sees the user's original wording. The UI shows it: *Understood as "show me escalated cases as a table with
customer name" (escalted -> escalated, tikets -> tickets, clients -> customer, tickets -> cases)*.

Router coverage, before and after query rewrite (no model involved; `python eval_router.py all [--no-rewrite]`):

| set | without rewrite | with rewrite |
|---|---|---|
| dev (12) | 7 answered, 7 correct | 7 answered, 7 correct |
| held-out (28) | 19 answered, 19 correct | 19 answered, 19 correct |
| held-out 2 (17) | 6 answered, 6 correct | 6 answered, 6 correct |
| **paraphrase (20: typos, synonyms, contractions)** | **5 answered, 5 correct** | **17 answered, 17 correct** |

The paraphrase set was written before the rewrite code. Its first run surfaced four general bugs (plural filler words
being remapped, a repeated word after an alias, synonyms applied after aliases, a filtered-field word not next to its
value), which were fixed before the numbers above; it is therefore no longer a clean held-out set.

### Measures that cut model time

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
"Escalated", so no filter is added) and "Customer satisfaction for the week across the team". After this
measurement, an `escalations -> escalated cases` alias was added, so the router now answers the first one directly
(correctly, in under a millisecond); the table above predates that alias.

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
python -m unittest discover -s tests -t .          # 111 tests, no model needed
python run.py "show high priority cases" --render markdown          # keyword baseline, no LLM
python run.py "show high priority cases" --backend ollama --model llama3.1:8b --render plan
python run.py "show escalated cases as a table" --backend ollama --model llama3.1:8b --render card
python run.py "summarize my day" --backend ollama --model llama3.1:8b --render html --out out.html
python eval.py heuristic ollama:llama3.2:3b ollama:llama3.1:8b     # model only, plan mode, dev cases
python eval.py --warm --set heldout2 fast+sel+ollama:llama3.1:8b   # production path: router + select mode
python eval_router.py all [--no-rewrite]                           # router only: coverage and accuracy
python eval_semantic.py [--model all-minilm]                       # calibrate / check the semantic level (needs nomic-embed-text)
python server.py --semantic                                        # web UI with the semantic level switched on
```

`eval.py` spec flags: `fast+` puts the router (and aliases) in front, `sel+` uses select mode, `sem+` adds the
semantic level; `--set` is
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

## HXL widget output

`cui/adapters/hxl.py` emits the format used by the HXL playground (checked against its component docs and
tutorials): a `tile/widget` whose `dataProviders` holds the data the view needs as an `inline-data` provider named
`$data`, and `children` built from `tile/container`, `tile/column`, `tile/row`, `tile/text`, `tile/badge` and
`tile/table`, using only documented attributes and values.

```json
{
  "definition": "tile/widget",
  "dataProviders": [
    { "definition": "inline-data", "expressionKey": "$data", "inline": true,
      "value": { "queue": { "open": 18 }, "cases": [ { "id": "C-10482", "subject": "...", "status": "Escalated" } ] } }
  ],
  "children": [
    { "definition": "tile/container", "children": [
      { "definition": "tile/column", "attributes": { "gap": "lg" }, "children": [
        { "definition": "tile/text", "attributes": { "text": "{!$data.queue.open}", "variant": "h2" } },
        { "definition": "tile/container",
          "meta": { "forEach": "{!$data.cases}", "forItem": "$item", "if": "{!$item.status = 'Escalated'}" },
          "children": [ { "definition": "tile/text", "attributes": { "text": "{!$item.subject}" } } ] }
      ] }
    ] }
  ]
}
```

* Values are **bound** with `{!$data.path}` expressions where HXL can compute the view itself: single values, and
  lists whose only operation is a filter (`meta.forEach` + `meta.if`).
* Sorted or limited lists, and all tables (`tile/table` with `caption`, `columns`, `rows`, `sort`), are written with
  **literal values**, because HXL expressions have no sort or slice.
* `dataProviders` holds only what the view uses (referenced values; lists reduced to the shown, filtered and sorted
  fields), so the output stays minimal.
* Data that itself contains `{!` is marked `ignoreExpressions`, and keys that are not plain identifiers fall back to
  literal values.
* Not verified inside a live Salesforce org; the format follows the playground's documentation and examples.

## Canonical DSL (cui/1.0)

The internal, renderer-neutral tree that every adapter (HXL, Teams card, HTML, Markdown) is generated from.

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
