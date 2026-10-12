import json
import tempfile
import unittest
from pathlib import Path

from cui import run
from cui.catalog import build_catalog
from cui.pipeline import PlanCache, normalize_prompt
from cui.router import apply_aliases, route, stem

ROOT = Path(__file__).parent.parent
DATA = json.loads((ROOT / "data" / "sample_api.json").read_text(encoding="utf-8"))
ALIASES = json.loads((ROOT / "data" / "aliases.json").read_text(encoding="utf-8"))
CAT = build_catalog(DATA)


def plan_for(prompt, aliases=None):
    r = route(prompt, CAT, aliases)
    return r.plan["nodes"] if r.plan else None


class Counting:
    name = "counting"

    def __init__(self, plan=None):
        self.plan, self.calls = plan or {"nodes": [{"type": "metric", "value": "/queue/open"}]}, 0

    def generate(self, messages, schema):
        self.calls += 1
        return json.dumps(self.plan), {}


class RouterAnswerTests(unittest.TestCase):
    def test_single_values(self):
        self.assertEqual(plan_for("Any SLA breaches?"), [{"type": "metric", "value": "/queue/slaBreaches"}])
        self.assertEqual(plan_for("What's my CSAT today?"), [{"type": "metric", "value": "/today/csat"}])
        self.assertEqual(plan_for("What is my current status?")[0]["value"], "/agent/status")

    def test_count_question_prefers_the_number_over_a_list(self):
        self.assertEqual(plan_for("How many open cases are in the queue?"), [{"type": "metric", "value": "/queue/open"}])

    def test_filter_and_columns(self):
        node = plan_for("Show escalated cases as a table with customer name and priority")[0]
        self.assertEqual((node["type"], node["source"], node["fields"]), ("table", "/cases", ["customer.name", "priority"]))
        self.assertEqual(node["filter"], {"field": "status", "equals": "Escalated"})

    def test_filter_word_naming_the_field_is_not_a_column(self):
        node = plan_for("Show me the high priority cases")[0]
        self.assertEqual(node["filter"], {"field": "priority", "equals": "High"})
        self.assertGreater(len(node["fields"]), 1)  # default columns, not just "priority"

    def test_limit_and_sort(self):
        node = plan_for("Show the 3 oldest cases")[0]
        self.assertEqual((node["limit"], node["sort_by"], node["sort_dir"]), (3, "ageHours", "desc"))
        node = plan_for("Latest 3 activity updates")[0]
        self.assertEqual((node["source"], node["limit"], node["sort_by"], node["sort_dir"]), ("/recentActivity", 3, "at", "desc"))

    def test_last_n_means_most_recent_n(self):
        node = plan_for("Show the last 2 activity updates")[0]
        self.assertEqual((node["limit"], node["sort_by"], node["sort_dir"]), (2, "at", "desc"))

    def test_sort_by_a_named_numeric_field(self):
        node = plan_for("Team members with the highest weekly CSAT")
        self.assertIsNone(node)  # "members" is not a known word: declines rather than guessing
        node = plan_for("Team with the highest weekly CSAT")[0]
        self.assertEqual((node["source"], node["sort_by"], node["sort_dir"]), ("/team", "csatWeek", "desc"))

    def test_plans_always_validate(self):
        from cui.validate import validate_plan
        for p in ["List all the cases", "Show Medium priority cases", "Table of cases with id, subject and priority", "Any SLA breaches?"]:
            self.assertEqual(validate_plan(route(p, CAT).plan, CAT), [], p)


class RouterDeclineTests(unittest.TestCase):
    """When in doubt the router must hand over to the model."""

    def test_vague_or_unrelated_prompts(self):
        for p in ["Summarize my day", "How are we doing?", "Give me an overview", "What's the weather like?", "What's new?", ""]:
            self.assertIsNone(plan_for(p), p)

    def test_ambiguous_data_source(self):
        self.assertIsNone(plan_for("Notes on cases"))  # "cases" names one source, "Note" is a value of another
        self.assertIsNone(plan_for("Who is on the team and how many open cases does each person have?"))

    def test_value_plus_list_without_a_clear_count(self):
        self.assertIsNone(plan_for("Which Platinum customers do we have open cases for?"))

    def test_two_requests_in_one(self):
        self.assertIsNone(plan_for("How many cases are in the queue and what's the average wait?"))

    def test_unknown_word_declines(self):
        self.assertIsNone(plan_for("Show cases with the weird flux capacitor"))
        self.assertIsNone(plan_for("Show cases from the last 7 days"))  # a number the router cannot honor

    def test_a_common_word_is_not_a_filter_without_the_collection_name(self):
        self.assertIsNone(plan_for("What is low?"))
        self.assertEqual(plan_for("Show low priority cases")[0]["filter"], {"field": "priority", "equals": "Low"})

    def test_two_filters_decline(self):
        self.assertIsNone(plan_for("Show high priority escalated cases"))


class AliasTests(unittest.TestCase):
    def test_alias_rewrites_everyday_words(self):
        self.assertEqual(apply_aliases("Show my tickets", {"tickets": "cases"}).split(), ["show", "my", "cases"])
        self.assertIsNone(plan_for("List all the tickets"))  # without the alias the word is unknown
        self.assertEqual(plan_for("List all the tickets", ALIASES)[0]["source"], "/cases")
        self.assertEqual(plan_for("What is the customer satisfaction?", ALIASES), [{"type": "metric", "value": "/today/csat"}])

    def test_longest_alias_wins_and_whole_words_only(self):
        out = apply_aliases("service level breaches", {"service level breaches": "sla breaches", "breaches": "x"})
        self.assertEqual(out.split(), ["sla", "breaches"])
        self.assertEqual(apply_aliases("tickets123", {"tickets": "cases"}).strip(), "tickets123")

    def test_stemming_is_symmetric(self):
        self.assertEqual(stem("breaches"), stem("breach"))
        self.assertEqual(stem("waiting"), stem("wait"))
        self.assertEqual(stem("resolved"), stem("resolve"))


class FastPathPipelineTests(unittest.TestCase):
    def test_confident_prompts_never_touch_the_model(self):
        backend = Counting()
        res = run("Any SLA breaches?", DATA, backend, fast_path=True)
        self.assertEqual(backend.calls, 0)
        self.assertTrue(res.attempts[0]["fast_path"])
        self.assertEqual(res.dsl["root"]["children"][0]["value"], 2)

    def test_declined_prompts_fall_back_to_the_model(self):
        backend, events = Counting(), []
        res = run("Summarize my day", DATA, backend, fast_path=True, on_event=lambda n, d=None: events.append(n))
        self.assertEqual(backend.calls, 1)
        self.assertIn("fast_path_declined", events)
        self.assertNotIn("fast_path", events)
        self.assertFalse(res.fallback)

    def test_fast_path_off_by_default(self):
        backend = Counting()
        run("Any SLA breaches?", DATA, backend)
        self.assertEqual(backend.calls, 1)

    def test_aliases_flow_through_the_pipeline(self):
        backend = Counting()
        res = run("List all the tickets", DATA, backend, fast_path=True, aliases=ALIASES)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(res.plan["nodes"][0]["source"], "/cases")

    def test_model_answers_are_cached_router_answers_are_not(self):
        cache, backend = PlanCache(), Counting()
        run("Any SLA breaches?", DATA, backend, cache=cache, fast_path=True)
        self.assertEqual(len(cache), 0)
        run("Summarize my day", DATA, backend, cache=cache, fast_path=True)
        self.assertEqual(len(cache), 1)
        again = run("my DAY, summarize!", DATA, backend, cache=cache, fast_path=True)  # reordered + filler
        self.assertEqual(backend.calls, 1)
        self.assertTrue(again.attempts[0]["cached"])


class CacheKeyAndPersistenceTests(unittest.TestCase):
    def test_key_ignores_order_case_punctuation_and_filler(self):
        a = normalize_prompt("Show me the HIGH priority cases!")
        self.assertEqual(a, normalize_prompt("cases, high priority"))
        self.assertNotEqual(a, normalize_prompt("show low priority cases"))
        self.assertNotEqual(normalize_prompt("cases and team"), normalize_prompt("cases or team"))
        self.assertNotEqual(normalize_prompt("top 3 cases"), normalize_prompt("top 5 cases"))

    def test_aliases_share_cache_entries(self):
        self.assertEqual(normalize_prompt("my tickets", ALIASES), normalize_prompt("my cases", ALIASES))

    def test_cache_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "plans.json"
            first = PlanCache(path=path)
            backend = Counting()
            run("Summarize my day", DATA, backend, cache=first)
            second = PlanCache(path=path)  # a new process would do exactly this
            res = run("summarize my day", DATA, backend, cache=second)
            self.assertEqual(backend.calls, 1)
            self.assertTrue(res.attempts[0]["cached"])

    def test_corrupt_cache_file_starts_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plans.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(len(PlanCache(path=path)), 0)


if __name__ == "__main__":
    unittest.main()
