import json
import unittest
from pathlib import Path

from cui import run
from cui.catalog import build_catalog
from cui.pipeline import PlanCache, prefix_messages
from cui.router import evidence, fill_collection
from cui.selector import MAX_PICKS, picks_to_plan, render_options, select_messages, select_options, select_schema

ROOT = Path(__file__).parent.parent
DATA = json.loads((ROOT / "data" / "sample_api.json").read_text(encoding="utf-8"))
CAT = build_catalog(DATA)
OPTS = select_options(CAT)


class Picker:
    """Backend that answers with a fixed pick list, like a model in select mode would."""
    name = "picker"

    def __init__(self, *picks):
        self.replies, self.calls, self.schemas = list(picks), 0, []

    def generate(self, messages, schema):
        self.schemas.append(schema)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply if isinstance(reply, str) else json.dumps({"pick": reply}), {}


class OptionsTests(unittest.TestCase):
    def test_every_scalar_and_collection_listed_once(self):
        self.assertEqual(len(OPTS), len(CAT.scalars) + len(CAT.collections))
        text = render_options(CAT, OPTS)
        self.assertIn("LIST /cases | 8 items", text)
        self.assertIn("priority (High/Low/Medium)", text)  # enum values shown so the model knows what lists contain
        self.assertIn("VALUE /today/csat", text)

    def test_schema_only_allows_catalog_paths(self):
        s = select_schema(OPTS)["properties"]["pick"]
        self.assertEqual(set(s["items"]["enum"]), set(CAT.scalars) | set(CAT.collections))
        self.assertEqual(s["maxItems"], MAX_PICKS)


class PicksToPlanTests(unittest.TestCase):
    def test_value_pick(self):
        plan, errs = picks_to_plan({"pick": ["/queue/slaBreaches"]}, "any sla breaches?", CAT, OPTS)
        self.assertEqual((plan["nodes"], errs), ([{"type": "metric", "value": "/queue/slaBreaches"}], []))

    def test_list_pick_gets_filter_sort_limit_columns_from_the_prompt(self):
        plan, _ = picks_to_plan({"pick": ["/cases"]}, "show the 2 oldest high priority cases as a table with customer name", CAT, OPTS)
        node = plan["nodes"][0]
        self.assertEqual((node["type"], node["filter"], node["limit"], node["sort_by"], node["sort_dir"]),
                         ("table", {"field": "priority", "equals": "High"}, 2, "ageHours", "desc"))
        self.assertEqual(node["fields"], ["customer.name"])

    def test_lenient_fill_keeps_what_it_can(self):
        # the strict router declines "came in by phone" (unknown word); after the model picks /cases the filter still applies
        node = fill_collection("Show cases that came in by phone", CAT, "/cases")
        self.assertEqual(node["filter"], {"field": "channel", "equals": "Phone"})
        node = fill_collection("Show high priority escalated cases", CAT, "/cases")  # two values: keeps one, does not fail
        self.assertIn(node["filter"]["equals"], ("High", "Escalated"))

    def test_extra_pick_without_any_supporting_word_is_dropped(self):
        plan, _ = picks_to_plan({"pick": ["/cases", "/agent/email"]}, "Show cases that came in by phone", CAT, OPTS)
        self.assertEqual([n.get("source") or n.get("value") for n in plan["nodes"]], ["/cases"])
        plan, _ = picks_to_plan({"pick": ["/cases", "/team"]}, "Which team members are available?", CAT, OPTS)
        self.assertEqual([n["source"] for n in plan["nodes"]], ["/team"])

    def test_supported_extra_picks_and_unsupported_sets_are_kept(self):
        plan, _ = picks_to_plan({"pick": ["/today/csat", "/team"]}, "Compare my CSAT with the team", CAT, OPTS)
        self.assertEqual(len(plan["nodes"]), 2)
        plan, _ = picks_to_plan({"pick": ["/today/resolved", "/today/csat"]}, "Summarize my day", CAT, OPTS)
        self.assertEqual(len(plan["nodes"]), 2)  # nothing supported: trust the model

    def test_evidence(self):
        self.assertGreater(evidence("Which team members are available?", CAT, "list", "/team"), 0)
        self.assertEqual(evidence("Which team members are available?", CAT, "list", "/cases"), 0)
        self.assertGreater(evidence("what's the average wait", CAT, "value", "/queue/avgWaitSeconds"), 0)
        self.assertEqual(evidence("what's the average wait", CAT, "value", "/queue/pending"), 0)

    def test_repeats_dropped_and_bad_replies_rejected(self):
        plan, _ = picks_to_plan({"pick": ["/today/csat"] * 2}, "csat", CAT, OPTS)
        self.assertEqual(len(plan["nodes"]), 1)
        for bad in ({"pick": []}, {"pick": [1]}, {"pick": ["/nope"]}, {"pick": [True]}, {"nodes": []}, []):
            self.assertEqual(picks_to_plan(bad, "x", CAT, OPTS)[0], None, bad)


class SelectModePipelineTests(unittest.TestCase):
    def test_model_sees_the_select_schema_and_plan_is_compiled(self):
        backend = Picker(["/cases"])
        res = run("Show the Platinum customer cases", DATA, backend, mode="select")
        self.assertIn("pick", backend.schemas[0]["properties"])
        node = res.dsl["root"]["children"][0]
        self.assertEqual((node["source"], node["filter"]["equals"], node["count"]), ("/cases", "Platinum", 2))
        self.assertEqual(res.stages["schema"]["mode"], "select")

    def test_invalid_reply_is_repaired_then_falls_back(self):
        backend = Picker("not json", ["/today/csat"])
        res = run("csat", DATA, backend, mode="select")
        self.assertEqual((backend.calls, res.fallback), (2, False))
        backend = Picker("{}")
        res = run("csat", DATA, backend, mode="select", max_retries=1)
        self.assertTrue(res.fallback)

    def test_select_and_plan_modes_do_not_share_cache_entries(self):
        cache = PlanCache()
        run("Summarize my day", DATA, Picker(["/today/resolved"]), cache=cache, mode="select")
        res = run("Summarize my day", DATA, Picker(["/today/resolved"]), cache=cache, mode="plan")
        self.assertFalse(res.attempts[0].get("cached", False))

    def test_router_still_runs_first(self):
        backend = Picker(["/cases"])
        res = run("Any SLA breaches?", DATA, backend, fast_path=True, mode="select")
        self.assertEqual(backend.calls, 0)
        self.assertTrue(res.attempts[0]["fast_path"])

    def test_prefix_matches_the_real_request_prefix(self):
        real = select_messages("x", CAT, OPTS)
        prefix = prefix_messages(CAT, "select")
        self.assertEqual(prefix[0], real[0])
        self.assertTrue(real[1]["content"].startswith(prefix[1]["content"]))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            run("x", DATA, Picker(["/cases"]), mode="fast")


if __name__ == "__main__":
    unittest.main()
