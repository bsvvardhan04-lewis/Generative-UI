import json
import unittest
from pathlib import Path

from cui import run
from cui.adapters import RENDERERS
from cui.catalog import build_catalog, resolve
from cui.llm import HeuristicBackend
from cui.schema import build_plan_schema
from cui.validate import validate_plan

DATA = json.loads((Path(__file__).parent.parent / "data" / "sample_api.json").read_text(encoding="utf-8"))
CAT = build_catalog(DATA)


class Scripted:
    """Backend that replays canned outputs, for testing the repair loop."""
    name = "scripted"

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), 0

    def generate(self, messages, schema):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        self.last_messages = messages
        return out if isinstance(out, str) else json.dumps(out), {}


class CatalogTests(unittest.TestCase):
    def test_scalars_and_collections(self):
        self.assertEqual(CAT.scalars["/queue/avgWaitSeconds"].unit, "seconds")
        self.assertEqual(CAT.scalars["/agent/avatarUrl"].type, "url")
        self.assertEqual(set(CAT.collections), {"/cases", "/recentActivity", "/team"})
        self.assertEqual(CAT.collections["/cases"].count, 8)

    def test_enum_detection(self):
        f = CAT.collections["/cases"].fields
        self.assertEqual(f["priority"].values, ["High", "Low", "Medium"])
        self.assertIn("customer.tier", f)
        self.assertIsNone(f["subject"].values)  # all distinct -> free text


class SchemaTests(unittest.TestCase):
    def test_pointers_are_enums(self):
        schema = build_plan_schema(CAT)
        variants = schema["properties"]["nodes"]["items"]["anyOf"]
        metric = next(v for v in variants if v["properties"]["type"]["enum"] == ["metric"])
        self.assertIn("/today/csat", metric["properties"]["value"]["enum"])
        self.assertNotIn("/agent/name", metric["properties"]["value"]["enum"])  # string, not numeric
        cases_list = next(v for v in variants if v["properties"].get("source", {}).get("enum") == ["/cases"] and v["properties"]["type"]["enum"] == ["list"])
        self.assertIn("customer.name", cases_list["properties"]["fields"]["items"]["enum"])
        self.assertIn("filter", cases_list["properties"])
        team_list = next(v for v in variants if v["properties"].get("source", {}).get("enum") == ["/team"])
        self.assertIn("filter", team_list["properties"])  # team.status is an enum


class ValidatorTests(unittest.TestCase):
    def check(self, node):
        return validate_plan({"intent": "x", "nodes": [node]}, CAT)

    def test_ok(self):
        self.assertEqual(self.check({"type": "metric", "label": "Open", "value": "/queue/open"}), [])

    def test_bad_pointer_and_type(self):
        self.assertTrue(self.check({"type": "metric", "label": "x", "value": "/queue/nope"}))
        self.assertTrue(self.check({"type": "metric", "label": "x", "value": "/agent/name"}))

    def test_collection_checks(self):
        base = {"type": "list", "label": "c", "source": "/cases", "fields": ["subject"]}
        self.assertEqual(self.check(base), [])
        self.assertTrue(self.check({**base, "fields": ["nope"]}))
        self.assertTrue(self.check({**base, "filter": {"field": "priority", "equals": "Urgent"}}))
        self.assertTrue(self.check({**base, "filter": {"field": "subject", "equals": "x"}}))
        self.assertTrue(self.check({**base, "limit": 0}))

    def test_duplicates_and_empty(self):
        n = {"type": "metric", "label": "x", "value": "/queue/open"}
        self.assertTrue(validate_plan({"nodes": [n, dict(n)]}, CAT))
        self.assertTrue(validate_plan({"nodes": []}, CAT))
        self.assertTrue(validate_plan([], CAT))


class PipelineTests(unittest.TestCase):
    def test_filter_sort_limit_compile(self):
        plan = {"intent": "x", "nodes": [{"type": "table", "label": "Cases", "source": "/cases", "fields": ["id", "customer.name", "ageHours"],
                                           "filter": {"field": "priority", "equals": "high"}, "sort_by": "ageHours", "sort_dir": "desc", "limit": 2}]}
        res = run("x", DATA, Scripted(plan))
        t = res.dsl["root"]["children"][0]
        self.assertEqual((t["total"], t["count"]), (8, 2))
        self.assertEqual([r[0]["value"] for r in t["rows"]], ["C-10468", "C-10475"])  # 48.3h, 41.2h
        self.assertEqual(t["rows"][0][1]["value"], "Lena Fischer")

    def test_formats_and_row_grouping(self):
        plan = {"intent": "x", "nodes": [
            {"type": "metric", "label": "Wait", "value": "/queue/avgWaitSeconds"},
            {"type": "metric", "label": "FRT", "value": "/today/firstResponseMinutes"},
            {"type": "metric", "label": "Res", "value": "/today/resolutionRatePct"},
            {"type": "status", "label": "Me", "value": "/agent/status"}]}
        res = run("x", DATA, Scripted(plan))
        row, status = res.dsl["root"]["children"]
        self.assertEqual(row["type"], "row")
        self.assertEqual([m["display"] for m in row["children"]], ["3m 34s", "3.8 min", "87.5%"])
        self.assertEqual(status["tone"], "positive")

    def test_values_come_from_data_not_model(self):
        plan = {"intent": "x", "nodes": [{"type": "metric", "label": "Open", "value": "/queue/open", "format": "auto", "number": 9999}]}
        res = run("x", DATA, Scripted(plan))
        self.assertEqual(res.dsl["root"]["children"][0]["value"], 18)

    def test_repair_loop(self):
        bad = {"intent": "x", "nodes": [{"type": "metric", "label": "x", "value": "/queue/openn"}]}
        good = {"intent": "x", "nodes": [{"type": "metric", "label": "x", "value": "/queue/open"}]}
        b = Scripted(bad, good)
        res = run("x", DATA, b)
        self.assertEqual((b.calls, res.fallback), (2, False))
        self.assertIn("/queue/openn", b.last_messages[-1]["content"])  # errors fed back

    def test_invalid_json_then_fallback(self):
        b = Scripted("not json at all")
        res = run("x", DATA, b, max_retries=1)
        self.assertEqual((b.calls, res.fallback), (2, True))
        self.assertTrue(res.dsl["root"]["children"])  # still renders something

    def test_adapters_on_every_node_type(self):
        plan = {"intent": "x", "nodes": [
            {"type": "metric", "label": "a", "value": "/queue/open"}, {"type": "metric", "label": "b", "value": "/queue/pending"},
            {"type": "text", "label": "Role", "value": "/agent/role"}, {"type": "status", "label": "S", "value": "/agent/status"},
            {"type": "list", "label": "L", "source": "/team", "fields": ["name", "status", "openCases"]},
            {"type": "table", "label": "T", "source": "/cases", "fields": ["id", "priority", "ageHours"]},
            {"type": "list", "label": "Empty", "source": "/cases", "fields": ["id"], "filter": {"field": "priority", "equals": "High"}, "limit": 1}]}
        dsl = run("x", DATA, Scripted(plan)).dsl
        for name, render in RENDERERS.items():
            out = render(dsl)
            self.assertTrue(out, name)
        self.assertEqual(RENDERERS["card"](dsl)["version"], "1.5")


class HeuristicEvalTests(unittest.TestCase):
    def test_baseline_never_falls_back(self):
        cases = json.loads((Path(__file__).parent.parent / "data" / "eval_cases.json").read_text(encoding="utf-8"))
        for c in cases:
            res = run(c["prompt"], DATA, HeuristicBackend())
            self.assertFalse(res.fallback, c["prompt"] + str(res.errors))
            self.assertEqual(len(res.attempts), 1)


if __name__ == "__main__":
    unittest.main()
