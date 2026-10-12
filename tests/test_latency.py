import copy
import http.client
import json
import threading
import unittest
from pathlib import Path

import server
from cui import run
from cui.catalog import build_catalog
from cui.compiler import humanize
from cui.pipeline import PlanCache, normalize_prompt, structure_fingerprint
from cui.schema import build_plan_schema

DATA = json.loads((Path(__file__).parent.parent / "data" / "sample_api.json").read_text(encoding="utf-8"))
PLAN = {"nodes": [{"type": "table", "source": "/cases", "fields": ["id", "priority"], "filter": {"field": "status", "equals": "Escalated"}}]}


class Counting:
    name = "counting"

    def __init__(self, plan):
        self.plan, self.calls = plan, 0

    def generate(self, messages, schema):
        self.calls += 1
        return json.dumps(self.plan), {}


class CompactSchemaTests(unittest.TestCase):
    def test_model_writes_no_label_intent_or_format(self):
        schema = build_plan_schema(build_catalog(DATA))
        self.assertEqual(list(schema["properties"]), ["nodes"])
        self.assertEqual(schema["required"], ["nodes"])
        for variant in schema["properties"]["nodes"]["items"]["anyOf"]:
            self.assertNotIn("label", variant["properties"])
            self.assertNotIn("format", variant["properties"])

    def test_labels_are_derived_by_the_compiler(self):
        plan = {"nodes": [{"type": "metric", "value": "/today/csat"}, {"type": "metric", "value": "/queue/slaBreaches"}] + PLAN["nodes"]}
        children = run("x", DATA, Counting(plan)).dsl["root"]["children"]
        labels = [m["label"] for m in children[0]["children"]]
        self.assertEqual(labels, ["CSAT", "SLA Breaches"])
        self.assertEqual(children[1]["label"], "Cases (Escalated)")

    def test_humanize_acronyms(self):
        self.assertEqual(humanize("avgWaitSeconds"), "Avg Wait Seconds")
        self.assertEqual(humanize("customer.name"), "Name")
        self.assertEqual(humanize("resolutionRatePct"), "Resolution Rate %")


class PlanCacheTests(unittest.TestCase):
    def test_second_identical_request_skips_the_model(self):
        cache, backend = PlanCache(), Counting(PLAN)
        first = run("Escalated cases", DATA, backend, cache=cache)
        second = run("  escalated CASES?  ", DATA, backend, cache=cache)  # same request after normalization
        self.assertEqual(backend.calls, 1)
        self.assertTrue(second.attempts[0]["cached"])
        self.assertEqual(first.dsl, second.dsl | {"meta": first.dsl["meta"]})  # only the echoed prompt differs
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_cache_survives_changed_values_but_not_changed_shape(self):
        cache, backend = PlanCache(), Counting(PLAN)
        run("escalated", DATA, backend, cache=cache)
        refreshed = copy.deepcopy(DATA)
        refreshed["queue"]["open"] = 99  # numbers changed, shape did not
        res = run("escalated", refreshed, backend, cache=cache)
        self.assertEqual(backend.calls, 1)
        self.assertTrue(res.attempts[0]["cached"])
        reshaped = copy.deepcopy(DATA)
        reshaped["queue"]["newMetric"] = 1  # shape changed
        run("escalated", reshaped, backend, cache=cache)
        self.assertEqual(backend.calls, 2)

    def test_invalid_plans_are_never_cached(self):
        cache, backend = PlanCache(), Counting({"nodes": [{"type": "metric", "value": "/nope"}]})
        res = run("x", DATA, backend, max_retries=0, cache=cache)
        self.assertTrue(res.fallback)
        run("x", DATA, backend, max_retries=0, cache=cache)
        self.assertEqual(backend.calls, 2)

    def test_different_models_do_not_share_entries(self):
        cache, a, b = PlanCache(), Counting(PLAN), Counting(PLAN)
        b.name = "other"
        run("x", DATA, a, cache=cache)
        run("x", DATA, b, cache=cache)
        self.assertEqual((a.calls, b.calls), (1, 1))

    def test_lru_eviction(self):
        cache = PlanCache(size=2)
        for i in range(3):
            cache.put(("m", i), {"nodes": []})
        self.assertIsNone(cache.get(("m", 0)))
        self.assertIsNotNone(cache.get(("m", 2)))

    def test_normalization_and_fingerprint(self):
        self.assertEqual(normalize_prompt("  Show   Cases! "), normalize_prompt("cases"))
        self.assertEqual(structure_fingerprint(build_catalog(DATA)), structure_fingerprint(build_catalog(copy.deepcopy(DATA))))


class WarmEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.PLAN_CACHE = PlanCache()  # tests must not write the real on-disk cache
        cls.srv = server.make_server(0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request("POST", "/api/warm", json.dumps(body), {"Content-Type": "application/json"})
        res = conn.getresponse()
        out = (res.status, json.loads(res.read()))
        conn.close()
        return out

    def test_baseline_needs_no_warmup(self):
        status, out = self.post({"backend": "heuristic"})
        self.assertEqual((status, out["warmed"]), (200, False))

    def test_unreachable_ollama_is_reported_not_raised(self):
        status, out = self.post({"backend": "ollama", "model": "x", "baseUrl": "http://127.0.0.1:9", "data": {"a": 1}})
        self.assertEqual(status, 200)
        self.assertFalse(out["warmed"])
        self.assertIn("cannot reach", out["error"].lower())

    def test_requires_a_model(self):
        self.assertEqual(self.post({"backend": "ollama"})[0], 400)


if __name__ == "__main__":
    unittest.main()
