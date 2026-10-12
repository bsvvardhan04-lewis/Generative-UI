import http.client
import json
import threading
import unittest

import server
from cui.pipeline import PlanCache


class ServerTests(unittest.TestCase):
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

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, data, hdrs)
        res = conn.getresponse()
        raw = res.read()
        conn.close()
        return res.status, res.getheader("Content-Type"), raw

    def run_events(self, body):
        status, _, raw = self.request("POST", "/api/run", body)
        self.assertEqual(status, 200)
        return [json.loads(line) for line in raw.decode().splitlines() if line]

    def test_static_index_and_spa_fallback(self):
        status, ctype, raw = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"UI Planner", raw)
        self.assertEqual(self.request("GET", "/some/client/route")[0], 200)

    def test_no_path_traversal(self):
        status, _, raw = self.request("GET", "/..%2f..%2fserver.py")
        self.assertNotIn(b"make_server", raw)

    def test_datasets_and_examples(self):
        _, _, raw = self.request("GET", "/api/datasets")
        self.assertEqual([d["id"] for d in json.loads(raw)], ["sample_api"])
        self.assertEqual(self.request("GET", "/api/datasets/nope")[0], 404)
        self.assertEqual(self.request("GET", "/api/datasets/..%2fserver")[0], 400)
        _, _, raw = self.request("GET", "/api/examples")
        self.assertGreaterEqual(len(json.loads(raw)), 10)

    def test_models_endpoint_degrades_when_ollama_is_down(self):
        status, _, raw = self.request("GET", "/api/models?ollamaUrl=http://127.0.0.1:9")
        data = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertFalse(data["reachable"])
        self.assertEqual(data["models"], [])
        self.assertEqual(self.request("GET", "/api/models?ollamaUrl=ftp://x")[0], 400)

    def test_run_streams_stage_events_then_result(self):
        events = self.run_events({"prompt": "show high priority cases", "backend": "heuristic", "dataset": "sample_api"})
        names = [e["event"] for e in events]
        self.assertEqual(names[0], "start")
        for stage in ("catalog", "schema", "model_start", "model_done", "validate", "compile"):
            self.assertIn(stage, names)
        self.assertLess(names.index("catalog"), names.index("model_start"))
        self.assertLess(names.index("validate"), names.index("compile"))
        self.assertEqual(names[-1], "result")
        result = events[-1]["data"]
        self.assertFalse(result["fallback"])
        self.assertEqual(set(result["renders"]), {"markdown", "card", "html"})
        self.assertEqual(result["dsl"]["cui"], "cui/1.0")
        self.assertIn("catalogText", result)

    def test_run_with_custom_data(self):
        events = self.run_events({"prompt": "how many open", "backend": "heuristic", "data": {"stats": {"open": 4}}})
        result = events[-1]["data"]
        self.assertEqual(result["dsl"]["root"]["children"][0]["value"], 4)

    def test_run_validation_errors(self):
        cases = [
            ({"prompt": "", "backend": "heuristic"}, "request"),
            ({"prompt": "x" * 501, "backend": "heuristic"}, "too long"),
            ({"prompt": "x", "backend": "heuristic", "data": []}, "object"),
            ({"prompt": "x", "backend": "heuristic", "retries": 9}, "between 0 and 3"),
            ({"prompt": "x", "backend": "ollama"}, "model"),
            ({"prompt": "x", "backend": "nope"}, "Unknown backend"),
        ]
        for body, expect in cases:
            status, _, raw = self.request("POST", "/api/run", body)
            self.assertEqual(status, 400, body)
            self.assertIn(expect.lower(), json.loads(raw)["error"].lower())

    def test_unreachable_model_falls_back_with_a_clear_error(self):
        # a prompt the fast path declines, so the (unreachable) model is really needed
        events = self.run_events({"prompt": "summarize my day", "backend": "ollama", "model": "x", "baseUrl": "http://127.0.0.1:9", "retries": 0})
        names = [e["event"] for e in events]
        self.assertIn("model_error", names)
        result = events[-1]["data"]
        self.assertTrue(result["fallback"])
        self.assertIn("cannot reach", result["errors"][0].lower())

    def test_post_requires_json_content_type(self):
        status, _, _ = self.request("POST", "/api/run", '{"prompt":"x"}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_rejects_non_loopback_host_header(self):
        status, _, _ = self.request("GET", "/api/datasets", headers={"Host": "evil.example.com"})
        self.assertEqual(status, 403)

    def test_eval_endpoint_rescored_runs(self):
        _, _, raw = self.request("GET", "/api/eval")
        data = json.loads(raw)
        self.assertEqual(len(data["cases"]), 12)
        by_id = {r["id"]: r for r in data["runs"]}
        self.assertEqual(by_id["heuristic__v1"]["summary"]["passed"], 6)
        self.assertEqual(by_id["ollama-llama3.2-1b__v1"]["summary"]["passed"], 3)
        self.assertEqual(by_id["ollama-llama3.2-3b__v1"]["summary"]["passed"], 8)


if __name__ == "__main__":
    unittest.main()
