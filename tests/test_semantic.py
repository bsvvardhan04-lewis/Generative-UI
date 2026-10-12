"""Semantic level logic with a fake, deterministic embedder (no model needed)."""
import json
import math
import re
import unittest
from pathlib import Path

from cui import run
from cui.catalog import build_catalog
from cui.semantic import SemanticIndex, describe

ROOT = Path(__file__).parent.parent
DATA = json.loads((ROOT / "data" / "sample_api.json").read_text(encoding="utf-8"))
CAT = build_catalog(DATA)

# Tiny "meaning" space: each concept is one dimension; a text's vector counts the concept words in it.
CONCEPTS = {
    "speed": {"quickly", "fast", "first", "response", "reply"},
    "mood": {"happy", "satisfied", "csat"},
    "people": {"team", "colleagues", "alongside", "works"},
    "role": {"job", "title", "role"},
}


class FakeEmbedder:
    name = "fake"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(len(texts))
        out = []
        for t in texts:
            words = set(re.findall(r"[a-z]+", t.lower()))
            # a small constant keeps vectors non-zero; catalog descriptions (they contain ": ") and requests use
            # different constant dimensions, so a request with no concept words is unrelated to every option
            doc = ": " in t
            v = [len(words & ws) for ws in CONCEPTS.values()] + [0.05 if doc else 0.0, 0.0 if doc else 0.05]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


class Broken:
    name = "broken"

    def embed(self, texts):
        raise RuntimeError("cannot reach http://127.0.0.1:11434/api/embed")


class Counting:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, schema):
        self.calls += 1
        return json.dumps({"nodes": [{"type": "metric", "value": "/queue/open"}]}), {}


class DescribeTests(unittest.TestCase):
    def test_descriptions_use_words_units_and_fields(self):
        self.assertEqual(describe(CAT, "value", "/today/firstResponseMinutes"),
                         "today first response minutes: a duration of time, how long (also: answer, reply)")
        self.assertEqual(describe(CAT, "value", "/today/resolutionRatePct"),
                         "today resolution rate percent: a percentage rate, a share or proportion")
        team = describe(CAT, "list", "/team")
        self.assertTrue(team.startswith("list of team (also: colleague, coworker, crew, teammate): 4 items, each with name, status, open cases"))
        self.assertIn("Available", team)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.index = SemanticIndex(self.embedder, threshold=0.6, margin=0.05)

    def test_strong_unambiguous_match_answers(self):
        m = self.index.match("what is my job title", CAT)
        self.assertEqual((m.best, m.plan["nodes"]), ("/agent/role", [{"type": "text", "value": "/agent/role"}]))

    def test_weak_match_declines(self):
        m = self.index.match("tell me a joke", CAT)
        self.assertIsNone(m.plan)
        self.assertIn("too weak", m.reason)

    def test_close_runner_up_declines(self):
        # "first response" and "reply" both point at speed; csat at mood: half and half, so no clear winner
        index = SemanticIndex(self.embedder, threshold=0.1, margin=0.2)
        m = index.match("happy reply", CAT)
        self.assertIsNone(m.plan)
        self.assertIn("too close", m.reason)

    def test_list_match_gets_filters_from_the_words(self):
        m = self.index.match("who works alongside me and is available", CAT)
        node = m.plan["nodes"][0]
        self.assertEqual((node["source"], node["filter"]), ("/team", {"field": "status", "equals": "Available"}))

    def test_catalog_embedded_once_per_data_shape(self):
        self.index.match("job title", CAT)
        self.index.match("reply quickly", CAT)
        refreshed = json.loads(json.dumps(DATA))
        refreshed["queue"]["open"] = 99  # same shape, new values
        self.index.match("job title", build_catalog(refreshed))
        self.assertEqual(self.embedder.calls[0], len(CAT.scalars) + len(CAT.collections))
        self.assertEqual(self.embedder.calls[1:], [1, 1, 1])  # afterwards only the query is embedded


class PipelineTests(unittest.TestCase):
    def test_semantic_answer_skips_the_model(self):
        backend, events = Counting(), []
        res = run("what is my job title", DATA, backend, fast_path=True, semantic=SemanticIndex(FakeEmbedder(), 0.6, 0.05),
                  on_event=lambda n, d=None: events.append(n))
        self.assertEqual(backend.calls, 0)
        self.assertIn("semantic", events)
        self.assertTrue(res.attempts[0]["semantic"])
        self.assertEqual(res.dsl["root"]["children"][0]["value"], "Tier 2 Support")

    def test_router_still_wins_first(self):
        backend, emb = Counting(), FakeEmbedder()
        run("Any SLA breaches?", DATA, backend, fast_path=True, semantic=SemanticIndex(emb, 0.6, 0.05))
        self.assertEqual((backend.calls, emb.calls), (0, []))  # answered before any embedding

    def test_declined_goes_to_the_model(self):
        backend, events = Counting(), []
        run("tell me a joke", DATA, backend, fast_path=True, semantic=SemanticIndex(FakeEmbedder(), 0.6, 0.05),
            on_event=lambda n, d=None: events.append(n))
        self.assertEqual(backend.calls, 1)
        self.assertIn("semantic_declined", events)

    def test_missing_embedding_model_is_skipped_not_fatal(self):
        backend, events = Counting(), []
        res = run("what is my job title", DATA, backend, fast_path=True, semantic=SemanticIndex(Broken()),
                  on_event=lambda n, d=None: events.append(n))
        self.assertEqual(backend.calls, 1)
        self.assertIn("semantic_unavailable", events)
        self.assertFalse(res.fallback)


if __name__ == "__main__":
    unittest.main()
