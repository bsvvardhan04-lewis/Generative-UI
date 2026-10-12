import json
import unittest
from pathlib import Path

from cui import run
from cui.catalog import build_catalog
from cui.pipeline import PlanCache
from cui.rewrite import _distance, data_words, rewrite

ROOT = Path(__file__).parent.parent
DATA = json.loads((ROOT / "data" / "sample_api.json").read_text(encoding="utf-8"))
ALIASES = json.loads((ROOT / "data" / "aliases.json").read_text(encoding="utf-8"))
CAT = build_catalog(DATA)


def rw(prompt, data=None, aliases=ALIASES):
    return rewrite(prompt, build_catalog(data) if data else CAT, aliases)


def levels(r):
    return [(s["level"], s["from"], s["to"]) for s in r.steps]


class Counting:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, schema):
        self.calls += 1
        return json.dumps({"nodes": [{"type": "metric", "value": "/today/resolved"}]}), {}


class LevelTests(unittest.TestCase):
    def test_l0_normalize(self):
        r = rw("Who's on call?", aliases={})
        self.assertEqual(r.text, "who is on call")
        self.assertIn(("normalize", "who's", "who is"), levels(r))

    def test_l1_spelling_fixes_typos_into_data_words(self):
        self.assertEqual(rw("show me escalted cases").text, "show me escalated cases")
        self.assertEqual(rw("number of sla breeches").text, "number of sla breaches")
        self.assertEqual(rw("table of cases with subjct").text, "table of cases with subject")

    def test_l1_spelling_is_conservative(self):
        self.assertEqual(rw("show me the weather forcast").text, "show me the weather forcast")  # nothing close in the data
        self.assertEqual(rw("wscalated cases").text, "wscalated cases")  # first letter differs: not a typo we fix
        self.assertEqual(rw("show the stats").text, "show the stats")  # never "corrected" into a synonym word
        self.assertEqual(rw("abc").text, "abc")  # short words are left alone

    def test_l2_concepts_are_data_aware(self):
        self.assertEqual(rw("how many solved cases").text, "how many resolved cases")
        self.assertIn(("concept", "clients", "customer"), levels(rw("clients with gold tier")))
        # on data that has no "customer", a client stays a client
        other = {"accounts": [{"client": "a", "tier": "x"}, {"client": "b", "tier": "x"}, {"client": "c", "tier": "y"}]}
        self.assertEqual(rw("show client tier", other, {}).text, "show client tier")

    def test_l2_never_remaps_generic_words(self):
        self.assertEqual(rw("latest 3 activity updates").text, "latest 3 activity updates")

    def test_l3_aliases_after_concepts(self):
        r = rw("how is client satisfaction")
        self.assertEqual(r.text, "how is csat")
        self.assertEqual([lv for lv, _, _ in levels(r)], ["concept", "alias"])

    def test_repeated_word_from_an_alias_is_collapsed(self):
        self.assertEqual(rw("first responce time").text, "first response")

    def test_idempotent(self):
        for p in ["show me escalted tickets", "how is client satisfaction", "Who's on call?"]:
            once = rw(p).text
            self.assertEqual(rw(once).text, once, p)

    def test_distance(self):
        self.assertEqual(_distance("avrage", "average", 2), 1)
        self.assertEqual(_distance("form", "from", 2), 1)  # adjacent swap counts once
        self.assertGreater(_distance("weather", "week", 1), 1)

    def test_data_words(self):
        words = data_words(CAT)
        self.assertTrue({"queue", "average", "wait", "escalated", "customer", "tier"} <= words)


class PipelineRewriteTests(unittest.TestCase):
    def test_typo_prompt_is_answered_by_the_router(self):
        backend = Counting()
        res = run("show me escalted tikets", DATA, backend, fast_path=True, aliases=ALIASES)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(res.plan["nodes"][0]["filter"], {"field": "status", "equals": "Escalated"})
        self.assertEqual(res.rewrite["text"], "show me escalated cases")

    def test_rewritten_variants_share_one_cache_entry(self):
        cache, backend = PlanCache(), Counting()
        run("how many reopened cases", DATA, backend, cache=cache, aliases=ALIASES)  # no router: the model answers
        again = run("How many REOPNED cases?", DATA, backend, cache=cache, aliases=ALIASES)  # typo + case + punctuation
        self.assertEqual(backend.calls, 1)
        self.assertTrue(again.attempts[0]["cached"])

    def test_model_keeps_the_original_wording(self):
        seen = {}

        class Spy(Counting):
            def generate(self, messages, schema):
                seen["user"] = messages[-1]["content"]
                return super().generate(messages, schema)

        run("how many solved cases?", DATA, Spy(), aliases=ALIASES)
        self.assertTrue(seen["user"].endswith("USER REQUEST: how many solved cases?"))


if __name__ == "__main__":
    unittest.main()
