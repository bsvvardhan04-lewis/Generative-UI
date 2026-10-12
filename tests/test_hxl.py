"""HXL output must follow the format documented in the HXL playground (component docs + tutorials)."""
import json
import re
import unittest
from pathlib import Path

from cui import run
from cui.adapters import hxl

ROOT = Path(__file__).parent.parent
DATA = json.loads((ROOT / "data" / "sample_api.json").read_text(encoding="utf-8"))

# From the HXL playground documentation (Properties / Variants sections of each component).
ALLOWED = {
    "tile/widget": set(),
    "tile/text": {"text", "color", "isItalic", "isStrikethrough", "variant", "weight"},
    "tile/badge": {"label", "variant"},
    "tile/row": {"align", "gap", "isWrapped", "justify"},
    "tile/column": {"align", "gap", "width"},
    "tile/container": {"borderless", "id", "isVisible", "variant"},
    "tile/table": {"caption", "columns", "rows", "appearance", "isStickyHeader", "pagination", "size", "sort"},
}
TEXT_VARIANTS = {"body", "caption", "code", "h1", "h2", "h3", "h4", "h5", "h6"}
BADGE_VARIANTS = {"error", "info", "outline", "primary", "secondary", "success", "warning"}
GAPS = {"none", "xs", "sm", "md", "lg", "xl"}
JUSTIFY = {"start", "center", "end", "between", "around", "evenly"}
NODE_KEYS = {"definition", "attributes", "children", "meta", "ignoreExpressions", "dataProviders"}


class Plan:
    """Backend returning a fixed plan (plan mode), to build specific views."""
    name = "fixed"

    def __init__(self, plan):
        self.plan = plan

    def generate(self, messages, schema):
        return json.dumps(self.plan), {}


def widget_for(nodes):
    res = run("x", DATA, Plan({"nodes": nodes}))
    return hxl.render(res.dsl, DATA), res


def walk(node):
    yield node
    for c in node.get("children", []):
        yield from walk(c)


def resolve(value_root, expr):
    """Evaluate a simple '{!$data.a.b}' expression against the provider value."""
    m = re.fullmatch(r"\{!\$data\.([A-Za-z0-9_.]+)\}", expr)
    obj = value_root
    for k in m.group(1).split("."):
        obj = obj[k]
    return obj


class FormatTests(unittest.TestCase):
    def check_documented(self, widget):
        self.assertEqual(widget["definition"], "tile/widget")
        for node in walk(widget):
            self.assertLessEqual(set(node), NODE_KEYS, node)
            self.assertIn(node["definition"], ALLOWED, node["definition"])
            attrs = node.get("attributes", {})
            self.assertLessEqual(set(attrs), ALLOWED[node["definition"]], node)
            d = node["definition"]
            if d == "tile/text":
                self.assertIn("text", attrs)
                self.assertIn(attrs.get("variant", "body"), TEXT_VARIANTS)
            if d == "tile/badge":
                self.assertIn(attrs["variant"], BADGE_VARIANTS)
            if d in ("tile/row", "tile/column") and "gap" in attrs:
                self.assertIn(attrs["gap"], GAPS)
            if d == "tile/row" and "justify" in attrs:
                self.assertIn(attrs["justify"], JUSTIFY)
            if d == "tile/table":
                self.assertTrue(attrs["caption"])  # required by the docs
                keys = {c["key"] for c in attrs["columns"]}
                for row in attrs["rows"]:
                    self.assertEqual(set(row), keys)  # every column key has a field in every row

    def test_metrics_status_and_data_provider(self):
        widget, _ = widget_for([{"type": "metric", "value": "/queue/open"}, {"type": "metric", "value": "/queue/avgWaitSeconds"},
                                {"type": "status", "value": "/agent/status"}])
        self.check_documented(widget)
        provider = widget["dataProviders"][0]
        self.assertEqual((provider["definition"], provider["expressionKey"], provider["inline"]), ("inline-data", "$data", True))
        self.assertEqual(provider["value"], {"queue": {"open": 18, "avgWaitSeconds": 214}, "agent": {"status": "On shift"}})  # only what is used
        texts = [n["attributes"]["text"] for n in walk(widget) if n["definition"] == "tile/text"]
        self.assertIn("{!$data.queue.open}", texts)
        for t in texts:
            if t.startswith("{!$data."):
                self.assertIsNotNone(resolve(provider["value"], t))  # every binding points at real data
        badge = next(n for n in walk(widget) if n["definition"] == "tile/badge")
        self.assertEqual(badge["attributes"], {"label": "{!$data.agent.status}", "variant": "success"})

    def test_filtered_list_is_bound_with_foreach_and_if(self):
        widget, _ = widget_for([{"type": "list", "source": "/cases", "fields": ["subject", "customer.name"],
                                 "filter": {"field": "status", "equals": "Escalated"}}])
        self.check_documented(widget)
        rep = next(n for n in walk(widget) if "meta" in n)
        self.assertEqual(rep["meta"], {"forEach": "{!$data.cases}", "forItem": "$item", "if": "{!$item.status = 'Escalated'}"})
        texts = [n["attributes"]["text"] for n in walk(rep) if n["definition"] == "tile/text"]
        self.assertEqual(texts, ["{!$item.subject}", "{!$item.customer.name}"])
        cases = widget["dataProviders"][0]["value"]["cases"]
        self.assertEqual(len(cases), 8)
        self.assertEqual(set(cases[0]), {"subject", "customer", "status"})  # fields shown + filter field only

    def test_sorted_or_limited_list_is_literal(self):
        widget, res = widget_for([{"type": "list", "source": "/cases", "fields": ["id", "priority"], "sort_by": "ageHours", "sort_dir": "desc", "limit": 2}])
        self.check_documented(widget)
        self.assertFalse(any("meta" in n for n in walk(widget)))
        texts = [n["attributes"]["text"] for n in walk(widget) if n["definition"] == "tile/text"]
        self.assertIn("C-10468", texts)  # oldest case first, written as a value
        self.assertIn("2 of 8", texts)

    def test_table(self):
        widget, _ = widget_for([{"type": "table", "source": "/cases", "fields": ["id", "customer.name", "ageHours", "openedAt"],
                                 "filter": {"field": "priority", "equals": "High"}, "sort_by": "ageHours", "sort_dir": "desc"}])
        self.check_documented(widget)
        table = next(n for n in walk(widget) if n["definition"] == "tile/table")["attributes"]
        self.assertEqual(table["caption"], "Cases (High)")
        cols = {c["key"]: c for c in table["columns"]}
        self.assertEqual(cols["customer_name"]["header"], "Customer Name")
        self.assertEqual(cols["ageHours"]["align"], "right")
        self.assertEqual(cols["ageHours"]["columnType"], {"type": "number"})
        self.assertEqual(cols["openedAt"]["columnType"], {"type": "date"})
        self.assertEqual(table["sort"], {"key": "ageHours", "direction": "desc"})
        self.assertEqual([r["id"] for r in table["rows"]], ["C-10468", "C-10475", "C-10482"])

    def test_without_data_everything_is_literal(self):
        _, res = widget_for([{"type": "metric", "value": "/queue/open"}])
        widget = hxl.render(res.dsl)
        self.assertNotIn("dataProviders", widget)
        self.assertNotIn("{!", json.dumps(widget))

    def test_data_containing_expression_syntax_is_not_evaluated(self):
        data = {"notes": [{"title": "{!$secret}", "kind": "a"}, {"title": "ok", "kind": "a"}, {"title": "x", "kind": "a"}]}
        res = run("x", data, Plan({"nodes": [{"type": "list", "source": "/notes", "fields": ["title"], "limit": 2}]}))
        widget = hxl.render(res.dsl, data)
        node = next(n for n in walk(widget) if n["definition"] == "tile/text" and n["attributes"]["text"] == "{!$secret}")
        self.assertTrue(node["ignoreExpressions"])

    def test_keys_that_are_not_identifiers_fall_back_to_literals(self):
        data = {"stats": {"open count": 5}}
        res = run("x", data, Plan({"nodes": [{"type": "metric", "value": "/stats/open count"}]}))
        widget = hxl.render(res.dsl, data)
        texts = [n["attributes"]["text"] for n in walk(widget) if n["definition"] == "tile/text"]
        self.assertIn("5", texts)
        self.assertFalse(any(t.startswith("{!") for t in texts))

    def test_filter_value_with_a_quote_falls_back_to_literal(self):
        data = {"items": [{"name": "a", "kind": "O'Neil"}, {"name": "b", "kind": "O'Neil"}, {"name": "c", "kind": "x"}]}
        res = run("x", data, Plan({"nodes": [{"type": "list", "source": "/items", "fields": ["name"], "filter": {"field": "kind", "equals": "O'Neil"}}]}))
        widget = hxl.render(res.dsl, data)
        self.assertFalse(any("meta" in n for n in walk(widget)))


if __name__ == "__main__":
    unittest.main()
