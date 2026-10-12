"""Stage 2: semantic validation of a view plan against the catalog.

Independent of the generation backend: even when the model was grammar
constrained, we re-check everything, because unconstrained backends (and
schema-keyword gaps in some engines) exist. Errors are phrased so they can be
fed straight back to the model for a repair attempt.
"""
from __future__ import annotations

from .catalog import NUMERIC, Catalog
from .schema import FORMATS, MAX_FIELDS, MAX_LIMIT, MAX_NODES

SCALAR_TYPES = {"metric", "text", "status"}
COLLECTION_TYPES = {"list", "table"}


def validate_plan(plan, cat: Catalog) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return ["`nodes` must be a non-empty array"]
    if len(nodes) > MAX_NODES:
        errors.append(f"too many nodes ({len(nodes)} > {MAX_NODES}); keep only what the request needs")

    seen = set()
    for i, n in enumerate(nodes[:MAX_NODES + 4]):
        where = f"nodes[{i}]"
        if not isinstance(n, dict):
            errors.append(f"{where}: must be an object")
            continue
        t = n.get("type")
        if t in SCALAR_TYPES:
            errors += _scalar(n, t, cat, where)
            key = (t, n.get("value"))
        elif t in COLLECTION_TYPES:
            errors += _collection(n, cat, where)
            key = (t, n.get("source"), tuple(n.get("fields") or ()))
        else:
            errors.append(f"{where}: unknown type {t!r}; use metric, text, status, list or table")
            continue
        if key in seen:
            errors.append(f"{where}: duplicate of an earlier node")
        seen.add(key)
    return errors


def _scalar(n: dict, t: str, cat: Catalog, where: str) -> list[str]:
    errs = []
    ptr = n.get("value")
    s = cat.scalars.get(ptr)
    if s is None:
        errs.append(f"{where}: value {ptr!r} is not a scalar pointer in the catalog")
        return errs
    if t == "metric":
        if s.type not in NUMERIC:
            errs.append(f"{where}: metric needs a numeric pointer, but {ptr} is {s.type}")
        fmt = n.get("format", "auto")
        if fmt not in FORMATS:
            errs.append(f"{where}: format {fmt!r} must be one of {FORMATS}")
    if t == "status" and s.type in NUMERIC:
        errs.append(f"{where}: status needs a string pointer, but {ptr} is {s.type}")
    return errs


def _collection(n: dict, cat: Catalog, where: str) -> list[str]:
    errs = []
    src = n.get("source")
    col = cat.collections.get(src)
    if col is None:
        return [f"{where}: source {src!r} is not a collection in the catalog"]
    fields = n.get("fields")
    if not isinstance(fields, list) or not fields:
        errs.append(f"{where}: fields must be a non-empty array of field names")
        fields = []
    if len(fields) > MAX_FIELDS:
        errs.append(f"{where}: at most {MAX_FIELDS} fields")
    for f in fields:
        if f not in col.fields:
            errs.append(f"{where}: field {f!r} does not exist on {src}; valid: {list(col.fields)}")
    if "sort_by" in n and n["sort_by"] not in col.fields:
        errs.append(f"{where}: sort_by {n['sort_by']!r} does not exist on {src}")
    if n.get("sort_dir", "asc") not in ("asc", "desc"):
        errs.append(f"{where}: sort_dir must be asc or desc")
    if "limit" in n and not (isinstance(n["limit"], int) and 1 <= n["limit"] <= MAX_LIMIT):
        errs.append(f"{where}: limit must be an integer 1..{MAX_LIMIT}")
    flt = n.get("filter")
    if flt is not None:
        f = col.fields.get(flt.get("field")) if isinstance(flt, dict) else None
        if f is None or not f.values:
            errs.append(f"{where}: filter.field must be one of the enum fields {[k for k, v in col.fields.items() if v.values]}")
        elif str(flt.get("equals", "")).casefold() not in {v.casefold() for v in f.values}:
            errs.append(f"{where}: filter.equals {flt.get('equals')!r} is not a value of {flt['field']}; valid: {f.values}")
    return errs
