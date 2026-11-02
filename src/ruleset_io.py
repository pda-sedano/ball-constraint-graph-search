"""JSON reading and writing for rulesets. A thin layer: all checks live elsewhere.

File format, version 1::

    {
      "format": 1,
      "name": "star-only",
      "r": 2,
      "types": [
        {"label": "leaf-view", "edges": [[0, 1], [1, 2], [1, 3]]}
      ],
      "metadata": {}
    }

Required keys: ``format``, ``name``, ``r``, ``types``. Optional: ``metadata``.
Each type requires ``edges`` and allows ``label`` and ``count``. Any other key,
at either level, is an UNKNOWN_KEY error. Canonical keys are never written.
"""

from __future__ import annotations

import json
from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import Any

from balltype import BallType, InvalidBallError
from report import IssueCode, LoadReport
from ruleset import Ruleset, check_ruleset

FORMAT_VERSION = 1
TOP_LEVEL_REQUIRED = frozenset({"format", "name", "r", "types"})
TOP_LEVEL_OPTIONAL = frozenset({"metadata"})
TYPE_REQUIRED = frozenset({"edges"})
TYPE_OPTIONAL = frozenset({"label", "count"})


# --- Loading ---------------------------------------------------------------


def load_ruleset(
    path: str | PathLike[str], *, strict: bool = False
) -> tuple[Ruleset | None, LoadReport]:
    """Read a ruleset file. See ``loads_ruleset``."""
    return loads_ruleset(Path(path).read_text(encoding="utf-8"), strict=strict)


def loads_ruleset(text: str, *, strict: bool = False) -> tuple[Ruleset | None, LoadReport]:
    """Parse JSON text, then hand off to ``ruleset_from_dict``.

    Invalid JSON gives MALFORMED_JSON and no ruleset.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        report = LoadReport()
        report.add(IssueCode.MALFORMED_JSON, f"invalid JSON: {exc}")
        if strict:
            report.raise_if_errors()
        return None, report
    return ruleset_from_dict(doc, strict=strict)


def ruleset_from_dict(
    doc: Any, *, strict: bool = False
) -> tuple[Ruleset | None, LoadReport]:
    """Build and check a ruleset from already-parsed JSON.

    Steps:
      1. ``_check_top_level``.
      2. ``_parse_type`` on every entry of ``types``, collecting issues from all
         of them even after one fails.
      3. If no errors so far, build the ``Ruleset`` and run ``check_ruleset``.
      4. If ``strict``, call ``report.raise_if_errors()``.

    Returns:
        ``(ruleset, report)``. ``ruleset`` is None whenever the report has errors.
    """
    report = LoadReport()
    types_usable = _check_top_level(doc, report)
    balls = []
    if types_usable:
        balls = [_parse_type(i, raw, report) for i, raw in enumerate(doc["types"])]

    ruleset = None
    if report.ok:
        ruleset = Ruleset(
            name=doc["name"],
            r=doc["r"],
            types=tuple(balls),
            metadata=doc.get("metadata", {}),
        )
        check_ruleset(ruleset, report)
        if not report.ok:
            ruleset = None

    if strict:
        report.raise_if_errors()
    return ruleset, report


def _check_top_level(doc: Any, report: LoadReport) -> bool:
    """Document shape: a JSON object with exactly the allowed keys.

    Reports MISSING_KEY, UNKNOWN_KEY, BAD_FORMAT_VERSION, and BAD_VALUE for
    ``name`` (str), ``r`` (int, not bool), ``types`` (list), ``metadata`` (object).

    Returns:
        True if ``types`` is usable enough to go on to per-type parsing.
    """
    if not isinstance(doc, dict):
        report.add(IssueCode.BAD_VALUE, f"document must be a JSON object, got {type(doc).__name__}")
        return False

    for key in sorted(TOP_LEVEL_REQUIRED - doc.keys()):
        report.add(IssueCode.MISSING_KEY, f"missing required key {key!r}")
    for key in sorted(doc.keys() - TOP_LEVEL_REQUIRED - TOP_LEVEL_OPTIONAL):
        report.add(IssueCode.UNKNOWN_KEY, f"unknown key {key!r}")

    # A bool is an int in Python, so "format": true would otherwise equal 1.
    if "format" in doc and (not _is_int(doc["format"]) or doc["format"] != FORMAT_VERSION):
        report.add(
            IssueCode.BAD_FORMAT_VERSION,
            f"format must be {FORMAT_VERSION}, got {doc['format']!r}",
        )
    if "name" in doc and not isinstance(doc["name"], str):
        report.add(IssueCode.BAD_VALUE, f"name must be a string, got {doc['name']!r}")
    # r below 1 is left to check_ruleset, which owns the rules about r.
    if "r" in doc and not _is_int(doc["r"]):
        report.add(IssueCode.BAD_VALUE, f"r must be an int, got {doc['r']!r}")
    if "metadata" in doc and not isinstance(doc["metadata"], dict):
        report.add(IssueCode.BAD_VALUE, f"metadata must be an object, got {doc['metadata']!r}")

    if "types" not in doc:
        return False
    if not isinstance(doc["types"], list):
        report.add(IssueCode.BAD_VALUE, f"types must be a list, got {type(doc['types']).__name__}")
        return False
    return True


def _parse_type(index: int, raw: Any, report: LoadReport) -> BallType | None:
    """One entry of ``types``: key checks, then ``BallType.from_edges``.

    Reports MISSING_KEY / UNKNOWN_KEY for the entry's keys, BAD_VALUE for a
    non-str ``label`` or a ``count`` that is not a non-negative int, and
    whatever ``from_edges`` rejects. Every issue carries ``type_index=index``.

    Returns:
        The ``BallType`` if this entry produced no errors, else None.
    """
    if not isinstance(raw, dict):
        report.add(IssueCode.BAD_VALUE, f"type must be an object, got {type(raw).__name__}", index)
        return None

    before = len(report.issues)
    for key in sorted(TYPE_REQUIRED - raw.keys()):
        report.add(IssueCode.MISSING_KEY, f"missing required key {key!r}", index)
    for key in sorted(raw.keys() - TYPE_REQUIRED - TYPE_OPTIONAL):
        report.add(IssueCode.UNKNOWN_KEY, f"unknown key {key!r}", index)

    label = raw.get("label")
    if label is not None and not isinstance(label, str):
        report.add(IssueCode.BAD_VALUE, f"label must be a string, got {label!r}", index)
    count = raw.get("count")
    if count is not None and not (_is_int(count) and count >= 0):
        report.add(IssueCode.BAD_VALUE, f"count must be a non-negative int, got {count!r}", index)

    ball = None
    edges = raw.get("edges", [])
    # from_edges accepts any iterable, so a string or dict would be read as edges.
    if not isinstance(edges, list):
        report.add(IssueCode.BAD_VALUE, f"edges must be a list, got {type(edges).__name__}", index)
    elif "edges" in raw:
        try:
            ball = BallType.from_edges(edges, label=label, count=count)
        except InvalidBallError as exc:
            report.extend([replace(issue, type_index=index) for issue in exc.issues])

    # Every issue raised here is an error, so any new issue means no ball.
    if len(report.issues) > before:
        return None
    return ball


def _is_int(value: Any) -> bool:
    """True for an int that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


# --- Saving ----------------------------------------------------------------


def save_ruleset(ruleset: Ruleset, path: str | PathLike[str]) -> None:
    """Write ``dumps_ruleset(ruleset)`` to ``path``."""
    Path(path).write_text(dumps_ruleset(ruleset), encoding="utf-8")


def dumps_ruleset(ruleset: Ruleset) -> str:
    """JSON text for ``ruleset_to_dict``.

    Keep it readable: indent the document, but write each edge list on one line.
    """
    # json.dumps(indent=...) would put every node on its own line, so the
    # document is assembled by hand with one line per type.
    parts = []
    for key, value in ruleset_to_dict(ruleset).items():
        if key == "types" and value:
            rendered = "[\n" + ",\n".join(f"    {json.dumps(t)}" for t in value) + "\n  ]"
        else:
            rendered = json.dumps(value, indent=2).replace("\n", "\n  ")
        parts.append(f"  {json.dumps(key)}: {rendered}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def ruleset_to_dict(ruleset: Ruleset) -> dict[str, Any]:
    """JSON-ready dict. Types in their stored order; ``metadata`` only if non-empty.

    ``ruleset_from_dict(ruleset_to_dict(rs))[0] == rs`` for any valid ``rs``.
    """
    doc: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "name": ruleset.name,
        "r": ruleset.r,
        "types": [ball.to_dict() for ball in ruleset.types],
    }
    if ruleset.metadata:
        doc["metadata"] = dict(ruleset.metadata)
    return doc
