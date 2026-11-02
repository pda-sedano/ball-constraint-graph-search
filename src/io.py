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

from os import PathLike
from typing import Any

from .balltype import BallType
from .report import LoadReport
from .ruleset import Ruleset

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
    raise NotImplementedError


def loads_ruleset(text: str, *, strict: bool = False) -> tuple[Ruleset | None, LoadReport]:
    """Parse JSON text, then hand off to ``ruleset_from_dict``.

    Invalid JSON gives MALFORMED_JSON and no ruleset.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def _check_top_level(doc: Any, report: LoadReport) -> bool:
    """Document shape: a JSON object with exactly the allowed keys.

    Reports MISSING_KEY, UNKNOWN_KEY, BAD_FORMAT_VERSION, and BAD_VALUE for
    ``name`` (str), ``r`` (int, not bool), ``types`` (list), ``metadata`` (object).

    Returns:
        True if ``types`` is usable enough to go on to per-type parsing.
    """
    raise NotImplementedError


def _parse_type(index: int, raw: Any, report: LoadReport) -> BallType | None:
    """One entry of ``types``: key checks, then ``check_edge_list``.

    Reports MISSING_KEY / UNKNOWN_KEY for the entry's keys, BAD_VALUE for a
    non-str ``label`` or a ``count`` that is not a non-negative int, and
    whatever ``check_edge_list`` finds. Every issue carries ``type_index=index``.

    Returns:
        The ``BallType`` if this entry produced no errors, else None.
    """
    raise NotImplementedError


# --- Saving ----------------------------------------------------------------


def save_ruleset(ruleset: Ruleset, path: str | PathLike[str]) -> None:
    """Write ``dumps_ruleset(ruleset)`` to ``path``."""
    raise NotImplementedError


def dumps_ruleset(ruleset: Ruleset) -> str:
    """JSON text for ``ruleset_to_dict``.

    Keep it readable: indent the document, but write each edge list on one line.
    """
    raise NotImplementedError


def ruleset_to_dict(ruleset: Ruleset) -> dict[str, Any]:
    """JSON-ready dict. Types in their stored order; ``metadata`` only if non-empty.

    ``ruleset_from_dict(ruleset_to_dict(rs))[0] == rs`` for any valid ``rs``.
    """
    raise NotImplementedError
