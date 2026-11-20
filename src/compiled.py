"""Bridge to the Rust core: compile a loaded ruleset and fold Rust's issues into the report.

Needs the ``graph_search_rs`` extension, built from ``rust/py`` (see
rust/README.md). This is the only module that imports it, and only when
compiling, so loading, checking and drawing rulesets work without it.

Typical use::

    compiled, report = load_and_compile("rulesets/star-only.json")
    if compiled is not None:
        compiled.layers(0)
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

from report import Issue, IssueCode, LoadReport
from ruleset import Ruleset
from ruleset_io import load_ruleset

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset

RawIssue = tuple[str, str, int | None]


def compile_ruleset(
    ruleset: Ruleset, report: LoadReport | None = None, *, strict: bool = False
) -> tuple[CompiledRuleset | None, LoadReport]:
    """Compile a ruleset in Rust and add every issue Rust finds to ``report``.

    The ruleset must already pass ``check_ruleset``, as one returned by
    ``load_ruleset`` without errors does. This function does not run it again,
    so ``load_and_compile`` does not repeat the loader's warnings.

    Steps:
      1. Refuse an ``r`` below 1 with ValueError: passing one here is a bug,
         since ``check_ruleset`` reports it for files.
      2. ``graph_search_rs.compile_ruleset(r, [ball.edges for ball in ruleset])``.
      3. ``to_issue`` on each returned tuple, added to the report in order.
      4. If ``strict``, ``report.raise_if_errors()``.

    Args:
        ruleset: A ruleset with no format errors.
        report: Report to append to, e.g. the one from loading. A new one is
            created if None.
        strict: Raise ``RulesetError`` if the report has any errors.

    Returns:
        ``(compiled, report)``. ``compiled`` is None whenever Rust reported an error.
    """
    import graph_search_rs  # here, not at the top, so this module imports without it

    r = ruleset.r
    if not (isinstance(r, int) and not isinstance(r, bool) and r >= 1):
        raise ValueError(f"r must be an int of at least 1, got {r!r}; run check_ruleset first")

    if report is None:
        report = LoadReport()
    compiled, raw_issues = graph_search_rs.compile_ruleset(r, [list(ball.edges) for ball in ruleset])
    report.extend([to_issue(raw) for raw in raw_issues])
    if strict:
        report.raise_if_errors()
    return compiled, report


def load_and_compile(
    path: str | PathLike[str], *, strict: bool = False
) -> tuple[CompiledRuleset | None, LoadReport]:
    """``load_ruleset``, then ``compile_ruleset`` into the same report.

    Compiling is skipped when loading reported errors, so the result is
    ``(None, report)`` with only the loader's issues.
    """
    ruleset, report = load_ruleset(path, strict=strict)
    if ruleset is None:
        return None, report
    return compile_ruleset(ruleset, report, strict=strict)


def to_issue(raw: RawIssue) -> Issue:
    """Convert one ``(code, message, type_index)`` tuple from Rust to an ``Issue``.

    Raises ValueError for a code ``IssueCode`` does not have, which means
    Rust and Python have drifted apart.
    """
    code, message, type_index = raw
    return Issue(IssueCode(code), message, type_index)
