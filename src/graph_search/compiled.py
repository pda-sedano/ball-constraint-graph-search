"""Bridge to the Rust core: compile a loaded ruleset and fold Rust's issues into the report.

Needs the ``graph_search_rs`` extension, built from ``rust/py`` (see
rust/README.md). This module and extract.py are the only ones that import
it, and only when called, so loading, checking and drawing rulesets work
without it.

Typical use::

    compiled, report = load_and_compile("rulesets/star-only.json")
    if compiled is not None:
        compiled.layers(0)
        compiled.index_of([(0, 2), (2, 1), (2, 3)])  # 0: same shape as type 0
        compiled.kept_indices                         # []: pruning removed it
        removals(compiled)                            # why
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, NamedTuple

from .balltype import BallType
from .report import Issue, IssueCode, LoadReport
from .ruleset import Ruleset
from .ruleset_io import load_ruleset

if TYPE_CHECKING:
    from graph_search_rs import CanonicalKey, CompiledRuleset

RawIssue = tuple[str, str, int | None]


class Removal(NamedTuple):
    """One ball type removed by visible sub-ball pruning, and why.

    Some node of the removed type, at ``depth`` from its root, sees its whole
    ball of radius ``radius = r - depth`` inside the type. In any graph, that
    node's own radius-r ball must be a type that cuts down to ``sub_ball``,
    and no remaining type does.
    """

    ball_index: int
    """Index in the compiled ruleset, as ``layers`` and ``edges`` take."""
    type_index: int
    """Position in the file, as in the PRUNED_TYPE issue."""
    round: int
    """The pruning round that removed it, from 1."""
    node: int
    """The removed type's node whose sub-ball no remaining type has."""
    depth: int
    """That node's distance from the root."""
    radius: int
    """The sub-ball's radius, r - depth."""
    sub_ball: BallType
    """The sub-ball, numbered from ``node`` as root 0."""
    lost_suppliers: tuple[int, ...]
    """Ball indices of the types that had this sub-ball at their root, all
    removed in earlier rounds. Empty in round 1: no type ever had it."""


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


def removals(compiled: CompiledRuleset) -> list[Removal]:
    """Every type pruning removed from ``compiled``, by round, then index."""
    return [
        Removal(
            ball_index=index,
            type_index=compiled.source_indices[index],
            round=round_,
            node=node,
            depth=depth,
            radius=compiled.r - depth,
            sub_ball=BallType.from_edges(edges),
            lost_suppliers=tuple(lost),
        )
        for index, round_, node, depth, edges, lost in compiled.removals()
    ]


def canonical_key(ball: BallType) -> CanonicalKey:
    """Canonical key of one ball type: equal keys mean isomorphic, root to root.

    A hashable ``(layer_sizes, edges)`` tuple. Raises ValueError if the edges
    are not a valid ball (the message lists every structural problem). Keys
    are not stable across versions of the Rust code, so never save them.
    """
    import graph_search_rs

    return graph_search_rs.canonical_key(list(ball.edges))


def to_issue(raw: RawIssue) -> Issue:
    """Convert one ``(code, message, type_index)`` tuple from Rust to an ``Issue``.

    Raises ValueError for a code ``IssueCode`` does not have, which means
    Rust and Python have drifted apart.
    """
    code, message, type_index = raw
    return Issue(IssueCode(code), message, type_index)
