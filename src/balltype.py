"""One ball type as written in a ruleset file: an edge list, plus an optional label and count.

Python checks only what it takes to hand a ball type to Rust: the edge list is a
list of pairs of non-negative integers. Everything about the graph itself
(self-loops, repeated edges, node numbering, connectivity, depth against r,
duplicates) is checked in Rust when the ruleset is compiled. Rust also computes
layers and degrees.

Edges are kept exactly as written, in order and orientation, so issues reported
by Rust point at the same positions as the file, and saving reproduces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from report import Issue, IssueCode

Edge = tuple[int, int]


def check_edge_format(raw_edges: Any, type_index: int | None = None) -> list[Issue]:
    """Format checks on an edge list, as it came out of JSON or from code.

    Reports every problem rather than stopping at the first:

    - ``raw_edges`` is a list or tuple (else BAD_VALUE, and stop).
    - Each entry is a 2-element list or tuple of ints, rejecting bools (MALFORMED_EDGE).
    - No negative nodes (NEGATIVE_NODE).

    Graph structure is not checked here; that happens in Rust on compiling.

    Args:
        raw_edges: The value of a type's ``"edges"`` key, unvalidated.
        type_index: Attached to every returned issue.

    Returns:
        Issues found. Empty means ``BallType.from_edges`` will succeed.
    """
    issues: list[Issue] = []

    def report(code: IssueCode, message: str) -> None:
        issues.append(Issue(code, message, type_index))

    if not isinstance(raw_edges, (list, tuple)):
        report(IssueCode.BAD_VALUE, f"edges must be a list, got {type(raw_edges).__name__}")
        return issues

    for i, entry in enumerate(raw_edges):
        if not (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in entry)
        ):
            report(IssueCode.MALFORMED_EDGE, f"edge {i} is not a pair of ints: {entry!r}")
        elif entry[0] < 0 or entry[1] < 0:
            report(IssueCode.NEGATIVE_NODE, f"edge {i} has a negative node: {list(entry)!r}")

    return issues


class InvalidBallError(ValueError):
    """Raised when building a ``BallType`` from edges ``check_edge_format`` rejects.

    Attributes:
        issues: Every issue found, so callers such as the loader can report them.
    """

    def __init__(self, issues: list[Issue]) -> None:
        details = "; ".join(str(i) for i in issues)
        super().__init__(f"invalid ball type: {details}")
        self.issues = issues


@dataclass(frozen=True)
class BallType:
    """An immutable ball type, edges as written. The root is node 0 by convention.

    Equality compares edges, label and count, so a save/load round trip can be
    checked with ``==``. Two types with the same graph written differently are
    not equal here; structural comparison belongs to the compiled ruleset.

    Attributes:
        edges: The edge list, in the order and orientation written.
        label: Optional name, only for reading.
        count: Optional occurrence count, from extraction off a reference graph.
    """

    edges: tuple[Edge, ...]
    label: str | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        """Check the edge format and store edges as a tuple of tuples, so the object is hashable.

        Raises ``InvalidBallError`` listing every format problem.
        """
        issues = check_edge_format(self.edges)
        if issues:
            raise InvalidBallError(issues)
        object.__setattr__(self, "edges", tuple((u, v) for u, v in self.edges))

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Sequence[int]],
        *,
        label: str | None = None,
        count: int | None = None,
    ) -> BallType:
        """Build from any edge iterable, keeping order and orientation.

        An empty iterable gives the single-node ball. Raises ``InvalidBallError``
        on input that ``check_edge_format`` rejects.
        """
        raw = edges if isinstance(edges, (list, tuple)) else list(edges)
        return cls(tuple(raw), label=label, count=count)

    def to_networkx(self) -> nx.Graph:
        """A fresh NetworkX graph of the edges as written.

        The nodes are exactly those named in an edge, except that an empty edge
        list is the single-node ball {0}. A ball type written without node 0 has
        no root here either. Repeated edges collapse and self-loops are kept, so
        this is safe to call on a ball type that would fail compiling, e.g. to
        draw it.
        """
        graph = nx.Graph()
        if not self.edges:
            graph.add_node(0)
        graph.add_edges_from(self.edges)
        return graph

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: ``edges`` always, ``label`` and ``count`` only when set."""
        # Key order matches the ruleset files: label first, then edges.
        out: dict[str, Any] = {}
        if self.label is not None:
            out["label"] = self.label
        out["edges"] = [list(e) for e in self.edges]
        if self.count is not None:
            out["count"] = self.count
        return out
