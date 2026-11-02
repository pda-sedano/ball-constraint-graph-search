"""A single rooted ball type: a small connected graph whose root is node 0.

A ``BallType`` knows nothing about r. Its depth is intrinsic (the largest
distance from the root). Whether that depth is right for a ruleset is checked
in ``ruleset.py``.

Raw input from a file goes through ``check_edge_list`` first, which reports
every problem as an ``Issue``. ``BallType.from_edges`` assumes the input
already passed and raises ``ValueError`` if it did not, so programmatic
construction (enumeration, extraction from a reference graph) fails loudly
instead of producing a bad object.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from report import Issue, IssueCode

Edge = tuple[int, int]


def normalize_edge(u: int, v: int) -> Edge:
    """Return the edge as ``(min, max)``."""
    return (u, v) if u <= v else (v, u)


def check_edge_list(raw_edges: Any, type_index: int | None = None) -> list[Issue]:
    """Structural checks on an edge list as it came out of JSON.

    Reports every problem rather than stopping at the first. Checks, in order:

    - ``raw_edges`` is a list (else BAD_VALUE, and stop: nothing else is checkable).
    - Each entry is a 2-element list of ints, rejecting bools (MALFORMED_EDGE).
    - No negative nodes (NEGATIVE_NODE).
    - No self-loops (SELF_LOOP).
    - No edge appears twice after normalization (REPEATED_EDGE).
    - Nodes are exactly 0..k-1 (NODE_GAP). An empty list means the single-node
      ball {0}, which passes.
    - Every node is connected to the root (DISCONNECTED). Skip if NODE_GAP fired.

    Args:
        raw_edges: The value of a type's ``"edges"`` key, unvalidated.
        type_index: Attached to every returned issue.

    Returns:
        Issues found. Empty means ``BallType.from_edges`` will succeed.
    """
    issues: list[Issue] = []

    def report(code: IssueCode, message: str) -> None:
        issues.append(Issue(code, message, type_index))

    if not isinstance(raw_edges, list):
        report(IssueCode.BAD_VALUE, f"edges must be a list, got {type(raw_edges).__name__}")
        return issues

    # Nodes and edges from entries that passed the per-edge checks. Self-loop
    # nodes still count as nodes; negative and malformed entries are dropped.
    nodes: set[int] = set()
    seen: set[Edge] = set()

    for i, entry in enumerate(raw_edges):
        if not (
            isinstance(entry, list)
            and len(entry) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in entry)
        ):
            report(IssueCode.MALFORMED_EDGE, f"edge {i} is not a pair of ints: {entry!r}")
            continue
        u, v = entry
        if u < 0 or v < 0:
            report(IssueCode.NEGATIVE_NODE, f"edge {i} has a negative node: {entry!r}")
            continue
        nodes.update((u, v))
        if u == v:
            report(IssueCode.SELF_LOOP, f"edge {i}: node {u} is adjacent to itself")
            continue
        edge = normalize_edge(u, v)
        if edge in seen:
            report(IssueCode.REPEATED_EDGE, f"edge {i} repeats {list(edge)}")
            continue
        seen.add(edge)

    nodes = nodes or {0}
    num_nodes = max(nodes) + 1
    missing = sorted(set(range(num_nodes)) - nodes)
    if missing:
        report(IssueCode.NODE_GAP, f"nodes must be 0..{num_nodes - 1}; missing {missing}")
        return issues

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(seen)
    unreachable = sorted(set(range(num_nodes)) - nx.node_connected_component(graph, 0))
    if unreachable:
        report(IssueCode.DISCONNECTED, f"nodes not connected to the root: {unreachable}")

    return issues


class InvalidBallError(ValueError):
    """Raised when building a ``BallType`` from edges ``check_edge_list`` rejects.

    Attributes:
        issues: Every issue found, so callers such as the loader can report them.
    """

    def __init__(self, issues: list[Issue]) -> None:
        details = "; ".join(str(i) for i in issues)
        super().__init__(f"invalid ball type: {details}")
        self.issues = issues


def _raise_if_issues(issues: list[Issue]) -> None:
    """Raise ``InvalidBallError`` listing every issue, if there are any."""
    if issues:
        raise InvalidBallError(issues)


@dataclass(frozen=True)
class BallType:
    """An immutable rooted ball, root at node 0, nodes 0..num_nodes-1.

    Equality compares structure *and* label and count, so a save/load round
    trip can be checked with ``==``. Use ``structure_key`` to compare structure
    only. Isomorphism is not handled here; that is step 2.

    Attributes:
        num_nodes: Number of nodes, at least 1.
        edges: Normalized ``(min, max)`` edges, sorted, no repeats.
        label: Optional name, only for reading.
        count: Optional occurrence count, from extraction off a reference graph.
        layers: ``layers[d]`` is the sorted tuple of nodes at distance d from
            the root. Computed in ``__post_init__``.
        degrees: ``degrees[v]`` is the degree of node v. Computed in ``__post_init__``.
    """

    num_nodes: int
    edges: tuple[Edge, ...]
    label: str | None = None
    count: int | None = None
    layers: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)
    degrees: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Verify invariants and compute ``layers`` and ``degrees``.

        Raise ``ValueError`` if ``edges`` is not normalized and sorted, if the
        structure would fail ``check_edge_list``, or if num_nodes is inconsistent
        with the edge list. Set the derived fields with
        ``object.__setattr__`` since the dataclass is frozen.
        """
        _raise_if_issues(check_edge_list([list(e) for e in self.edges]))
        if any(tuple(e) != normalize_edge(*e) for e in self.edges):
            raise ValueError("edges must be normalized as (min, max)")
        if list(self.edges) != sorted(self.edges):
            raise ValueError("edges must be sorted")
        inferred = max(v for e in self.edges for v in e) + 1 if self.edges else 1
        if self.num_nodes != inferred:
            raise ValueError(f"num_nodes is {self.num_nodes} but edges span {inferred} nodes")

        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.edges)

        # bfs_layers does not promise any order within a layer, so sort each one.
        layers = tuple(tuple(sorted(layer)) for layer in nx.bfs_layers(graph, 0))
        degrees = tuple(graph.degree[v] for v in range(self.num_nodes))

        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "degrees", degrees)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Sequence[int]],
        *,
        label: str | None = None,
        count: int | None = None,
    ) -> BallType:
        """Build from any edge iterable: normalize, sort, infer ``num_nodes``.

        An empty iterable gives the single-node ball. Raises ``InvalidBallError``
        on input that ``check_edge_list`` would reject, with issues describing
        the edges as passed in.
        """
        # Valid input is checked once, in __post_init__. Only on failure are the
        # edges checked again as given, so positions in messages match the input
        # rather than the normalized, sorted edges.
        raw = [list(e) if isinstance(e, (list, tuple)) else e for e in edges]
        try:
            normalized = tuple(sorted(normalize_edge(*e) for e in raw))
            num_nodes = max(v for _, v in normalized) + 1 if normalized else 1
            return cls(num_nodes, normalized, label=label, count=count)
        except (TypeError, ValueError):
            _raise_if_issues(check_edge_list(raw))
            raise

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        root: Any,
        *,
        label: str | None = None,
        count: int | None = None,
    ) -> BallType:
        """Build from a NetworkX graph, renumbering so ``root`` becomes 0.

        The renumbering is arbitrary (e.g. BFS order); identity up to
        isomorphism is step 2's job. Step 3 uses this for ball extraction.
        """
        if graph.is_directed():
            raise ValueError("graph must be undirected")
        if root not in graph:
            raise ValueError(f"root {root!r} is not in the graph")

        order = [root] + [v for _, v in nx.bfs_edges(graph, root)]
        if len(order) != graph.number_of_nodes():
            # Checked here because isolated nodes have no edges for
            # __post_init__ to catch; they would be dropped silently.
            raise ValueError(
                f"{graph.number_of_nodes() - len(order)} nodes not connected to the root"
            )

        index = {node: i for i, node in enumerate(order)}
        edges = [(index[u], index[v]) for u, v in graph.edges()]
        return cls.from_edges(edges, label=label, count=count)

    @property
    def depth(self) -> int:
        """Largest distance from the root. 0 for the single-node ball."""
        return len(self.layers) - 1

    @property
    def root_degree(self) -> int:
        """Degree of node 0. Used later as a prefilter."""
        return self.degrees[0]

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Number of nodes at each distance. Part of the canonical key in step 2."""
        return tuple(len(layer) for layer in self.layers)

    @property
    def structure_key(self) -> tuple[int, tuple[Edge, ...]]:
        """``(num_nodes, edges)``: structure without label or count.

        Two types with equal keys are exact duplicates. Not an isomorphism key.
        """
        return (self.num_nodes, self.edges)

    def distance(self, node: int) -> int:
        """Distance of ``node`` from the root, read from ``layers``."""
        for d, layer in enumerate(self.layers):
            if node in layer:
                return d
        raise ValueError(f"node {node!r} is not in this ball (nodes 0..{self.num_nodes - 1})")

    def to_networkx(self) -> nx.Graph:
        """A fresh NetworkX graph, with node attribute ``"depth"`` set on every node."""
        graph = nx.Graph()
        for d, layer in enumerate(self.layers):
            graph.add_nodes_from(layer, depth=d)
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
