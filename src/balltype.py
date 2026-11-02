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

from .report import Issue

Edge = tuple[int, int]


def normalize_edge(u: int, v: int) -> Edge:
    """Return the edge as ``(min, max)``."""
    raise NotImplementedError


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
    raise NotImplementedError


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

        Raise ``ValueError`` if ``edges`` is not normalized and sorted, or if the
        structure would fail ``check_edge_list``. Set the derived fields with
        ``object.__setattr__`` since the dataclass is frozen.
        """
        raise NotImplementedError

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Sequence[int]],
        *,
        label: str | None = None,
        count: int | None = None,
    ) -> BallType:
        """Build from any edge iterable: normalize, sort, infer ``num_nodes``.

        An empty iterable gives the single-node ball. Raises ``ValueError`` on
        input that ``check_edge_list`` would reject.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def depth(self) -> int:
        """Largest distance from the root. 0 for the single-node ball."""
        raise NotImplementedError

    @property
    def root_degree(self) -> int:
        """Degree of node 0. Used later as a prefilter."""
        raise NotImplementedError

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Number of nodes at each distance. Part of the canonical key in step 2."""
        raise NotImplementedError

    @property
    def structure_key(self) -> tuple[int, tuple[Edge, ...]]:
        """``(num_nodes, edges)``: structure without label or count.

        Two types with equal keys are exact duplicates. Not an isomorphism key.
        """
        raise NotImplementedError

    def distance(self, node: int) -> int:
        """Distance of ``node`` from the root, read from ``layers``."""
        raise NotImplementedError

    def to_networkx(self) -> nx.Graph:
        """A fresh NetworkX graph, with node attribute ``"depth"`` set on every node."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: ``edges`` always, ``label`` and ``count`` only when set."""
        raise NotImplementedError
