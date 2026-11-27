"""The validity checker, through the Rust core.

A graph with an open set is valid for a ruleset when it is connected, the
open set is within the cap m (if one is given), and every node passes:

- **Core** nodes, at least r from every open node, see their whole ball. It
  must be one of the types that survived pruning.
- **Open** and **shadow** nodes, closer than r to an open node, may see only
  part of their ball. It must fit inside some kept type (see ``embed.py``).

The result says why, node by node, rather than giving a bare verdict. Like
compiled.py, this module imports the ``graph_search_rs`` extension only when
called.

Example::

    import networkx as nx
    from graph_search.check import check_graph
    from graph_search.compiled import load_and_compile

    compiled, _ = load_and_compile("rulesets/path-ball-r2.json")
    print(check_graph(compiled, nx.cycle_graph(8)))               # valid
    print(check_graph(compiled, nx.path_graph(9), open_nodes={0, 8}, max_open=2))
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING

import networkx as nx

from .extract import number_nodes
from .graph_io import load_graph

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset


@dataclass(frozen=True)
class NodeCheck:
    """One node's result.

    ``outcome`` is "matched" (core, its ball is a kept type), "pruned" (core,
    its ball is a type pruning removed), "unmatched" (core, no type) or
    "compatible" (open or shadow). ``types`` holds the matched or pruned type,
    or the compatible types; ball indices, as ``compiled.edges(i)`` takes.
    """

    node_class: str
    """ "open", "shadow" or "core"."""
    distance_to_open: int | None
    outcome: str
    types: tuple[int, ...]

    @property
    def passes(self) -> bool:
        return self.outcome == "matched" or (self.outcome == "compatible" and bool(self.types))


@dataclass(frozen=True)
class CheckResult:
    """Everything the checker found."""

    valid: bool
    components: int
    open_count: int
    max_open: int | None
    energy: int
    """Failing nodes plus (components - 1): what the sampler drives to 0."""
    core_fraction: float
    core_depth: int | None
    """Largest distance to the open set minus r; None in closed mode."""
    nodes: dict[Hashable, NodeCheck]

    @property
    def connected(self) -> bool:
        return self.components <= 1

    @property
    def failing(self) -> list[Hashable]:
        return [v for v, check in self.nodes.items() if not check.passes]

    def __str__(self) -> str:
        """A short summary, e.g. ``invalid: 2 of 9 nodes fail ([0, 8])``."""
        if self.valid:
            head = f"valid: {len(self.nodes)} nodes"
        else:
            reasons = []
            failing = self.failing
            if failing:
                reasons.append(f"{len(failing)} of {len(self.nodes)} nodes fail ({failing[:8]})")
            if not self.connected:
                reasons.append(f"{self.components} components")
            if self.max_open is not None and self.open_count > self.max_open:
                reasons.append(f"{self.open_count} open nodes, above the cap of {self.max_open}")
            head = "invalid: " + "; ".join(reasons)
        depth = "unbounded" if self.core_depth is None else str(self.core_depth)
        return f"{head}\n  core fraction {self.core_fraction:.2f}, core depth {depth}, energy {self.energy}"


def check_graph(
    compiled: CompiledRuleset,
    graph: nx.Graph,
    open_nodes: Iterable[Hashable] = (),
    max_open: int | None = None,
) -> CheckResult:
    """Check ``graph`` with these open nodes against ``compiled``.

    ``max_open`` is the cap m on the open set; None means no cap. Raises
    ValueError for a directed graph, a multigraph, self-loops, or an open
    node not in the graph.
    """
    labels, number, edges = number_nodes(graph)
    open_list = list(open_nodes)
    missing = [v for v in open_list if v not in number]
    if missing:
        raise ValueError(f"open nodes not in the graph: {missing[:5]}")
    report = compiled.check(len(labels), edges, [number[v] for v in open_list], max_open)
    return _result(report, labels)


def check_file(
    compiled: CompiledRuleset, path: str | PathLike[str], max_open: int | None = None
) -> CheckResult:
    """Load a graph file (``graph_io``) and check it. Format errors raise
    ``RulesetError`` with the full report; graph errors such as self-loops or
    repeated edges raise ValueError from Rust."""
    graph_file, _ = load_graph(path, strict=True)
    report = compiled.check(
        graph_file.num_nodes, list(graph_file.edges), list(graph_file.open_nodes), max_open
    )
    return _result(report, list(range(graph_file.num_nodes)))


def _result(report, labels: list[Hashable]) -> CheckResult:
    """Wrap the Rust report, keying nodes by their labels."""
    nodes = {
        label: NodeCheck(node_class, distance, kind, tuple(types))
        for label, node_class, distance, (kind, types) in zip(
            labels, report.classes(), report.distances(), report.outcomes()
        )
    }
    return CheckResult(
        valid=report.valid,
        components=report.components,
        open_count=report.open_count,
        max_open=report.max_open,
        energy=report.energy,
        core_fraction=report.core_fraction,
        core_depth=report.core_depth,
        nodes=nodes,
    )
