"""Balls from graphs, through the Rust core.

- ``extract_ball``: the radius-r ball around one node, for exploring.
- ``ruleset_from_graph``: a ruleset with every ball shape a reference graph
  contains, counted ("all balls in a reference graph" in the design doc).

Both take NetworkX graphs with any hashable node labels. Like compiled.py,
this module imports the ``graph_search_rs`` extension only when called.

Example::

    import networkx as nx
    from graph_search.extract import ruleset_from_graph
    from graph_search.ruleset_io import save_ruleset

    grid = nx.grid_2d_graph(6, 6)                # open grid: corner, edge, interior shapes
    save_ruleset(ruleset_from_graph(grid, r=2, name="6x6 grid"), "rulesets/grid.json")
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, NamedTuple

import networkx as nx

from .balltype import BallType
from .ruleset import Ruleset


class ExtractedBall(NamedTuple):
    """The radius-r ball around a node, in ball numbering: center 0, then BFS order."""

    ball: BallType
    """The induced ball as a ball type, edges normalized and sorted."""
    nodes: list[Hashable]
    """The graph label of each ball node, by ball number. ``nodes[0]`` is the center."""
    depths: list[int]
    """Each ball node's distance from the center, by ball number."""


def extract_ball(graph: nx.Graph, center: Hashable, r: int) -> ExtractedBall:
    """The ball of radius r around ``center``: every node within distance r,
    with the edges among them (including edges between two nodes at distance r).

    Raises ValueError for a directed graph, a multigraph, self-loops, or a
    center not in the graph.
    """
    import graph_search_rs

    labels, number, edges = number_nodes(graph)
    if center not in number:
        raise ValueError(f"center {center!r} is not a node of the graph")
    numbers, depths, ball_edges = graph_search_rs.extract_ball(len(labels), edges, number[center], r)
    return ExtractedBall(
        ball=BallType.from_edges(ball_edges),
        nodes=[labels[i] for i in numbers],
        depths=list(depths),
    )


def ruleset_from_graph(
    graph: nx.Graph,
    r: int,
    name: str = "reference",
    metadata: Mapping[str, Any] | None = None,
) -> Ruleset:
    """A ruleset of every radius-r ball shape in ``graph``, with how many nodes have each.

    Types are most common first, ties broken by canonical key, so isomorphic
    graphs give the same types in the same order. Each type's edges are in
    canonical numbering, its ``count`` is how many nodes have that ball, and
    its label names one such node (``"e.g. node (0, 0)"``) to make boundary
    shapes easy to identify when drawn. ``metadata`` records the reference
    graph's size, plus anything passed in.

    A graph too small for some balls to reach depth r gives shallow types,
    which compiling warns about. Raises ValueError for r below 1, a directed
    graph, a multigraph, or self-loops.
    """
    import graph_search_rs

    if not (isinstance(r, int) and not isinstance(r, bool) and r >= 1):
        raise ValueError(f"r must be an int of at least 1, got {r!r}")
    labels, _, edges = number_nodes(graph)
    census = graph_search_rs.ball_census(len(labels), edges, r)
    types = tuple(
        BallType.from_edges(ball_edges, label=f"e.g. node {labels[example]}", count=count)
        for ball_edges, count, example in census
    )
    info: dict[str, Any] = {
        "reference_graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    }
    info.update(metadata or {})
    return Ruleset(name=name, r=r, types=types, metadata=info)


def number_nodes(
    graph: nx.Graph,
) -> tuple[list[Hashable], dict[Hashable, int], list[tuple[int, int]]]:
    """Number the nodes 0..n-1 in the graph's own order, for passing to Rust:
    the labels in that order, each label's number, and the edges in those
    numbers. Raises ValueError for a directed graph, a multigraph, or self-loops."""
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("expected a simple undirected graph (nx.Graph)")
    loops = list(nx.selfloop_edges(graph))
    if loops:
        raise ValueError(f"self-loops are not allowed: {loops[:5]}")
    labels = list(graph.nodes)
    number = {label: i for i, label in enumerate(labels)}
    return labels, number, [(number[u], number[v]) for u, v in graph.edges()]
