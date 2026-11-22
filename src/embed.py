"""The embedding check, through the Rust core: which allowed types could a
node's visible ball be part of?

In a patch, open nodes may have neighbors the patch does not show, so a node
near them sees only part of its ball. Such a node passes if what it sees fits
inside some type that survived pruning, as a rooted induced subgraph, with
two extra rules (the design doc's "Embedding check"):

- Distance window: a node u that looks d(v, u) away may really be closer,
  but only through open nodes, so in the type it sits at a depth between
  min(d(v, u), d(v, O) + 2 + d(u, O)) and d(v, u), where O is the open set.
- Closed degree: a closed node closer than r has all its neighbors in view,
  so its image has exactly its degree; an open one, at least that.

A node far from every open node sees its whole ball, and then the check is
the same as looking its ball up by canonical key. Like compiled.py, this
module imports the ``graph_search_rs`` extension only when called.

Example::

    import networkx as nx
    from compiled import load_and_compile
    from embed import compatible_types

    compiled, _ = load_and_compile("rulesets/path-ball-r2.json")
    path = nx.path_graph(6)
    compatible_types(compiled, path, open_nodes={0, 5})  # every node: [0]
    compatible_types(compiled, path)  # closed: nodes 0, 1, 4, 5 get [], 2 and 3 get [0]
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

import networkx as nx

from extract import number_nodes

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset


def compatible_types(
    compiled: CompiledRuleset,
    graph: nx.Graph,
    open_nodes: Iterable[Hashable] = (),
    nodes: Iterable[Hashable] | None = None,
) -> dict[Hashable, list[int]]:
    """For each of ``nodes`` (default: every node), the indices of the kept
    types in ``compiled`` that its visible ball fits, ascending, given which
    nodes of the patch ``graph`` are open. An empty list means the node fails.

    Indices are ball indices, as ``compiled.edges(i)`` takes; pruned types
    never appear. Raises ValueError for a directed graph, a multigraph,
    self-loops, or a node that is not in the graph.
    """
    labels, number, edges = number_nodes(graph)
    open_list = list(open_nodes)
    centers = labels if nodes is None else list(nodes)
    missing = [v for v in [*open_list, *centers] if v not in number]
    if missing:
        raise ValueError(f"not nodes of the graph: {missing[:5]}")
    found = compiled.compatible_types(
        len(labels),
        edges,
        [number[v] for v in open_list],
        [number[v] for v in centers],
    )
    return dict(zip(centers, found))
