"""Slow NetworkX versions of the embedding check and the validity checker.

Test-only oracles (the design doc's "Python oracle checker"). They restate the
rules directly, compute every distance with NetworkX, and match balls with
NetworkX's VF2, sharing no code with the Rust core. Fine for graphs of a few
dozen nodes.
"""

import math

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


def _distances_to(graph, sources):
    sources = set(sources)
    return nx.multi_source_dijkstra_path_length(graph, sources) if sources else {}


def _pattern(compiled, i):
    """Ball type i as a NetworkX graph, each node labeled with depth and degree."""
    pattern = nx.Graph(compiled.edges(i))
    pattern.add_node(0)
    depth = nx.single_source_shortest_path_length(pattern, 0)
    for t in pattern:
        pattern.nodes[t].update(depth=depth[t], degree=pattern.degree(t))
    return pattern


def _ball(graph, v, r):
    """v's radius-r ball, each node labeled with its distance from v."""
    seen = nx.single_source_shortest_path_length(graph, v, cutoff=r)
    ball = nx.Graph(graph.subgraph(seen))
    for u in ball:
        ball.nodes[u]["depth"] = seen[u]
    return ball


def compatible_types(compiled, graph, open_nodes, v):
    """Indices of the kept types that v's visible ball fits, the slow way."""
    r = compiled.r
    open_nodes = set(open_nodes)
    to_open = _distances_to(graph, open_nodes)
    d_open = lambda u: to_open.get(u, math.inf)  # noqa: E731
    ball = _ball(graph, v, r)
    for u in ball:
        depth = ball.nodes[u]["depth"]
        ball.nodes[u].update(
            lower=min(depth, d_open(v) + 2 + d_open(u)),
            open=u in open_nodes,
            degree=ball.degree(u),
        )

    def fits(t, u):
        """Type node attributes t, visible node attributes u."""
        if not (u["lower"] <= t["depth"] <= u["depth"]):
            return False
        if u["depth"] < r:
            return t["degree"] >= u["degree"] if u["open"] else t["degree"] == u["degree"]
        return True

    return [
        i
        for i in compiled.kept_indices
        if GraphMatcher(_pattern(compiled, i), ball, node_match=fits).subgraph_is_isomorphic()
    ]


def check(compiled, graph, open_nodes=(), max_open=None):
    """The validity checker, the slow way.

    Returns ``(valid, components, nodes)`` where ``nodes`` maps each node to
    ``(node_class, distance_to_open, outcome, types)`` exactly as
    ``check.NodeCheck`` holds them.
    """
    r = compiled.r
    open_nodes = set(open_nodes)
    to_open = _distances_to(graph, open_nodes)
    kept = set(compiled.kept_indices)
    nodes = {}
    for v in graph:
        d = to_open.get(v)
        if d == 0:
            node_class = "open"
        elif d is not None and d < r:
            node_class = "shadow"
        else:
            node_class = "core"
        if node_class == "core":
            ball = _ball(graph, v, r)
            same_depth = lambda a, b: a["depth"] == b["depth"]  # noqa: E731
            matches = [
                i
                for i in range(len(compiled))
                if nx.is_isomorphic(ball, _pattern(compiled, i), node_match=same_depth)
            ]
            if not matches:
                result = ("unmatched", ())
            else:
                [i] = matches  # compiling merged isomorphic types
                result = ("matched" if i in kept else "pruned", (i,))
        else:
            result = ("compatible", tuple(compatible_types(compiled, graph, open_nodes, v)))
        nodes[v] = (node_class, d, *result)
    components = nx.number_connected_components(graph) if len(graph) else 0
    passes = all(
        outcome == "matched" or (outcome == "compatible" and types)
        for _, _, outcome, types in nodes.values()
    )
    within = max_open is None or len(open_nodes) <= max_open
    return components <= 1 and within and passes, components, nodes


def cut_patch(graph, size, flips, rng):
    """A connected piece of graph, with nodes that lost a neighbor open, and
    optionally a few random edge flips."""
    start = rng.choice(list(graph))
    region, frontier = set(), [start]
    while len(region) < size and frontier:
        v = frontier.pop(rng.randrange(len(frontier)))
        if v not in region:
            region.add(v)
            frontier.extend(w for w in graph[v] if w not in region)
    patch = nx.Graph(graph.subgraph(region))
    open_nodes = {v for v in region if any(w not in region for w in graph[v])}
    nodes = list(patch)
    for _ in range(flips if len(nodes) >= 2 else 0):
        a, b = rng.sample(nodes, 2)
        if patch.has_edge(a, b):
            patch.remove_edge(a, b)
        else:
            patch.add_edge(a, b)
    return patch, open_nodes
