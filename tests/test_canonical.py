"""Cross-check of Rust's canonical form against an independent NetworkX oracle.

The design doc's test for canonical form: two implementations sharing no code
must agree on thousands of random balls. The oracle buckets balls by
Weisfeiler-Lehman hash with depth labels (an isomorphism invariant, so
different hashes prove non-isomorphic) and settles equal hashes with VF2,
matching nodes on depth so the root can only map to the root.

Skipped as a whole when the extension is not built (see rust/README.md).
"""

import random
from collections import defaultdict

import networkx as nx
import pytest

pytest.importorskip("graph_search_rs")

from graph_search.balltype import BallType  # noqa: E402
from graph_search.compiled import canonical_key  # noqa: E402


def to_graph(edges):
    """A NetworkX graph with each node's distance from root 0 as attribute "depth"."""
    graph = nx.Graph(edges) if edges else nx.empty_graph(1)
    for node, depth in nx.single_source_shortest_path_length(graph, 0).items():
        graph.nodes[node]["depth"] = str(depth)
    return graph


def wl_hash(graph):
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="depth")


def oracle_isomorphic(a, b):
    """Root-preserving isomorphism, decided without any Rust code."""
    if wl_hash(a) != wl_hash(b):
        return False
    return nx.is_isomorphic(a, b, node_match=lambda x, y: x["depth"] == y["depth"])


def random_ball(rng, n, extra):
    """A random connected graph on 0..n-1: a random tree plus `extra` random edges."""
    edges = {(rng.randrange(v), v) for v in range(1, n)}
    for _ in range(extra if n >= 2 else 0):
        u, v = rng.sample(range(n), 2)
        edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def relabel(rng, edges, n):
    """Renumber nodes at random, keeping the root at 0, and shuffle edges."""
    rest = list(range(1, n))
    rng.shuffle(rest)
    perm = [0] + rest
    out = [(perm[u], perm[v]) if rng.random() < 0.5 else (perm[v], perm[u]) for u, v in edges]
    rng.shuffle(out)
    return out


def key(edges):
    return canonical_key(BallType.from_edges(edges))


def test_relabeled_copies_get_the_same_key():
    rng = random.Random(1)
    for _ in range(2000):
        n = rng.randint(1, 25)
        edges = random_ball(rng, n, rng.randint(0, n))
        assert key(relabel(rng, edges, n)) == key(edges), edges


def test_keys_partition_balls_exactly_as_the_oracle_does():
    """2,000 small random balls, so many are isomorphic by chance. Within each
    WL bucket, every pair must agree: same key exactly when VF2 finds a
    root-preserving isomorphism. Different buckets are never isomorphic, so
    their keys must differ."""
    rng = random.Random(2)
    balls = []
    for _ in range(2000):
        n = rng.randint(3, 8)
        edges = random_ball(rng, n, rng.randint(0, n))
        balls.append((edges, to_graph(edges), key(edges)))

    buckets = defaultdict(list)
    for ball in balls:
        buckets[wl_hash(ball[1])].append(ball)

    keys_seen = {}
    for bucket_hash, members in buckets.items():
        for i, (edges_a, graph_a, key_a) in enumerate(members):
            assert keys_seen.setdefault(key_a, bucket_hash) == bucket_hash, (
                f"key shared across WL buckets: {edges_a}"
            )
            for edges_b, graph_b, key_b in members[i + 1 :]:
                same = nx.is_isomorphic(
                    graph_a, graph_b, node_match=lambda x, y: x["depth"] == y["depth"]
                )
                assert (key_a == key_b) == same, (edges_a, edges_b)

    distinct = len({ball[2] for ball in balls})
    assert distinct < len(balls), "test should include isomorphic pairs"


def test_wl_hard_pair():
    """Root joined to a 6-cycle versus to two triangles: WL hashes agree, so
    only the canonical search (and VF2) can separate them."""
    hexagon = [(0, v) for v in range(1, 7)] + [(v, v % 6 + 1) for v in range(1, 7)]
    triangles = [(0, v) for v in range(1, 7)] + [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
    assert wl_hash(to_graph(hexagon)) == wl_hash(to_graph(triangles))
    assert not oracle_isomorphic(to_graph(hexagon), to_graph(triangles))
    assert key(hexagon) != key(triangles)


def test_layer_sizes_match_networkx():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 20)
        edges = random_ball(rng, n, rng.randint(0, n))
        layer_sizes = [len(layer) for layer in nx.bfs_layers(to_graph(edges), 0)]
        assert key(edges)[0] == tuple(layer_sizes)


def test_agrees_with_oracle_where_wl_sees_nothing():
    """A root joined to every node of a random cubic graph on 10 nodes: all of
    these share one WL hash, but there are 21 non-isomorphic connected cubic
    graphs on 10 nodes, so many pairs differ.

    Isomorphism is an equivalence relation, so instead of every pair it is
    enough that each key group is isomorphic to its first member and that the
    groups' first members are pairwise non-isomorphic. That keeps VF2, slow at
    proving regular graphs non-isomorphic, to a few hundred calls."""
    groups = defaultdict(list)
    for seed in range(120):
        cubic = nx.random_regular_graph(3, 10, seed=seed)
        edges = [(0, v + 1) for v in range(10)] + [(u + 1, v + 1) for u, v in cubic.edges()]
        groups[key(edges)].append(to_graph(edges))

    graphs = [graph for members in groups.values() for graph in members]
    assert len({wl_hash(graph) for graph in graphs}) == 1
    assert 1 < len(groups) < len(graphs), "needs both isomorphic and non-isomorphic pairs"

    same_depth = lambda x, y: x["depth"] == y["depth"]  # noqa: E731
    for members in groups.values():
        for graph in members[1:]:
            assert nx.is_isomorphic(members[0], graph, node_match=same_depth)
    representatives = [members[0] for members in groups.values()]
    for i, a in enumerate(representatives):
        for b in representatives[i + 1 :]:
            assert not nx.is_isomorphic(a, b, node_match=same_depth)
