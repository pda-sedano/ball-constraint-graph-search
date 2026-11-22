"""Ball extraction and rulesets from reference graphs, checked against NetworkX.

nx.ego_graph computes the same induced radius-r ball independently, so it is
the oracle for extract_ball. Skipped as a whole when the extension is not
built (see rust/README.md).
"""

import random

import networkx as nx
import pytest

pytest.importorskip("graph_search_rs")

from compiled import compile_ruleset  # noqa: E402
from extract import extract_ball, ruleset_from_graph  # noqa: E402
from report import IssueCode  # noqa: E402
from ruleset_io import load_ruleset, save_ruleset  # noqa: E402


def random_graph(rng, n, m):
    """A random graph whose node labels are strings, to exercise label handling."""
    graph = nx.gnm_random_graph(n, m, seed=rng.randrange(10**9))
    return nx.relabel_nodes(graph, {v: f"v{v}" for v in graph})


# --- extract_ball -------------------------------------------------------------

def test_extract_ball_matches_ego_graph():
    rng = random.Random(4)
    for _ in range(300):
        n = rng.randint(1, 40)
        graph = random_graph(rng, n, rng.randint(0, min(2 * n, n * (n - 1) // 2)))
        center = rng.choice(list(graph))
        r = rng.randint(0, 4)
        got = extract_ball(graph, center, r)
        want = nx.ego_graph(graph, center, radius=r)

        assert got.nodes[0] == center
        assert set(got.nodes) == set(want)
        distances = nx.single_source_shortest_path_length(graph, center, cutoff=r)
        assert got.depths == [distances[v] for v in got.nodes]
        assert got.depths == sorted(got.depths)

        got_edges = {frozenset((got.nodes[a], got.nodes[b])) for a, b in got.ball.edges}
        assert got_edges == {frozenset(e) for e in want.edges()}


def test_extract_ball_keeps_edges_between_outermost_nodes():
    """In a 5-cycle, the two nodes at distance 2 are adjacent: the ball keeps that edge."""
    got = extract_ball(nx.cycle_graph(5), 0, 2)
    assert got.depths == [0, 1, 1, 2, 2]
    assert len(got.ball.edges) == 5


def test_extract_ball_rejects_bad_input():
    with pytest.raises(ValueError, match="not a node"):
        extract_ball(nx.path_graph(3), 7, 1)
    with pytest.raises(ValueError, match="self-loops"):
        extract_ball(nx.Graph([(0, 1), (1, 1)]), 0, 1)
    with pytest.raises(ValueError, match="simple undirected"):
        extract_ball(nx.DiGraph([(0, 1)]), 0, 1)


# --- ruleset_from_graph -----------------------------------------------------------

def test_torus_has_one_ball_type():
    ruleset = ruleset_from_graph(nx.grid_2d_graph(8, 8, periodic=True), r=2)
    [ball] = ruleset.types
    assert ball.count == 64
    assert ruleset.metadata["reference_graph"] == {"nodes": 64, "edges": 128}


def test_open_grid_counts_boundary_shapes():
    """A 5x5 grid at radius 1: 12 edge nodes, 9 interior, 4 corners, most common first."""
    ruleset = ruleset_from_graph(nx.grid_2d_graph(5, 5), r=1, name="grid")
    assert [t.count for t in ruleset.types] == [12, 9, 4]
    assert ruleset.types[2].label == "e.g. node (0, 0)"  # a corner
    assert sum(t.count for t in ruleset.types) == 25


def test_same_types_for_a_relabeled_graph():
    rng = random.Random(8)
    for _ in range(20):
        graph = random_graph(rng, 30, 45)
        shuffled = list(graph)
        rng.shuffle(shuffled)
        relabeled = nx.relabel_nodes(graph, dict(zip(graph, shuffled)))
        summary = lambda rs: [(t.edges, t.count) for t in rs.types]  # noqa: E731
        assert summary(ruleset_from_graph(graph, 2)) == summary(ruleset_from_graph(relabeled, 2))


def test_reference_ruleset_round_trips_and_compiles_cleanly(tmp_path):
    graph = nx.hexagonal_lattice_graph(5, 5)
    ruleset = ruleset_from_graph(graph, r=2, name="honeycomb patch", metadata={"source": "test"})
    path = tmp_path / "honeycomb.json"
    save_ruleset(ruleset, path)
    loaded, report = load_ruleset(path)
    assert report.issues == []
    assert loaded == ruleset
    assert loaded.metadata["source"] == "test"

    compiled, report = compile_ruleset(loaded)
    assert report.issues == []
    assert len(compiled) == len(ruleset.types)


def test_small_component_gives_shallow_type():
    """An isolated node's ball is just the node: depth 0, which compiling flags as shallow."""
    graph = nx.path_graph(6)
    graph.add_node("alone")
    ruleset = ruleset_from_graph(graph, r=2)
    _, report = compile_ruleset(ruleset)
    assert IssueCode.SHALLOW_TYPE in report.codes()
    assert not report.errors


def test_ruleset_from_graph_rejects_bad_r():
    with pytest.raises(ValueError, match="r must be"):
        ruleset_from_graph(nx.path_graph(3), r=0)
