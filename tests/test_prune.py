"""Visible sub-ball pruning, run by every compile.

Checked three ways: hand cases, a slow NetworkX version of the same rule
(the oracle), and soundness: the balls of a real graph are never pruned, with
or without unrelated types mixed in. Skipped as a whole when the extension is
not built (see rust/README.md).
"""

import random

import networkx as nx
import pytest

graph_search_rs = pytest.importorskip("graph_search_rs")

from balltype import BallType  # noqa: E402
from compiled import compile_ruleset, load_and_compile, removals  # noqa: E402
from extract import ruleset_from_graph  # noqa: E402
from report import IssueCode  # noqa: E402
from ruleset import Ruleset  # noqa: E402

PRUNING_CODES = {IssueCode.PRUNED_TYPE, IssueCode.UNREALIZABLE}


def compile_types(r, edge_lists):
    """Compile a ruleset made of these edge lists, asserting it has no errors."""
    ruleset = Ruleset(name="test", r=r, types=tuple(BallType.from_edges(e) for e in edge_lists))
    compiled, report = compile_ruleset(ruleset)
    assert report.ok, str(report)
    return compiled, report


# --- Hand cases ------------------------------------------------------------------

def test_star_removal_explained(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "star-only.json")
    [removal] = removals(compiled)
    assert removal.ball_index == removal.type_index == 0
    assert (removal.round, removal.node, removal.depth, removal.radius) == (1, 1, 1, 1)
    assert removal.sub_ball.edges == ((0, 1), (0, 2), (0, 3))  # node a with three neighbors
    assert removal.lost_suppliers == ()


def test_cascade_fixture(fixtures_dir):
    compiled, report = load_and_compile(fixtures_dir / "pruned_type.json")
    assert [(i.code, i.type_index) for i in report.issues] == [
        (IssueCode.PRUNED_TYPE, 1),
        (IssueCode.PRUNED_TYPE, 2),
    ]
    assert compiled.kept_indices == [0]
    first, second = removals(compiled)
    assert (first.type_index, first.round) == (1, 1)
    assert (second.type_index, second.round, second.node) == (2, 2, 1)
    assert second.lost_suppliers == (1,)
    assert "only type 1 had" in report.issues[1].message


def test_removal_indices_follow_merged_duplicates():
    """ball_index counts distinct types; type_index is the position in the file."""
    path_ball = [(0, 1), (0, 2), (1, 3), (2, 4)]
    star = [(0, 1), (1, 2), (1, 3)]
    compiled, report = compile_types(2, [path_ball, path_ball, star])
    [removal] = removals(compiled)
    assert (removal.ball_index, removal.type_index) == (1, 2)
    assert [(i.code, i.type_index) for i in report.issues] == [
        (IssueCode.EXACT_DUPLICATE, 1),
        (IssueCode.PRUNED_TYPE, 2),
    ]


def test_friendship_paradox_is_unrealizable_in_three_rounds(rulesets_dir):
    """Each node's neighbors have one more neighbor than it does: removed from the top degree down."""
    compiled, report = load_and_compile(rulesets_dir / "friendship-paradox.json")
    assert [(rm.type_index, rm.round) for rm in removals(compiled)] == [(2, 1), (1, 2), (0, 3)]
    assert IssueCode.UNREALIZABLE in report.codes()


def test_cornerless_patch_keeps_what_a_tube_needs(rulesets_dir):
    """Without the corner type, everything that could see a corner goes, in two rounds.

    What is left is truly realizable, not just unpruned: every ball of a grid
    rolled into a tube is among the kept types.
    """
    compiled, report = load_and_compile(rulesets_dir / "cornerless-patch-r3.json")
    assert [(rm.type_index, rm.round) for rm in removals(compiled)] == [
        (4, 1),
        (5, 1),
        (7, 1),
        (6, 2),
    ]
    assert IssueCode.UNREALIZABLE not in report.codes()
    tube = nx.cartesian_product(nx.cycle_graph(8), nx.path_graph(8))
    kept = set(compiled.kept_indices)
    for ball in ruleset_from_graph(tube, 3).types:
        assert compiled.index_of(list(ball.edges)) in kept


# --- Oracle ----------------------------------------------------------------------

def _ball_graph(edges):
    graph = nx.Graph(edges)
    graph.add_node(0)
    return graph


def _key(graph, center, radius):
    """Canonical key of the radius ball around center, renumbered center-first by distance."""
    distance = nx.single_source_shortest_path_length(graph, center, cutoff=radius)
    order = sorted(distance, key=lambda v: (distance[v], v))
    number = {v: i for i, v in enumerate(order)}
    sub = graph.subgraph(order)
    return graph_search_rs.canonical_key([(number[u], number[v]) for u, v in sub.edges()])


def naive_prune(r, edge_lists):
    """The pruning rule written as directly as possible: {index: round removed}.

    Each round, recompute every surviving type's cuts at radius 1..r-1, then
    remove every type with a node at depth 1..r-1 whose sub-ball is not
    among them.
    """
    graphs = [_ball_graph(edges) for edges in edge_lists]
    depths = [nx.single_source_shortest_path_length(g, 0) for g in graphs]
    alive = set(range(len(graphs)))
    removed = {}
    round_ = 0
    while True:
        round_ += 1
        cuts = {(k, _key(graphs[t], 0, k)) for t in alive for k in range(1, r)}
        failing = {
            t
            for t in alive
            if any(
                (r - d, _key(graphs[t], u, r - d)) not in cuts
                for u, d in depths[t].items()
                if 1 <= d < r
            )
        }
        if not failing:
            return removed
        for t in failing:
            removed[t] = round_
        alive -= failing


def random_graph(rng):
    n = rng.randint(3, 14)
    graph = nx.gnm_random_graph(n, rng.randint(n - 1, 2 * n), seed=rng.randrange(10**9))
    return graph


def random_ruleset_types(rng, r):
    """Balls from a few random graphs, with some dropped so that others lose support."""
    types = []
    for _ in range(rng.randint(1, 3)):
        types += [list(t.edges) for t in ruleset_from_graph(random_graph(rng), r).types]
    rng.shuffle(types)
    keep = max(1, int(len(types) * rng.uniform(0.4, 1.0)))
    return types[:keep]


def test_matches_naive_pruning():
    rng = random.Random(5)
    pruned_some = cascades = 0
    for _ in range(150):
        r = rng.choice([2, 2, 3])
        compiled, _ = compile_types(r, random_ruleset_types(rng, r))
        distinct = [compiled.edges(i) for i in range(len(compiled))]
        want = naive_prune(r, distinct)
        got = {rm.ball_index: rm.round for rm in removals(compiled)}
        assert got == want
        assert compiled.kept_indices == sorted(set(range(len(compiled))) - set(want))
        pruned_some += bool(want)
        cascades += any(round_ > 1 for round_ in want.values())
    # The random rulesets must actually exercise removals and cascades.
    assert pruned_some > 100
    assert cascades > 50


def test_removal_reasons_hold():
    """Each reported sub-ball really is the node's ball inside the removed type."""
    rng = random.Random(6)
    for _ in range(60):
        r = rng.choice([2, 3])
        compiled, _ = compile_types(r, random_ruleset_types(rng, r))
        for rm in removals(compiled):
            graph = _ball_graph(compiled.edges(rm.ball_index))
            assert nx.shortest_path_length(graph, 0, rm.node) == rm.depth
            assert graph_search_rs.canonical_key(list(rm.sub_ball.edges)) == _key(
                graph, rm.node, rm.radius
            )
            assert all(compiled.kept_indices.count(s) == 0 for s in rm.lost_suppliers)


# --- Soundness and invariance ---------------------------------------------------------

def test_balls_of_a_real_graph_are_never_pruned():
    """Every ball of G occurs in G, so pruning must keep them all, even beside junk types."""
    rng = random.Random(7)
    for _ in range(80):
        r = rng.choice([1, 2, 3])
        graph = random_graph(rng)
        own = [list(t.edges) for t in ruleset_from_graph(graph, r).types]
        junk = [list(t.edges) for t in ruleset_from_graph(random_graph(rng), r).types]
        junk = rng.sample(junk, rng.randint(0, len(junk)))
        compiled, report = compile_types(r, own + junk)
        kept = set(compiled.kept_indices)
        for edges in own:
            assert compiled.index_of(edges) in kept


def test_reference_rulesets_compile_without_pruning():
    for graph in [
        nx.grid_2d_graph(6, 6),
        nx.hexagonal_lattice_graph(4, 4),
        nx.petersen_graph(),
        nx.balanced_tree(2, 4),
        nx.disjoint_union(nx.cycle_graph(7), nx.path_graph(3)),
    ]:
        for r in (1, 2, 3):
            _, report = compile_ruleset(ruleset_from_graph(graph, r))
            assert not (report.codes() & PRUNING_CODES), (graph, r, str(report))


def test_type_order_does_not_change_the_result():
    rng = random.Random(8)
    for _ in range(40):
        r = rng.choice([2, 3])
        types = random_ruleset_types(rng, r)
        shuffled = types[:]
        rng.shuffle(shuffled)

        def by_key(edge_lists):
            compiled, _ = compile_types(r, edge_lists)
            return {
                graph_search_rs.canonical_key(compiled.edges(rm.ball_index)): rm.round
                for rm in removals(compiled)
            }

        assert by_key(types) == by_key(shuffled)


def test_pruning_twice_removes_nothing_more():
    rng = random.Random(9)
    for _ in range(40):
        r = rng.choice([2, 3])
        compiled, _ = compile_types(r, random_ruleset_types(rng, r))
        kept = [compiled.edges(i) for i in compiled.kept_indices]
        if kept:
            again, report = compile_types(r, kept)
            assert removals(again) == []
            assert not (report.codes() & PRUNING_CODES)
