"""The sampler: reach, uniformity against the baseline, reproducibility, files.

The exact detailed-balance test and the zero-energy uniformity test are in
Rust (``rust/core/src/sample.rs``), where they need no checker. These tests
run the real chain with real rulesets.
"""

import math
from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("graph_search_rs")

from balltype import BallType  # noqa: E402
from check import check_graph  # noqa: E402
from compiled import compile_ruleset, load_and_compile  # noqa: E402
from enumeration import GengNotFound, Level, ValidGraph, find_geng  # noqa: E402
from graph_io import load_graph  # noqa: E402
from ruleset import Ruleset  # noqa: E402
from sampler import (  # noqa: E402
    canonical_graph6,
    compare_with_baseline,
    default_start_edges,
    sample,
    save_snapshots,
)

RULESETS_DIR = Path(__file__).parent.parent / "rulesets"
STAR = [[0, 1], [0, 2], [0, 3]]
CUBIC_TYPES = [STAR, STAR + [[1, 2]], STAR + [[1, 2], [2, 3]], STAR + [[1, 2], [2, 3], [1, 3]]]


@pytest.fixture(scope="module")
def cubic():
    """r = 1, every star of degree 3: valid graphs are exactly the connected cubic graphs."""
    types = tuple(BallType.from_edges(edges) for edges in CUBIC_TYPES)
    compiled, report = compile_ruleset(Ruleset(name="cubic", r=1, types=types))
    assert compiled is not None, report
    return compiled


@pytest.fixture(scope="module")
def path_ball():
    compiled, _ = load_and_compile(RULESETS_DIR / "path-ball-r2.json")
    return compiled


def g6(graph):
    return nx.to_graph6_bytes(graph, header=False).decode().strip()


def level_of(n, graphs):
    """A baseline level built by hand, so these tests need no geng."""
    return Level(n, 0, tuple(ValidGraph(g6(g), round(canonical_graph6(g)[1])) for g in graphs))


# --- canonical_graph6 -----------------------------------------------------------------


def test_canonical_graph6_separates_exactly_the_isomorphism_classes():
    """The atlas lists every graph up to 7 nodes once per class: keys must all differ,
    and must survive renumbering."""
    import random

    rng = random.Random(1)
    seen = {}
    for graph in nx.graph_atlas_g():
        key, aut = canonical_graph6(graph)
        assert key not in seen
        seen[key] = graph
        nodes = list(graph)
        rng.shuffle(nodes)
        shuffled = nx.relabel_nodes(graph, dict(zip(graph, nodes, strict=True)))
        assert canonical_graph6(shuffled) == (key, aut)
    assert len(seen) == len(nx.graph_atlas_g())


def test_canonical_graph6_counts_automorphisms():
    assert canonical_graph6(nx.petersen_graph())[1] == 120
    assert canonical_graph6(nx.hypercube_graph(3))[1] == 48
    assert canonical_graph6(g6(nx.cycle_graph(7)))[1] == 14  # from a graph6 line too


# --- running a chain ------------------------------------------------------------------


def test_finds_the_cycle_and_its_snapshots_are_valid(path_ball):
    run = sample(path_ball, 8, 20_000, burn_in=2_000, seed=3, snapshot_every=500)
    cycle_key = canonical_graph6(nx.cycle_graph(8))[0]
    assert [v.graph6 for v in run.visits] == [cycle_key]
    assert run.visits[0].steps == run.valid_steps > 0
    assert run.max_degree == 2
    assert run.snapshots
    for snapshot in run.snapshots:
        assert snapshot.step % 500 == 0
        assert snapshot.automorphisms == 16
        assert check_graph(path_ball, snapshot.to_networkx(8)).valid
    assert "valid at" in str(run)


def test_same_seed_same_run(path_ball):
    a = sample(path_ball, 9, 3_000, seed=5, trace_every=100)
    b = sample(path_ball, 9, 3_000, seed=5, trace_every=100)
    c = sample(path_ball, 9, 3_000, seed=6, trace_every=100)
    assert a == b
    assert a.trace != c.trace or a.final_edges != c.final_edges
    assert len(a.trace) == 30


def test_cubic_graphs_on_six_nodes_turn_up_as_n_factorial_over_aut(cubic):
    """K3,3 (|Aut| 72) and the prism (|Aut| 12): expected shares 1/7 and 6/7."""
    level = level_of(6, [nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3)])
    run = sample(cubic, 6, 150_000, burn_in=5_000, seed=2)
    comparison = compare_with_baseline(run, level)
    assert [round(r.expected, 4) for r in comparison.rows] == [round(6 / 7, 4), round(1 / 7, 4)]
    assert not comparison.unexpected
    assert run.valid_fraction > 0.05
    assert comparison.total_variation < 0.05, comparison


def _geng_available():
    try:
        find_geng()
    except GengNotFound:
        return False
    return True


@pytest.mark.skipif(not _geng_available(), reason="geng (nauty) not found")
def test_cubic_graphs_on_eight_nodes_match_the_enumeration(cubic, tmp_path):
    """Five cubic graphs on 8 nodes with |Aut| 48, 16, 16, 12, 4."""
    from enumeration import baseline

    level = baseline(cubic, 8, min_n=8, root=tmp_path).levels[8]
    assert sorted(v.automorphisms for v in level.valid) == [4, 12, 16, 16, 48]
    run = sample(cubic, 8, 300_000, burn_in=5_000, seed=4)
    comparison = compare_with_baseline(run, level)
    assert not comparison.unexpected
    assert comparison.total_variation < 0.06, comparison


# --- options and errors ---------------------------------------------------------------


def test_starting_graphs(path_ball):
    cycle = nx.cycle_graph(7)
    run = sample(path_ball, 7, 1, start=cycle)
    assert run.final_energy in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    run = sample(path_ball, 7, 0, start=list(cycle.edges))
    assert run.final_energy == 0.0
    assert sorted(run.final_edges) == sorted(tuple(sorted(e)) for e in cycle.edges)
    run = sample(path_ball, 7, 0, start="empty")
    assert run.final_edges == ()
    assert default_start_edges(path_ball, 7) == 7
    with pytest.raises(ValueError, match="nodes, not"):
        sample(path_ball, 8, 10, start=cycle)
    with pytest.raises(ValueError, match="starting graph"):
        sample(path_ball, 4, 10, start=[(0, 1), (0, 2), (0, 3)])  # degree 3 over the cap
    with pytest.raises(ValueError, match="start must be"):
        sample(path_ball, 4, 10, start="full")


def test_bad_settings(path_ball):
    with pytest.raises(ValueError, match="beta"):
        sample(path_ball, 6, 10, beta=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        sample(path_ball, 6, 10, snapshot_every=0)
    nothing_kept, _ = load_and_compile(RULESETS_DIR / "friendship-paradox.json")
    with pytest.raises(ValueError, match="no ball type is kept"):
        sample(nothing_kept, 6, 10)


def test_compare_needs_matching_n_and_counts(path_ball):
    run = sample(path_ball, 6, 2_000, seed=1, count_graphs=False)
    assert not run.counted and run.visits == ()
    assert "graphs not counted" in str(run)
    with pytest.raises(ValueError, match="count_graphs"):
        compare_with_baseline(run, level_of(6, [nx.cycle_graph(6)]))
    with pytest.raises(ValueError, match="n = 7"):
        compare_with_baseline(sample(path_ball, 6, 10), level_of(7, [nx.cycle_graph(7)]))


def test_counting_is_automatic_for_small_n_and_refused_for_large(path_ball):
    """Counting keys are graph6, about n^2/12 bytes each, and graphs rarely recur at
    large n: on by default up to the limit, off above it, and refused if asked for."""
    from sampler import COUNT_GRAPHS_MAX_NODES as limit

    assert sample(path_ball, limit, 10).counted
    large = sample(path_ball, limit + 1, 2_000, seed=2, snapshot_every=100)
    assert not large.counted and large.visits == ()
    assert large.steps == 2_000  # everything else still runs
    with pytest.raises(ValueError, match="count_graphs is for n up to"):
        sample(path_ball, limit + 1, 10, count_graphs=True)
    assert not sample(path_ball, 8, 10, count_graphs=False).counted


def test_save_snapshots_round_trip(path_ball, tmp_path):
    run = sample(path_ball, 8, 6_000, burn_in=1_000, seed=3, snapshot_every=1_000)
    paths = save_snapshots(run, tmp_path / "out", ruleset_name="path-ball")
    assert len(paths) == len(run.snapshots) > 0
    for path, snapshot in zip(paths, run.snapshots, strict=True):
        graph_file, report = load_graph(path)
        assert graph_file is not None, report
        assert graph_file.num_nodes == 8
        assert graph_file.edges == snapshot.edges
        assert graph_file.metadata["step"] == snapshot.step
        assert graph_file.metadata["automorphisms"] == 16
        assert graph_file.metadata["ruleset"] == "path-ball"
        assert math.isclose(graph_file.metadata["beta"], 1.0)
