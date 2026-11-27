"""The sampler: reach, uniformity against the baseline, reproducibility, files.

The exact detailed-balance test and the zero-energy uniformity test are in
Rust (``rust/core/src/sample.rs``), where they need no checker. These tests
run the real chain with real rulesets.
"""

import functools
import json
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
    canonical_key,
    compare_with_baseline,
    default_start_edges,
    load_run,
    sample,
    save_run,
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
    """K3,3 (|Aut| 72) and the prism (|Aut| 12): the chain's own shares are 1/7
    and 6/7; counting each unlabelled graph once, 1/2 each."""
    level = level_of(6, [nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3)])
    run = sample(cubic, 6, 150_000, burn_in=5_000, seed=2)
    numbered = compare_with_baseline(run, level, count="numbered")
    assert [round(r.expected, 4) for r in numbered.rows] == [round(6 / 7, 4), round(1 / 7, 4)]
    assert not numbered.unexpected
    assert run.valid_fraction > 0.05
    assert numbered.total_variation < 0.05, numbered
    unlabelled = compare_with_baseline(run, level)
    assert unlabelled.count == "unlabelled"
    assert [r.expected for r in unlabelled.rows] == [0.5, 0.5]
    assert unlabelled.total_variation < 0.05, unlabelled
    assert "each unlabelled state counted once" in str(unlabelled)


def _geng_available():
    try:
        find_geng()
    except GengNotFound:
        return False
    return True


@pytest.mark.skipif(not _geng_available(), reason="geng (nauty) not found")
def test_cubic_graphs_on_eight_nodes_match_the_enumeration(cubic, tmp_path):
    """Five cubic graphs on 8 nodes with |Aut| 48, 16, 16, 12, 4.

    Counting numberings is the tight test of the chain. Counting each graph once
    weights the cube's rare visits 12 times as heavily as the least symmetric
    graph's, so its estimate is noisier (worst of 10 seeds: 0.060 against 0.038
    numbered) and gets a looser limit."""
    from enumeration import baseline

    level = baseline(cubic, 8, min_n=8, root=tmp_path).levels[8]
    assert sorted(v.automorphisms for v in level.valid) == [4, 12, 16, 16, 48]
    run = sample(cubic, 8, 300_000, burn_in=5_000, seed=4)
    numbered = compare_with_baseline(run, level, count="numbered")
    assert not numbered.unexpected
    assert numbered.total_variation < 0.06, numbered
    unlabelled = compare_with_baseline(run, level)
    assert unlabelled.total_variation < 0.1, unlabelled


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


# --- open mode ------------------------------------------------------------------------


@functools.cache
def atlas():
    """Every graph on up to 7 nodes, for open-mode ground truth without geng."""
    return tuple(nx.graph_atlas_g())


def test_open_level_of_the_path_ball():
    """n = 6, two open nodes: the 6-cycle with the open pair at distance 1, 2 or 3,
    and the 6-path with both ends open."""
    from enumeration import open_level

    compiled, _ = load_and_compile(RULESETS_DIR / "path-ball-r2.json")
    level = open_level(compiled, 6, 2, graphs=atlas())
    assert level.open_nodes == 2
    assert sorted(v.automorphisms for v in level.valid) == [2, 2, 2, 4]
    for valid in level.valid:
        graph = valid.to_networkx()
        assert check_graph(compiled, graph, valid.open, max_open=2).valid
    closed = open_level(compiled, 6, 0, graphs=atlas())
    assert [v.automorphisms for v in closed.valid] == [12]


def test_open_path_ball_runs_match_the_enumeration(path_ball):
    from enumeration import open_level

    level = open_level(path_ball, 6, 2, graphs=atlas())
    run = sample(path_ball, 6, 300_000, open_nodes=2, burn_in=5_000, seed=1)
    comparison = compare_with_baseline(run, level)
    assert run.proposed_swap > 0 and run.accepted_swap > 0
    assert not comparison.unexpected
    assert comparison.total_variation < 0.05, comparison
    assert "2 open nodes" in str(run) and ", 2 open:" in str(comparison)


def test_open_cubic_runs_match_the_enumeration(cubic):
    """Cubic r = 1 with two open nodes on 6 nodes: open nodes may have degree 1 to 3,
    every other node exactly 3, so many more classes than in closed mode."""
    from enumeration import open_level

    level = open_level(cubic, 6, 2, graphs=atlas())
    assert len(level.valid) > 2
    run = sample(cubic, 6, 400_000, open_nodes=2, burn_in=5_000, seed=3)
    comparison = compare_with_baseline(run, level)
    assert not comparison.unexpected
    assert comparison.total_variation < 0.05, comparison


@pytest.mark.skipif(not _geng_available(), reason="geng (nauty) not found")
def test_open_level_from_geng_matches_the_atlas(cubic, tmp_path):
    from enumeration import open_level

    for m in (1, 2, 3):
        ours = open_level(cubic, 6, m, root=tmp_path)
        expected = open_level(cubic, 6, m, graphs=atlas())
        assert sorted((v.graph6, v.open, v.automorphisms) for v in ours.valid) == sorted(
            (v.graph6, v.open, v.automorphisms) for v in expected.valid
        )


def test_open_snapshots_are_valid_patches(cubic):
    run = sample(cubic, 7, 60_000, open_nodes=3, seed=2, snapshot_every=500)
    assert run.snapshots
    assert len(run.final_open) == 3
    for snapshot in run.snapshots:
        assert len(snapshot.open) == 3
        graph = snapshot.to_networkx(7)
        assert check_graph(cubic, graph, snapshot.open, max_open=3).valid
        assert canonical_key(graph, snapshot.open)[2] == snapshot.automorphisms


def test_canonical_key_respects_the_open_set():
    """Equal keys exactly for states that NetworkX finds isomorphic with open nodes
    matched to open nodes, over every graph on 4 nodes and every open set."""
    from itertools import combinations

    from networkx.algorithms.isomorphism import categorical_node_match

    states = []
    for graph in nx.graph_atlas_g()[1:]:
        if graph.number_of_nodes() != 4:
            continue
        for m in range(5):
            for open_set in combinations(graph, m):
                g = graph.copy()
                nx.set_node_attributes(g, {v: v in open_set for v in g}, "open")
                states.append((g, open_set, canonical_key(graph, open_set)))
    match = categorical_node_match("open", False)
    for i, (g, _, key) in enumerate(states):
        for h, _, other in states[i + 1 :]:
            same = nx.is_isomorphic(g, h, node_match=match)
            assert (key[:2] == other[:2]) == same
    cycle = nx.cycle_graph(4)
    assert canonical_key(cycle, [0, 2])[2] == 4
    assert canonical_key(cycle, [0, 1])[2] == 2
    assert canonical_key(cycle)[1] == ()


def test_open_starts_and_errors(path_ball):
    labelled = nx.relabel_nodes(nx.path_graph(6), {v: f"v{v}" for v in range(6)})
    run = sample(path_ball, 6, 0, open_nodes=2, start=labelled, start_open=["v0", "v5"])
    assert run.final_open == (0, 5)
    assert run.final_energy == 0.0
    run = sample(path_ball, 6, 0, open_nodes=2, start="empty", start_open=[3, 1])
    assert run.final_open == (1, 3)
    with pytest.raises(ValueError, match="open nodes not in the starting graph"):
        sample(path_ball, 6, 10, open_nodes=1, start=labelled, start_open=["x"])
    with pytest.raises(ValueError, match="starting open set"):
        sample(path_ball, 6, 10, open_nodes=2, start_open=[1])
    with pytest.raises(ValueError, match="more than the 6 nodes"):
        sample(path_ball, 6, 10, open_nodes=7)
    with pytest.raises(ValueError, match="swap_rate"):
        sample(path_ball, 6, 10, open_nodes=2, swap_rate=1.5)
    with pytest.raises(ValueError, match="open nodes, baseline level"):
        compare_with_baseline(sample(path_ball, 6, 10, open_nodes=2), level_of(6, []))


def test_open_snapshots_round_trip(path_ball, tmp_path):
    run = sample(path_ball, 6, 20_000, open_nodes=2, seed=4, snapshot_every=1_000)
    paths = save_snapshots(run, tmp_path / "open", ruleset_name="path-ball")
    assert len(paths) == len(run.snapshots) > 0
    for path, snapshot in zip(paths, run.snapshots, strict=True):
        graph_file, report = load_graph(path)
        assert graph_file is not None, report
        assert graph_file.open_nodes == snapshot.open
        assert graph_file.metadata["open_nodes"] == 2


# --- counting each unlabelled state once --------------------------------------------


def test_shares_weight_time_by_automorphisms(cubic):
    run = sample(cubic, 6, 60_000, burn_in=2_000, seed=5)
    numbered, unlabelled = run.shares("numbered"), run.shares()
    assert sum(numbered.values()) == pytest.approx(1)
    assert sum(unlabelled.values()) == pytest.approx(1)
    total = sum(v.steps * v.automorphisms for v in run.visits)
    for visit in run.visits:
        key = (visit.graph6, visit.open)
        assert numbered[key] == pytest.approx(visit.steps / run.valid_steps)
        assert unlabelled[key] == pytest.approx(visit.steps * visit.automorphisms / total)
    # K3,3 has 6 times the prism's symmetries and about 1/6 of its time.
    k33 = canonical_key(nx.complete_bipartite_graph(3, 3))[:2]
    assert numbered[k33] < 0.2
    assert unlabelled[k33] == pytest.approx(0.5, abs=0.05)


def test_average_counts_each_unlabelled_state_once(path_ball):
    """Path ball, n = 6, 2 open nodes: three of the four valid states are 6-cycles
    (|Aut| 2, 2, 4) and one is the 6-path (|Aut| 2). Counting each unlabelled state
    once, a quarter are paths; counting numberings, 2/7 are."""
    run = sample(path_ball, 6, 300_000, open_nodes=2, burn_in=5_000, seed=6, snapshot_every=50)

    def is_path(snapshot):
        return float(len(snapshot.edges) == 5)

    assert len(run.snapshots) > 2_000
    assert run.average(is_path) == pytest.approx(1 / 4, abs=0.03)
    assert run.average(is_path, count="numbered") == pytest.approx(2 / 7, abs=0.03)
    weights = run.snapshot_weights()
    assert sum(weights) == pytest.approx(1)
    assert run.effective_samples("numbered") == pytest.approx(len(run.snapshots))
    assert 0.8 * len(run.snapshots) < run.effective_samples() < len(run.snapshots)


def test_equal_automorphisms_lose_nothing_to_weighting(path_ball):
    """Closed path ball at n = 8 has one valid graph, so every weight is equal."""
    run = sample(path_ball, 8, 20_000, burn_in=2_000, seed=3, snapshot_every=200)
    assert run.snapshots
    assert run.effective_samples() == pytest.approx(len(run.snapshots))
    assert run.shares() == run.shares("numbered")


def test_counting_errors(path_ball):
    run = sample(path_ball, 8, 2_000, seed=1, count_graphs=False)
    with pytest.raises(ValueError, match="count must be one of"):
        run.snapshot_weights("labelled")
    with pytest.raises(ValueError, match="count_graphs"):
        run.shares()
    with pytest.raises(ValueError, match="no snapshots"):
        run.average(lambda s: 1.0)
    assert run.effective_samples() == 0.0
    with pytest.raises(ValueError, match="count must be one of"):
        compare_with_baseline(sample(path_ball, 8, 10), level_of(8, []), count="labelled")


# --- saving and loading runs ----------------------------------------------------------


def test_closed_run_round_trips(path_ball, tmp_path):
    run = sample(path_ball, 8, 6_000, burn_in=1_000, seed=3, snapshot_every=500, trace_every=100)
    assert run.snapshots and run.visits and run.trace
    folder = save_run(run, tmp_path / "run", ruleset_name="path-ball", metadata={"note": "hi"})
    assert sorted(p.name for p in folder.iterdir()) == [
        "final.safetensors",
        "run.json",
        "snapshots",
    ]
    loaded = load_run(folder)
    assert loaded == run
    assert loaded.shares() == run.shares()
    record = json.loads((folder / "run.json").read_text())
    assert (record["ruleset"], record["metadata"]) == ("path-ball", {"note": "hi"})
    graph_file, _ = load_graph(folder / "final.safetensors")
    assert graph_file.metadata["step"] == 7_000


def test_open_run_round_trips(cubic, tmp_path):
    run = sample(cubic, 7, 20_000, open_nodes=3, seed=2, snapshot_every=500)
    assert run.snapshots and run.final_open
    assert load_run(save_run(run, tmp_path / "open")) == run


def test_runs_without_snapshots_or_counts_round_trip(path_ball, tmp_path):
    run = sample(path_ball, 40, 500, seed=1)
    assert not run.counted and not run.snapshots
    assert load_run(save_run(run, tmp_path / "bare")) == run


def test_save_run_refuses_to_overwrite_unless_asked(path_ball, tmp_path):
    folder = tmp_path / "run"
    first = sample(path_ball, 8, 6_000, burn_in=1_000, seed=3, snapshot_every=250)
    save_run(first, folder)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_run(first, folder)
    second = sample(path_ball, 8, 3_000, burn_in=1_000, seed=4, snapshot_every=1_000)
    assert len(second.snapshots) < len(first.snapshots)
    save_run(second, folder, overwrite=True)
    assert load_run(folder) == second
    # No stale snapshot files from the first run.
    assert len(list((folder / "snapshots").iterdir())) == len(second.snapshots)


def test_reserved_metadata_keys_are_refused(path_ball, tmp_path):
    run = sample(path_ball, 8, 2_000, seed=1, snapshot_every=500)
    with pytest.raises(ValueError, match=r"\['step'\] are reserved"):
        save_run(run, tmp_path / "run", metadata={"step": 5})
    with pytest.raises(ValueError, match="reserved"):
        save_snapshots(run, tmp_path / "snaps", metadata={"automorphisms": 1})
    assert not (tmp_path / "run").exists()


def test_load_run_errors(path_ball, tmp_path):
    with pytest.raises(FileNotFoundError, match="not a complete saved run"):
        load_run(tmp_path / "nothing")
    run = sample(path_ball, 8, 6_000, burn_in=1_000, seed=3, snapshot_every=500)
    folder = save_run(run, tmp_path / "run")
    record_path = folder / "run.json"
    good = record_path.read_text()

    def edited(change):
        record = json.loads(good)
        change(record)
        record_path.write_text(json.dumps(record))

    edited(lambda r: r.update(format=99))
    with pytest.raises(ValueError, match="format 99"):
        load_run(folder)
    edited(lambda r: r.pop("trace"))
    with pytest.raises(ValueError, match="missing entry 'trace'"):
        load_run(folder)
    edited(lambda r: r["settings"].update(num_nodes=9))
    with pytest.raises(ValueError, match="has 8 nodes, the run 9"):
        load_run(folder)
    edited(lambda r: r["snapshots"][0].update(step=1))
    with pytest.raises(ValueError, match="record says 1"):
        load_run(folder)
    record_path.write_text(good)
    (folder / run_snapshot_name(folder)).unlink()
    with pytest.raises(FileNotFoundError):
        load_run(folder)


def run_snapshot_name(folder):
    return json.loads((folder / "run.json").read_text())["snapshots"][0]["file"]
