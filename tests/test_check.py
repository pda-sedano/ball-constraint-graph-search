"""The validity checker, checked against the design doc's hand cases and a
slow NetworkX oracle (tests/oracle.py). Skipped as a whole when the extension
is not built (see rust/README.md)."""

import random

import networkx as nx
import pytest

pytest.importorskip("graph_search_rs")

import oracle  # noqa: E402
from balltype import BallType  # noqa: E402
from check import check_file, check_graph  # noqa: E402
from compiled import compile_ruleset, load_and_compile  # noqa: E402
from extract import ruleset_from_graph  # noqa: E402
from graph_io import save_graph  # noqa: E402
from report import RulesetError  # noqa: E402
from ruleset import Ruleset  # noqa: E402


@pytest.fixture
def path_ball(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    return compiled


# --- Hand cases (the design doc's, as literals) ---------------------------------------

def test_star_fails_everywhere_once_pruned(rulesets_dir):
    """r = 2, closed: node a's ball is no type, and the leaves' balls are the
    star type, which pruning removed."""
    compiled, _ = load_and_compile(rulesets_dir / "star-only.json")
    result = check_graph(compiled, nx.Graph([("root", "a"), ("a", "b"), ("a", "c")]))
    assert not result.valid
    assert {v: c.outcome for v, c in result.nodes.items()} == {
        "root": "pruned", "a": "unmatched", "b": "pruned", "c": "pruned"
    }


def test_cycles_valid_from_six_nodes(path_ball):
    for n in range(3, 12):
        assert check_graph(path_ball, nx.cycle_graph(n)).valid == (n >= 6), n


def test_open_path_valid_only_with_room_for_its_ends(path_ball):
    path = nx.path_graph(9)
    assert check_graph(path_ball, path, {0, 8}, max_open=2).valid
    capped = check_graph(path_ball, path, {0, 8}, max_open=0)
    assert not capped.valid and capped.failing == []
    assert "above the cap" in str(capped)
    assert check_graph(path_ball, path).failing == [0, 1, 7, 8]


def test_classes_and_core_measures(path_ball):
    result = check_graph(path_ball, nx.path_graph(9), {0, 8})
    assert [c.node_class for c in result.nodes.values()] == (
        ["open", "shadow"] + ["core"] * 5 + ["shadow", "open"]
    )
    assert (result.core_fraction, result.core_depth) == (5 / 9, 2)
    assert check_graph(path_ball, nx.cycle_graph(7)).core_depth is None


def test_disconnected_is_invalid_with_energy_one(path_ball):
    result = check_graph(path_ball, nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(7)))
    assert (result.valid, result.components, result.energy, result.failing) == (False, 2, 1, [])


def test_rejects_bad_input(path_ball):
    with pytest.raises(ValueError, match="not in the graph"):
        check_graph(path_ball, nx.path_graph(3), ["x"])
    with pytest.raises(ValueError, match="self-loops"):
        check_graph(path_ball, nx.Graph([(0, 0)]))


# --- Properties ----------------------------------------------------------------------

def connected_random_graph(rng, n, m):
    while True:
        graph = nx.gnm_random_graph(n, m, seed=rng.randrange(10**9))
        if nx.is_connected(graph):
            return graph


def test_reference_graph_is_valid_for_its_census():
    rng = random.Random(41)
    for _ in range(10):
        graph = connected_random_graph(rng, 30, 45)
        for r in (1, 2, 3):
            compiled, _ = compile_ruleset(ruleset_from_graph(graph, r))
            assert check_graph(compiled, graph).valid


def test_cut_from_a_tube_is_valid_for_the_cornerless_patch(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "cornerless-patch-r3.json")
    tube = nx.cartesian_product(nx.cycle_graph(10), nx.path_graph(9))
    rng = random.Random(42)
    for _ in range(5):
        patch, open_nodes = oracle.cut_patch(tube, 40, flips=0, rng=rng)
        assert check_graph(compiled, patch, open_nodes, max_open=len(open_nodes)).valid


def test_renumbering_never_changes_a_verdict():
    rng = random.Random(43)
    for round_ in range(30):
        reference = connected_random_graph(rng, 25, 35)
        compiled, _ = compile_ruleset(ruleset_from_graph(reference, 2))
        patch, open_nodes = oracle.cut_patch(reference, 12, flips=round_ % 2 * 2, rng=rng)
        nodes = list(patch)
        shuffled = nodes[:]
        rng.shuffle(shuffled)
        rename = dict(zip(nodes, (f"n{v}" for v in shuffled)))
        before = check_graph(compiled, patch, open_nodes)
        after = check_graph(compiled, nx.relabel_nodes(patch, rename), {rename[v] for v in open_nodes})
        assert before.valid == after.valid
        assert all(before.nodes[v] == after.nodes[rename[v]] for v in nodes)


# --- Against the oracle --------------------------------------------------------------

def test_matches_oracle():
    """Random small graphs, open sets and rulesets, node by node."""
    rng = random.Random(44)
    valid = invalid = 0
    for round_ in range(60):
        r = rng.choice([1, 2, 2, 3])
        source = nx.gnm_random_graph(12, rng.randint(12, 20), seed=rng.randrange(10**9))
        types = [list(t.edges) for t in ruleset_from_graph(source, r).types]
        if round_ % 2 == 0:
            # Every ball of the source, plus a few strangers: patches cut from
            # the source should pass unless flipped.
            other = nx.gnm_random_graph(12, rng.randint(12, 20), seed=rng.randrange(10**9))
            types += [list(t.edges) for t in ruleset_from_graph(other, r).types][:3]
        else:
            rng.shuffle(types)
            types = types[: max(1, len(types) * rng.randint(2, 4) // 4)]
        compiled, report = compile_ruleset(
            Ruleset("random", r, tuple(BallType.from_edges(e) for e in types))
        )
        assert report.ok
        if round_ % 2 == 0:
            graph, open_nodes = oracle.cut_patch(source, 9, flips=round_ % 4 // 2, rng=rng)
        else:
            graph = nx.gnm_random_graph(rng.randint(3, 10), rng.randint(2, 14), seed=rng.randrange(10**9))
            open_nodes = {v for v in graph if rng.random() < 0.3}
        max_open = rng.choice([None, len(open_nodes), len(open_nodes), max(0, len(open_nodes) - 1)])
        got = check_graph(compiled, graph, open_nodes, max_open)
        want_valid, want_components, want_nodes = oracle.check(compiled, graph, open_nodes, max_open)
        assert got.components == want_components
        for v in graph:
            c = got.nodes[v]
            assert (c.node_class, c.distance_to_open, c.outcome, c.types) == want_nodes[v], (round_, v)
        assert got.valid == want_valid, round_
        if round_ % 4 == 0 and (max_open is None or len(open_nodes) <= max_open):
            assert want_valid, f"round {round_}: an unflipped cut patch must pass"
        valid += want_valid
        invalid += not want_valid
    assert valid >= 10 and invalid >= 10, (valid, invalid)


# --- From a file ----------------------------------------------------------------------

def test_check_file(path_ball, tmp_path):
    path = tmp_path / "path.safetensors"
    save_graph(path, nx.path_graph(9), open_nodes=[0, 8])
    result = check_file(path_ball, path, max_open=2)
    assert result.valid and result.nodes[0].node_class == "open"


def test_check_file_reports_format_errors(path_ball, tmp_path):
    path = tmp_path / "junk.safetensors"
    path.write_bytes(b"not a graph")
    with pytest.raises(RulesetError, match="malformed_graph_file"):
        check_file(path_ball, path)
