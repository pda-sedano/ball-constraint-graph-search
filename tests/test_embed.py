"""The embedding check, against an independent NetworkX oracle.

The oracle (tests/oracle.py) restates the design doc's rules as a node-match
function for NetworkX's VF2 matcher, which finds induced subgraphs, and
computes every distance with NetworkX itself. It shares no code with the Rust
matcher, its twin symmetry breaking, or its propagation fallback. Skipped as a whole when
the extension is not built (see rust/README.md).
"""

import random

import networkx as nx
import pytest

pytest.importorskip("graph_search_rs")

from graph_search.balltype import BallType  # noqa: E402
from graph_search.compiled import compile_ruleset, load_and_compile  # noqa: E402
from graph_search.embed import compatible_types  # noqa: E402
from graph_search.extract import ruleset_from_graph  # noqa: E402
from oracle import compatible_types as oracle_compatible  # noqa: E402
from oracle import cut_patch  # noqa: E402
from graph_search.ruleset import Ruleset  # noqa: E402


# --- Hand cases ---------------------------------------------------------------------

def test_open_ends_let_a_path_pass(rulesets_dir):
    """The design doc's case: valid with both ends open, not with none."""
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    path = nx.path_graph(7)
    assert set(map(tuple, compatible_types(compiled, path, {0, 6}).values())) == {(0,)}
    closed = compatible_types(compiled, path)
    assert [v for v, found in closed.items() if not found] == [0, 1, 5, 6]


def test_nodes_and_labels(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    path = nx.relabel_nodes(nx.path_graph(5), {i: f"n{i}" for i in range(5)})
    assert compatible_types(compiled, path, ["n0", "n4"], nodes=["n2"]) == {"n2": [0]}
    with pytest.raises(ValueError, match="not nodes"):
        compatible_types(compiled, path, ["x"])
    with pytest.raises(ValueError, match="simple undirected"):
        compatible_types(compiled, nx.DiGraph([(0, 1)]))


def test_pruned_types_never_match(rulesets_dir):
    """friendship-paradox.json prunes everything, so nothing fits anything."""
    compiled, _ = load_and_compile(rulesets_dir / "friendship-paradox.json")
    star = nx.star_graph(3)
    assert all(found == [] for found in compatible_types(compiled, star, {1}).values())


# --- Against the oracle --------------------------------------------------------------

def random_graph(rng, n, m):
    return nx.gnm_random_graph(n, m, seed=rng.randrange(10**9))


def test_matches_oracle_on_random_patches():
    rng = random.Random(21)
    passed = failed = 0
    for round_ in range(40):
        r = 2 if round_ % 3 else 3
        reference = random_graph(rng, 40, rng.randint(45, 70))
        compiled, report = compile_ruleset(ruleset_from_graph(reference, r))
        assert report.ok
        patch, open_nodes = cut_patch(reference, 18, flips=round_ % 2 * 3, rng=rng)
        got = compatible_types(compiled, patch, open_nodes)
        for v in patch:
            want = oracle_compatible(compiled, patch, open_nodes, v)
            assert got[v] == want, (round_, v)
            passed += bool(want)
            failed += not want
    assert passed > 200 and failed > 50, (passed, failed)


def test_matches_oracle_with_mixed_rulesets():
    """Types from two different graphs, some of them pruned, on patches of a third."""
    rng = random.Random(22)
    for round_ in range(25):
        r = 2
        types = []
        for _ in range(2):
            source = random_graph(rng, 30, rng.randint(35, 50))
            types += [list(t.edges) for t in ruleset_from_graph(source, r).types]
        rng.shuffle(types)
        kept = types[: max(1, len(types) * 2 // 3)]
        ruleset = Ruleset("mixed", r, tuple(BallType.from_edges(e) for e in kept))
        compiled, report = compile_ruleset(ruleset)
        assert report.ok
        target = random_graph(rng, 30, rng.randint(35, 50))
        patch, open_nodes = cut_patch(target, 16, flips=0, rng=rng)
        got = compatible_types(compiled, patch, open_nodes)
        for v in patch:
            assert got[v] == oracle_compatible(compiled, patch, open_nodes, v), (round_, v)


# --- Cut test -----------------------------------------------------------------------

def test_cut_from_a_tube_passes_the_cornerless_patch(rulesets_dir):
    """A piece cut from a grid rolled into a tube has only edge and interior
    balls, all kept by cornerless-patch-r3.json, so every node passes."""
    compiled, _ = load_and_compile(rulesets_dir / "cornerless-patch-r3.json")
    tube = nx.cartesian_product(nx.cycle_graph(10), nx.path_graph(9))
    rng = random.Random(23)
    for _ in range(5):
        patch, open_nodes = cut_patch(tube, 40, flips=0, rng=rng)
        found = compatible_types(compiled, patch, open_nodes)
        assert all(found.values()), [v for v, f in found.items() if not f]
