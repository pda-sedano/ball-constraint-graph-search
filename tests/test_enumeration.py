"""The enumeration baseline: graph6 reading, automorphism counts, geng lists
and caching, checked against NetworkX and its graph atlas.

The geng tests are skipped when geng is not installed (see enumeration.py);
the rest need only the extension.
"""

import random
from collections import Counter
from math import factorial

import networkx as nx
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

graph_search_rs = pytest.importorskip("graph_search_rs")

import enumeration  # noqa: E402
from check import check_graph  # noqa: E402
from compiled import compile_ruleset, load_and_compile  # noqa: E402
from enumeration import GengNotFound, baseline, find_geng, ruleset_key  # noqa: E402
from extract import ruleset_from_graph  # noqa: E402
from ruleset import Ruleset  # noqa: E402


def have_geng():
    try:
        find_geng()
        return True
    except GengNotFound:
        return False


needs_geng = pytest.mark.skipif(not have_geng(), reason="geng (nauty) is not installed")


# --- graph6 and |Aut|, no geng needed ---------------------------------------------------

def test_parse_graph6_matches_networkx():
    rng = random.Random(51)
    for n in [*range(0, 12), 62, 63, 64, 100]:
        for _ in range(3):
            graph = nx.gnp_random_graph(n, rng.random(), seed=rng.randrange(10**9))
            line = nx.to_graph6_bytes(graph, header=False).decode().strip()
            num_nodes, edges = graph_search_rs.parse_graph6(line)
            assert num_nodes == n
            assert {frozenset(e) for e in edges} == {frozenset(e) for e in graph.edges()}


def test_parse_graph6_rejects_junk():
    with pytest.raises(ValueError, match="not graph6"):
        graph_search_rs.parse_graph6("B w")


def count_automorphisms(graph):
    return sum(1 for _ in GraphMatcher(graph, graph).isomorphisms_iter())


def test_automorphism_count_matches_networkx():
    rng = random.Random(52)
    for _ in range(150):
        n = rng.randint(0, 8)
        graph = nx.gnp_random_graph(n, rng.random(), seed=rng.randrange(10**9))
        got = graph_search_rs.automorphism_count(n, list(graph.edges()))
        assert got == count_automorphisms(graph), nx.to_graph6_bytes(graph)


# --- Baselines ------------------------------------------------------------------------

@needs_geng
def test_path_ball_baseline_is_the_cycles(rulesets_dir, tmp_path):
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    result = baseline(compiled, max_n=12, root=tmp_path)
    assert result.counts() == {n: int(n >= 6) for n in range(1, 13)}
    for n in range(6, 13):
        [graph] = result.levels[n].valid
        assert nx.is_isomorphic(graph.to_networkx(), nx.cycle_graph(n))
        assert graph.automorphisms == 2 * n
        assert graph.numberings == factorial(n) // (2 * n)


@needs_geng
def test_showcase_finds_the_cube(rulesets_dir, tmp_path):
    compiled, _ = load_and_compile(rulesets_dir / "showcase.json")
    result = baseline(compiled, max_n=8, root=tmp_path)
    assert result.degree_range == (3, 6)
    assert result.counts() == {n: int(n == 8) for n in range(1, 9)}
    [cube] = result.levels[8].valid
    assert nx.is_isomorphic(cube.to_networkx(), nx.hypercube_graph(3))
    assert cube.automorphisms == 48


@needs_geng
def test_impossible_degree_ranges_give_empty_levels(rulesets_dir, tmp_path):
    """dimer-r1: every node has degree 1, so only n = 2 has a connected graph,
    and geng refuses the other sizes outright."""
    compiled, _ = load_and_compile(rulesets_dir / "dimer-r1.json")
    result = baseline(compiled, max_n=6, root=tmp_path)
    assert result.counts() == {1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}


def test_unrealizable_ruleset_needs_no_geng(rulesets_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("GENG", "/nonexistent/geng")
    compiled, _ = load_and_compile(rulesets_dir / "friendship-paradox.json")
    result = baseline(compiled, max_n=5, root=tmp_path)
    assert result.degree_range is None
    assert result.counts() == dict.fromkeys(range(1, 6), 0)


def atlas_valid(compiled, max_n):
    """Valid graphs by the full checker over NetworkX's atlas of every graph
    on up to 7 nodes, as graph6 lines grouped by n."""
    found = {}
    for graph in nx.graph_atlas_g()[1:]:
        if graph.number_of_nodes() <= max_n and nx.is_connected(graph):
            if check_graph(compiled, graph).valid:
                found.setdefault(graph.number_of_nodes(), []).append(graph)
    return found


def same_classes(a, b):
    """Two lists of graphs hold the same isomorphism classes, each once."""
    if len(a) != len(b):
        return False
    unmatched = list(b)
    for g in a:
        match = next((h for h in unmatched if nx.is_isomorphic(g, h)), None)
        if match is None:
            return False
        unmatched.remove(match)
    return True


@needs_geng
def test_matches_the_atlas_up_to_seven_nodes(rulesets_dir, tmp_path):
    rng = random.Random(53)
    rulesets = [load_and_compile(rulesets_dir / name)[0] for name in ("path-ball-r2.json", "dimer-r1.json")]
    for _ in range(12):
        n = rng.randint(3, 7)
        source = nx.empty_graph(2)
        while not nx.is_connected(source):
            source = nx.gnm_random_graph(n, rng.randint(n - 1, 2 * n), seed=rng.randrange(10**9))
        others = nx.gnm_random_graph(6, rng.randint(5, 10), seed=rng.randrange(10**9))
        r = rng.choice([1, 2, 3])
        types = ruleset_from_graph(source, r).types + ruleset_from_graph(others, r).types
        compiled, report = compile_ruleset(Ruleset("mix", r, types))
        assert report.ok
        rulesets.append(compiled)
    total = 0
    for compiled in rulesets:
        result = baseline(compiled, max_n=7, root=tmp_path)
        expected = atlas_valid(compiled, 7)
        for n in range(1, 8):
            found = [g.to_networkx() for g in result.levels[n].valid]
            assert same_classes(found, expected.get(n, [])), (n, ruleset_key(compiled))
            for g in result.levels[n].valid:
                assert g.automorphisms == count_automorphisms(g.to_networkx())
            total += len(found)
    assert total >= 15, total


# --- Cache ------------------------------------------------------------------------------

@needs_geng
def test_second_run_reads_the_cache(rulesets_dir, tmp_path, monkeypatch):
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    first = baseline(compiled, max_n=9, root=tmp_path)
    assert sorted(p.name for p in (tmp_path / "geng").iterdir())[:2] == [
        "connected_n3_deg2-2.g6", "connected_n4_deg2-2.g6"
    ]

    def no_geng(*args, **kwargs):
        raise AssertionError("geng should not run again")

    monkeypatch.setattr(enumeration.subprocess, "run", no_geng)
    assert baseline(compiled, max_n=9, root=tmp_path) == first
    # A different ruleset with the same degree range reuses geng's lists too.
    cycles = ruleset_from_graph(nx.cycle_graph(7), 2)
    other, _ = compile_ruleset(cycles)
    assert ruleset_key(other) == ruleset_key(compiled)  # the same single type


@needs_geng
def test_results_are_read_back_without_geng_or_its_lists(rulesets_dir, tmp_path, monkeypatch):
    import shutil

    compiled, _ = load_and_compile(rulesets_dir / "showcase.json")
    first = baseline(compiled, max_n=8, root=tmp_path)
    shutil.rmtree(tmp_path / "geng")
    monkeypatch.setenv("GENG", "/nonexistent/geng")
    assert baseline(compiled, max_n=8, root=tmp_path) == first


def test_other_geng_failures_are_raised(rulesets_dir, tmp_path, monkeypatch):
    fake = tmp_path / "geng"
    fake.write_text("#!/bin/sh\necho '>E geng: out of memory' >&2\nexit 2\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GENG", str(fake))
    compiled, _ = load_and_compile(rulesets_dir / "path-ball-r2.json")
    with pytest.raises(RuntimeError, match="out of memory"):
        baseline(compiled, max_n=6, root=tmp_path / "cache")
    assert not list((tmp_path / "cache" / "geng").glob("*"))  # nothing half-written kept


def test_ruleset_key_ignores_order_labels_and_duplicates():
    a, _ = compile_ruleset(ruleset_from_graph(nx.petersen_graph(), 2))
    grid = ruleset_from_graph(nx.grid_2d_graph(4, 4), 2)
    b, _ = compile_ruleset(grid)
    shuffled = Ruleset("other name", 2, tuple(reversed(grid.types)) + grid.types[:1])
    c, _ = compile_ruleset(shuffled)
    assert ruleset_key(b) == ruleset_key(c)
    assert ruleset_key(a) != ruleset_key(b)


def test_missing_geng_is_explained(monkeypatch):
    monkeypatch.delenv("GENG", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GengNotFound, match="install nauty"):
        find_geng()
    monkeypatch.setenv("GENG", "/nonexistent/geng")
    with pytest.raises(GengNotFound, match="not an executable"):
        find_geng()


def test_counts_table(rulesets_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("GENG", "/nonexistent/geng")
    compiled, _ = load_and_compile(rulesets_dir / "friendship-paradox.json")
    text = str(baseline(compiled, max_n=2, root=tmp_path))
    assert "candidates" in text and Counter(text.split())["0"] >= 4
