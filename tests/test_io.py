import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from balltype import BallType
from report import IssueCode, RulesetError
from ruleset import Ruleset
from ruleset_io import (
    TOP_LEVEL_OPTIONAL,
    TOP_LEVEL_REQUIRED,
    TYPE_OPTIONAL,
    TYPE_REQUIRED,
    dumps_ruleset,
    load_ruleset,
    loads_ruleset,
    ruleset_from_dict,
    ruleset_to_dict,
    save_ruleset,
)

# Each fixture file has one problem. Loading it must report this code.
FIXTURE_CODES = [
    ("malformed.json", IssueCode.MALFORMED_JSON),
    ("not_an_object.json", IssueCode.BAD_VALUE),
    ("missing_format.json", IssueCode.MISSING_KEY),
    ("bad_format_version.json", IssueCode.BAD_FORMAT_VERSION),
    ("missing_r.json", IssueCode.MISSING_KEY),
    ("missing_types.json", IssueCode.MISSING_KEY),
    ("bad_r.json", IssueCode.BAD_VALUE),
    ("bool_r.json", IssueCode.BAD_VALUE),
    ("zero_r.json", IssueCode.BAD_VALUE),
    ("unknown_top_key.json", IssueCode.UNKNOWN_KEY),
    ("unknown_type_key.json", IssueCode.UNKNOWN_KEY),
    ("missing_edges.json", IssueCode.MISSING_KEY),
    ("bad_label.json", IssueCode.BAD_VALUE),
    ("bad_count.json", IssueCode.BAD_VALUE),
    ("malformed_edge.json", IssueCode.MALFORMED_EDGE),
    ("bool_node.json", IssueCode.MALFORMED_EDGE),
    ("negative_node.json", IssueCode.NEGATIVE_NODE),
    ("self_loop.json", IssueCode.SELF_LOOP),
    ("repeated_edge.json", IssueCode.REPEATED_EDGE),
    ("reversed_edge.json", IssueCode.REPEATED_EDGE),
    ("node_gap.json", IssueCode.NODE_GAP),
    ("disconnected.json", IssueCode.DISCONNECTED),
    ("beyond_radius.json", IssueCode.BEYOND_RADIUS),
]

# These load (report.ok) but must carry this warning.
WARNING_FIXTURES = [
    ("shallow.json", IssueCode.SHALLOW_TYPE),
    ("exact_duplicate.json", IssueCode.EXACT_DUPLICATE),
    ("empty_types.json", IssueCode.EMPTY_RULESET),
]

EXAMPLE_RULESETS = ["star-only.json", "path-ball-r2.json", "dimer-r1.json"]


# --- Errors -------------------------------------------------------------------

@pytest.mark.parametrize("filename, code", FIXTURE_CODES)
def test_fixture_reports_error(fixtures_dir, filename, code):
    """code in report.codes(), ruleset is None, report not ok."""
    ruleset, report = load_ruleset(fixtures_dir / filename)
    assert code in report.codes()
    assert ruleset is None
    assert not report.ok


@pytest.mark.parametrize("filename, code", FIXTURE_CODES)
def test_strict_raises(fixtures_dir, filename, code):
    """strict=True raises RulesetError whose report contains code."""
    with pytest.raises(RulesetError) as excinfo:
        load_ruleset(fixtures_dir / filename, strict=True)
    assert code in excinfo.value.report.codes()


def test_errors_collected_across_types(fixtures_dir):
    """two_bad_types.json reports SELF_LOOP at index 0 and NODE_GAP at index 2."""
    _, report = load_ruleset(fixtures_dir / "two_bad_types.json")
    assert {(i.code, i.type_index) for i in report.errors} == {
        (IssueCode.SELF_LOOP, 0),
        (IssueCode.NODE_GAP, 2),
    }


# --- Warnings -----------------------------------------------------------------

@pytest.mark.parametrize("filename, code", WARNING_FIXTURES)
def test_fixture_loads_with_warning(fixtures_dir, filename, code):
    """report.ok, ruleset is not None, code in report.codes()."""
    ruleset, report = load_ruleset(fixtures_dir / filename)
    assert report.ok
    assert ruleset is not None
    assert code in report.codes()


def test_shallow_type_is_kept(fixtures_dir):
    """shallow.json keeps both types; shallow_indices() == (1,)."""
    ruleset, _ = load_ruleset(fixtures_dir / "shallow.json")
    assert len(ruleset) == 2
    assert ruleset.shallow_indices() == (1,)


def test_exact_duplicates_are_kept_in_order(fixtures_dir):
    """exact_duplicate.json keeps both types: merging waits for compilation."""
    ruleset, _ = load_ruleset(fixtures_dir / "exact_duplicate.json")
    assert [t.label for t in ruleset] == ["a", "b"]
    assert ruleset.types[0].structure_key == ruleset.types[1].structure_key


# --- Clean loads ----------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_RULESETS)
def test_example_rulesets_load_cleanly(rulesets_dir, filename):
    """No issues at all, errors or warnings."""
    ruleset, report = load_ruleset(rulesets_dir / filename)
    assert report.issues == []
    assert ruleset is not None


def test_star_only_contents(rulesets_dir):
    """name "star-only", r 2, one type labelled "leaf-view", layers ((0,), (1,), (2, 3))."""
    ruleset, _ = load_ruleset(rulesets_dir / "star-only.json")
    assert ruleset.name == "star-only"
    assert ruleset.r == 2
    [ball] = ruleset.types
    assert ball.label == "leaf-view"
    assert ball.layers == ((0,), (1,), (2, 3))


# --- Round trip -----------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_RULESETS)
def test_round_trip_file(rulesets_dir, tmp_path, filename):
    """load -> save -> load gives an equal Ruleset."""
    original, _ = load_ruleset(rulesets_dir / filename)
    save_ruleset(original, tmp_path / filename)
    reloaded, report = load_ruleset(tmp_path / filename)
    assert report.ok
    assert reloaded == original


def test_round_trip_preserves_metadata_and_counts():
    """Metadata dict and per-type counts survive dumps -> loads unchanged."""
    metadata = {"source": "grid", "size": [4, 4], "notes": {"seed": 7}}
    ruleset = Ruleset(
        name="extracted",
        r=1,
        types=(
            BallType.from_edges([[0, 1], [0, 2]], count=12),
            BallType.from_edges([[0, 1]], label="edge", count=0),
        ),
        metadata=metadata,
    )
    reloaded, report = loads_ruleset(dumps_ruleset(ruleset))
    assert report.ok
    assert reloaded == ruleset
    assert reloaded.metadata == metadata
    assert [t.count for t in reloaded] == [12, 0]


def test_saved_file_has_no_canonical_keys(tmp_path):
    """The written JSON has only the documented keys at both levels."""
    ruleset = Ruleset(
        name="x",
        r=1,
        types=(BallType.from_edges([[0, 1]], label="a", count=3),),
        metadata={"k": "v"},
    )
    path = tmp_path / "out.json"
    save_ruleset(ruleset, path)
    doc = json.loads(path.read_text())
    assert set(doc) <= TOP_LEVEL_REQUIRED | TOP_LEVEL_OPTIONAL
    for entry in doc["types"]:
        assert set(entry) <= TYPE_REQUIRED | TYPE_OPTIONAL


@st.composite
def ball_types(draw):
    """A connected rooted graph: a random tree on 0..n-1 plus a few extra edges."""
    n = draw(st.integers(min_value=1, max_value=8))
    edges = {(draw(st.integers(0, v - 1)), v) for v in range(1, n)}
    if n >= 3:
        pairs = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1))
        extra = draw(st.lists(pairs.filter(lambda p: p[0] != p[1]), max_size=4))
        edges |= {tuple(sorted(p)) for p in extra}
    label = draw(st.none() | st.text(max_size=8))
    count = draw(st.none() | st.integers(min_value=0, max_value=1000))
    return BallType.from_edges(sorted(edges), label=label, count=count)


@given(types=st.lists(ball_types(), max_size=5), name=st.text(max_size=10))
def test_round_trip_property(types, name):
    """Hypothesis: random small connected rooted graphs, built in code, survive
    ruleset_to_dict -> ruleset_from_dict unchanged."""
    # r at least every depth, so nothing is BEYOND_RADIUS and the ruleset loads.
    r = max([1] + [t.depth for t in types])
    ruleset = Ruleset(name=name, r=r, types=tuple(types))
    reloaded, report = ruleset_from_dict(ruleset_to_dict(ruleset))
    assert report.ok, str(report)
    assert reloaded == ruleset


# --- Waiting on later steps -------------------------------------------------------

@pytest.mark.skip(reason="needs step 2: canonical form")
def test_isomorphic_duplicate_detected(fixtures_dir):
    """isomorphic_duplicate.json: compiling gives ISOMORPHIC_DUPLICATE and one merged type."""
