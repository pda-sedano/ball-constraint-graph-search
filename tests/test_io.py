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

# Each fixture file has one format problem. Loading it in Python must report this code.
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
]

# These load in Python (report.ok) but must carry this warning.
WARNING_FIXTURES = [
    ("empty_types.json", IssueCode.EMPTY_RULESET),
]

# Graph problems: well-formed files, so Python loads them without issues, and
# Rust's compile must report this code. Kept here so step 2 turns them into
# end-to-end tests by filling in test_compile_fixture_reports.
COMPILE_FIXTURES = [
    ("self_loop.json", IssueCode.SELF_LOOP),
    ("repeated_edge.json", IssueCode.REPEATED_EDGE),
    ("reversed_edge.json", IssueCode.REPEATED_EDGE),
    ("node_gap.json", IssueCode.NODE_GAP),
    ("disconnected.json", IssueCode.DISCONNECTED),
    ("beyond_radius.json", IssueCode.BEYOND_RADIUS),
    ("shallow.json", IssueCode.SHALLOW_TYPE),
    ("exact_duplicate.json", IssueCode.EXACT_DUPLICATE),
    ("isomorphic_duplicate.json", IssueCode.ISOMORPHIC_DUPLICATE),
    ("two_bad_types.json", IssueCode.SELF_LOOP),  # and NODE_GAP at index 2
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


def test_errors_collected_across_types():
    """Format errors in two different types are both reported, each with its index."""
    ok = {"edges": [[0, 1]]}
    doc = {
        "format": 1,
        "name": "x",
        "r": 1,
        "types": [{"edges": [[0, 1], [1]]}, ok, {"edges": [[0, -1]]}],
    }
    ruleset, report = ruleset_from_dict(doc)
    assert ruleset is None
    assert {(i.code, i.type_index) for i in report.errors} == {
        (IssueCode.MALFORMED_EDGE, 0),
        (IssueCode.NEGATIVE_NODE, 2),
    }


# --- Warnings -----------------------------------------------------------------

@pytest.mark.parametrize("filename, code", WARNING_FIXTURES)
def test_fixture_loads_with_warning(fixtures_dir, filename, code):
    """report.ok, ruleset is not None, code in report.codes()."""
    ruleset, report = load_ruleset(fixtures_dir / filename)
    assert report.ok
    assert ruleset is not None
    assert code in report.codes()


# --- Graph problems: Python loads them, Rust rejects them ---------------------

@pytest.mark.parametrize("filename, code", COMPILE_FIXTURES)
def test_compile_fixture_loads_in_python(fixtures_dir, filename, code):
    """Well-formed files load with no issues: the graph checks live in Rust."""
    ruleset, report = load_ruleset(fixtures_dir / filename)
    assert report.issues == []
    assert ruleset is not None


@pytest.mark.skip(reason="needs step 2: Rust compile")
@pytest.mark.parametrize("filename, code", COMPILE_FIXTURES)
def test_compile_fixture_reports(fixtures_dir, filename, code):
    """Loading then compiling reports code; errors give no compiled ruleset."""


def test_types_kept_as_written(fixtures_dir):
    """exact_duplicate.json keeps both types, in order and as written."""
    ruleset, _ = load_ruleset(fixtures_dir / "exact_duplicate.json")
    assert [t.label for t in ruleset] == ["a", "b"]
    assert [t.edges for t in ruleset] == [((0, 1), (1, 2)), ((2, 1), (0, 1))]


# --- Clean loads ----------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_RULESETS)
def test_example_rulesets_load_cleanly(rulesets_dir, filename):
    """No issues at all, errors or warnings."""
    ruleset, report = load_ruleset(rulesets_dir / filename)
    assert report.issues == []
    assert ruleset is not None


def test_star_only_contents(rulesets_dir):
    """name "star-only", r 2, one type labelled "leaf-view" with the doc's edges."""
    ruleset, _ = load_ruleset(rulesets_dir / "star-only.json")
    assert ruleset.name == "star-only"
    assert ruleset.r == 2
    [ball] = ruleset.types
    assert ball.label == "leaf-view"
    assert ball.edges == ((0, 1), (1, 2), (1, 3))


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
    """Any well-formed edge list: pairs of non-negative ints, in any order.

    Graph validity does not matter here, since Python only checks the format.
    """
    node = st.integers(min_value=0, max_value=12)
    edges = draw(st.lists(st.tuples(node, node), max_size=10))
    label = draw(st.none() | st.text(max_size=8))
    count = draw(st.none() | st.integers(min_value=0, max_value=1000))
    return BallType.from_edges(edges, label=label, count=count)


@given(
    types=st.lists(ball_types(), min_size=1, max_size=5),
    name=st.text(max_size=10),
    r=st.integers(min_value=1, max_value=5),
)
def test_round_trip_property(types, name, r):
    """Hypothesis: well-formed rulesets built in code survive
    ruleset_to_dict -> ruleset_from_dict unchanged, edges as written."""
    ruleset = Ruleset(name=name, r=r, types=tuple(types))
    reloaded, report = ruleset_from_dict(ruleset_to_dict(ruleset))
    assert report.issues == [], str(report)
    assert reloaded == ruleset
