import pytest

from ballsearch.report import IssueCode

TODO = pytest.mark.skip(reason="not written yet")

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

@TODO
@pytest.mark.parametrize("filename, code", FIXTURE_CODES)
def test_fixture_reports_error(fixtures_dir, filename, code):
    """code in report.codes(), ruleset is None, report not ok."""


@TODO
@pytest.mark.parametrize("filename, code", FIXTURE_CODES)
def test_strict_raises(fixtures_dir, filename, code):
    """strict=True raises RulesetError whose report contains code."""


@TODO
def test_errors_collected_across_types(fixtures_dir):
    """two_bad_types.json reports SELF_LOOP at index 0 and NODE_GAP at index 2."""


# --- Warnings -----------------------------------------------------------------

@TODO
@pytest.mark.parametrize("filename, code", WARNING_FIXTURES)
def test_fixture_loads_with_warning(fixtures_dir, filename, code):
    """report.ok, ruleset is not None, code in report.codes()."""


@TODO
def test_shallow_type_is_kept(fixtures_dir):
    """shallow.json keeps both types; shallow_indices() == (1,)."""


@TODO
def test_exact_duplicates_are_kept_in_order(fixtures_dir):
    """exact_duplicate.json keeps both types: merging waits for compilation."""


# --- Clean loads ----------------------------------------------------------------

@TODO
@pytest.mark.parametrize("filename", EXAMPLE_RULESETS)
def test_example_rulesets_load_cleanly(rulesets_dir, filename):
    """No issues at all, errors or warnings."""


@TODO
def test_star_only_contents(rulesets_dir):
    """name "star-only", r 2, one type labelled "leaf-view", layers ((0,), (1,), (2, 3))."""


# --- Round trip -----------------------------------------------------------------

@TODO
@pytest.mark.parametrize("filename", EXAMPLE_RULESETS)
def test_round_trip_file(rulesets_dir, tmp_path, filename):
    """load -> save -> load gives an equal Ruleset."""


@TODO
def test_round_trip_preserves_metadata_and_counts():
    """Metadata dict and per-type counts survive dumps -> loads unchanged."""


@TODO
def test_saved_file_has_no_canonical_keys(tmp_path):
    """The written JSON has only the documented keys at both levels."""


@TODO
def test_round_trip_property():
    """Hypothesis: random small connected rooted graphs, built in code, survive
    ruleset_to_dict -> ruleset_from_dict unchanged."""


# --- Waiting on later steps -------------------------------------------------------

@pytest.mark.skip(reason="needs step 2: canonical form")
def test_isomorphic_duplicate_detected(fixtures_dir):
    """isomorphic_duplicate.json: compiling gives ISOMORPHIC_DUPLICATE and one merged type."""
