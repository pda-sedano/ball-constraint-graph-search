"""The Python side of compiling: the bridge in compiled.py and the Rust extension through it.

Skipped as a whole when the extension is not built (see rust/README.md).
"""

import pytest

graph_search_rs = pytest.importorskip("graph_search_rs")

from balltype import BallType  # noqa: E402
from compiled import compile_ruleset, load_and_compile, to_issue  # noqa: E402
from report import (  # noqa: E402
    COMPILE_CODES,
    SEVERITY,
    Issue,
    IssueCode,
    LoadReport,
    RulesetError,
)
from ruleset import Ruleset  # noqa: E402


def issue_pairs(report):
    return [(issue.code, issue.type_index) for issue in report.issues]


# --- Parity between Rust and Python -------------------------------------------

def test_issue_codes_match_python():
    """Rust's (code, severity) pairs are exactly Python's compile codes and severities."""
    expected = {(code.value, SEVERITY[code].value) for code in COMPILE_CODES}
    assert set(graph_search_rs.issue_codes()) == expected


def test_to_issue_converts_tuple():
    assert to_issue(("self_loop", "msg", 2)) == Issue(IssueCode.SELF_LOOP, "msg", 2)


def test_to_issue_rejects_unknown_code():
    with pytest.raises(ValueError):
        to_issue(("not_a_code", "msg", None))


# --- Compiling ----------------------------------------------------------------

def test_star_only_compiles_cleanly(rulesets_dir):
    compiled, report = load_and_compile(rulesets_dir / "star-only.json")
    assert report.issues == []
    assert compiled is not None
    assert compiled.r == 2
    assert len(compiled) == 1


def test_star_only_layers_and_degrees(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "star-only.json")
    assert compiled.layers(0) == [[0], [1], [2, 3]]
    assert compiled.degrees(0) == [1, 3, 1, 1]
    assert compiled.depth(0) == 2
    assert compiled.edges(0) == [(0, 1), (1, 2), (1, 3)]
    assert compiled.max_root_degree == 1


def test_errors_give_no_compiled_ruleset(fixtures_dir):
    compiled, report = load_and_compile(fixtures_dir / "beyond_radius.json")
    assert IssueCode.BEYOND_RADIUS in report.codes()
    assert compiled is None
    assert not report.ok


def test_all_issues_reported_at_once(fixtures_dir):
    _, report = load_and_compile(fixtures_dir / "two_bad_types.json")
    assert {(i.code, i.type_index) for i in report.errors} == {
        (IssueCode.SELF_LOOP, 0),
        (IssueCode.NODE_GAP, 2),
    }


def test_exact_duplicates_merged(fixtures_dir):
    compiled, report = load_and_compile(fixtures_dir / "exact_duplicate.json")
    assert issue_pairs(report) == [(IssueCode.EXACT_DUPLICATE, 1)]
    assert len(compiled) == 1
    assert compiled.source_indices == [0]


def test_shallow_types_kept_and_listed(fixtures_dir):
    compiled, report = load_and_compile(fixtures_dir / "shallow.json")
    assert issue_pairs(report) == [(IssueCode.SHALLOW_TYPE, 1)]
    assert len(compiled) == 2
    assert compiled.shallow_indices() == [1]


def test_index_out_of_range_raises(rulesets_dir):
    compiled, _ = load_and_compile(rulesets_dir / "star-only.json")
    with pytest.raises(IndexError):
        compiled.layers(len(compiled))


# --- The bridge's own behaviour -------------------------------------------------

def test_bad_r_raises_before_rust():
    ruleset = Ruleset(name="x", r=0, types=(BallType.from_edges([[0, 1]]),))
    with pytest.raises(ValueError, match="r must be"):
        compile_ruleset(ruleset)


def test_issues_appended_to_given_report():
    """Passing the loader's report appends Rust's issues after the existing ones."""
    report = LoadReport()
    report.add(IssueCode.EMPTY_RULESET, "earlier issue")
    ruleset = Ruleset(name="x", r=2, types=(BallType.from_edges([[0, 1]]),))
    _, returned = compile_ruleset(ruleset, report)
    assert returned is report
    assert issue_pairs(report) == [(IssueCode.EMPTY_RULESET, None), (IssueCode.SHALLOW_TYPE, 0)]


def test_load_errors_skip_compiling(fixtures_dir):
    """A file with format errors returns only the loader's issues, without calling Rust."""
    compiled, report = load_and_compile(fixtures_dir / "malformed_edge.json")
    assert compiled is None
    assert report.codes() == {IssueCode.MALFORMED_EDGE}


def test_strict_raises_on_rust_errors(fixtures_dir):
    with pytest.raises(RulesetError) as excinfo:
        load_and_compile(fixtures_dir / "self_loop.json", strict=True)
    assert IssueCode.SELF_LOOP in excinfo.value.report.codes()
