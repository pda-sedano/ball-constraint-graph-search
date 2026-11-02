import pytest

from balltype import BallType
from report import IssueCode, LoadReport, Severity
from ruleset import Ruleset, check_ruleset

EDGE = BallType.from_edges([[0, 1]])  # depth 1
STAR = BallType.from_edges([[0, 1], [1, 2], [1, 3]])  # depth 2
PATH3 = BallType.from_edges([[0, 1], [1, 2], [2, 3]])  # depth 3


def make_ruleset(*types: BallType, r: int = 2) -> Ruleset:
    return Ruleset(name="test", r=r, types=types)


def issues_with(report: LoadReport, code: IssueCode):
    return [i for i in report.issues if i.code is code]


def test_beyond_radius_is_error():
    """r = 2 with a path of length 3 from the root gives BEYOND_RADIUS."""
    report = check_ruleset(make_ruleset(STAR, PATH3))
    [issue] = issues_with(report, IssueCode.BEYOND_RADIUS)
    assert issue.type_index == 1
    assert issue.severity is Severity.ERROR
    assert not report.ok


def test_shallow_is_warning_and_kept():
    """r = 2 with the single-edge ball: SHALLOW_TYPE warning, report.ok, type still present."""
    ruleset = make_ruleset(EDGE)
    report = check_ruleset(ruleset)
    [issue] = issues_with(report, IssueCode.SHALLOW_TYPE)
    assert issue.type_index == 0
    assert issue.severity is Severity.WARNING
    assert report.ok
    assert list(ruleset) == [EDGE]


def test_shallow_indices():
    """shallow_indices lists exactly the types with depth < r, in order."""
    single = BallType.from_edges([])  # depth 0
    ruleset = make_ruleset(STAR, EDGE, PATH3, single, STAR, r=2)
    assert ruleset.shallow_indices() == (1, 3)
    assert [ruleset.is_shallow(t) for t in ruleset] == [False, True, False, True, False]


def test_exact_duplicate_warns_once_per_later_copy():
    """Three identical types give two EXACT_DUPLICATE warnings, at indices 1 and 2."""
    report = check_ruleset(make_ruleset(STAR, STAR, STAR))
    duplicates = issues_with(report, IssueCode.EXACT_DUPLICATE)
    assert [i.type_index for i in duplicates] == [1, 2]
    assert all(i.severity is Severity.WARNING for i in duplicates)


def test_exact_duplicate_ignores_labels():
    """Same structure with different labels still counts as a duplicate."""
    a = BallType.from_edges([[0, 1], [1, 2], [1, 3]], label="a")
    b = BallType.from_edges([[0, 1], [1, 2], [1, 3]], label="b", count=4)
    report = check_ruleset(make_ruleset(a, b))
    assert [i.type_index for i in issues_with(report, IssueCode.EXACT_DUPLICATE)] == [1]


def test_empty_ruleset_warns():
    """No types gives EMPTY_RULESET as a warning."""
    report = check_ruleset(make_ruleset())
    assert IssueCode.EMPTY_RULESET in report.codes()
    assert report.ok


@pytest.mark.parametrize("r", [0, -1, True])
def test_bad_r(r):
    """r below 1, or a bool, gives BAD_VALUE."""
    report = check_ruleset(make_ruleset(STAR, r=r))
    assert IssueCode.BAD_VALUE in report.codes()


def test_max_root_degree():
    """Largest root degree across types; used as the sampler's degree cap."""
    hub = BallType.from_edges([[0, 1], [0, 2], [0, 3], [1, 4]])  # root degree 3
    assert make_ruleset(STAR, hub, EDGE).max_root_degree == 3


def test_check_ruleset_appends_to_given_report():
    """Passing an existing report appends to it and returns the same object."""
    report = LoadReport()
    report.add(IssueCode.UNKNOWN_KEY, "earlier issue")
    returned = check_ruleset(make_ruleset(EDGE), report)
    assert returned is report
    assert report.issues[0].code is IssueCode.UNKNOWN_KEY
    assert IssueCode.SHALLOW_TYPE in report.codes()
