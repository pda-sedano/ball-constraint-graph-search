import pytest

from balltype import BallType
from report import IssueCode, LoadReport, Severity
from ruleset import Ruleset, check_ruleset

EDGE = BallType.from_edges([[0, 1]])  # depth 1
STAR = BallType.from_edges([[0, 1], [1, 2], [1, 3]])  # depth 2
PATH3 = BallType.from_edges([[0, 1], [1, 2], [2, 3]])  # depth 3


def make_ruleset(*types: BallType, r: int = 2) -> Ruleset:
    return Ruleset(name="test", r=r, types=types)


def test_empty_ruleset_warns():
    """No types gives EMPTY_RULESET as a warning."""
    report = check_ruleset(make_ruleset())
    [issue] = report.issues
    assert issue.code is IssueCode.EMPTY_RULESET
    assert issue.severity is Severity.WARNING
    assert report.ok


@pytest.mark.parametrize("r", [0, -1, True, "2"])
def test_bad_r(r):
    """r below 1, a bool, or not an int gives BAD_VALUE."""
    report = check_ruleset(make_ruleset(STAR, r=r))
    assert IssueCode.BAD_VALUE in report.codes()


def test_graph_checks_are_left_to_rust():
    """Too deep, too shallow and duplicated types raise nothing here; compiling reports them."""
    report = check_ruleset(make_ruleset(STAR, PATH3, EDGE, STAR))
    assert report.issues == []


def test_check_ruleset_appends_to_given_report():
    """Passing an existing report appends to it and returns the same object."""
    report = LoadReport()
    report.add(IssueCode.UNKNOWN_KEY, "earlier issue")
    returned = check_ruleset(make_ruleset(), report)
    assert returned is report
    assert [i.code for i in report.issues] == [IssueCode.UNKNOWN_KEY, IssueCode.EMPTY_RULESET]


def test_iteration_and_length():
    ruleset = make_ruleset(STAR, EDGE)
    assert len(ruleset) == 2
    assert list(ruleset) == [STAR, EDGE]
