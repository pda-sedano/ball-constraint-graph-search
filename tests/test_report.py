import pytest

from ballsearch.report import SEVERITY, IssueCode

TODO = pytest.mark.skip(reason="not written yet")


def test_every_code_has_a_severity():
    """Guards against adding an IssueCode and forgetting SEVERITY."""
    assert set(SEVERITY) == set(IssueCode)


@TODO
def test_issue_severity_comes_from_table():
    """Issue(code).severity matches SEVERITY[code]."""


@TODO
def test_errors_and_warnings_partition_issues():
    """Every issue lands in exactly one of report.errors / report.warnings."""


@TODO
def test_ok_ignores_warnings():
    """A report with only warnings is ok; one error makes it not ok."""


@TODO
def test_raise_if_errors_carries_full_report():
    """RulesetError.report holds every issue, not just the first error."""


@TODO
def test_raise_if_errors_silent_on_warnings_only():
    """No exception when the report has warnings but no errors."""
