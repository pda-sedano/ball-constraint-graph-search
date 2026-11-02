import pytest

from report import SEVERITY, Issue, IssueCode, LoadReport, RulesetError, Severity

ERROR_CODE = IssueCode.SELF_LOOP
WARNING_CODE = IssueCode.SHALLOW_TYPE


def make_report(*codes: IssueCode) -> LoadReport:
    report = LoadReport()
    for i, code in enumerate(codes):
        report.add(code, f"issue {i}", type_index=i)
    return report


def test_every_code_has_a_severity():
    """Guards against adding an IssueCode and forgetting SEVERITY."""
    assert set(SEVERITY) == set(IssueCode)


@pytest.mark.parametrize("code", list(IssueCode))
def test_issue_severity_comes_from_table(code):
    """Issue(code).severity matches SEVERITY[code]."""
    assert Issue(code, "msg").severity == SEVERITY[code]


def test_errors_and_warnings_partition_issues():
    """Every issue lands in exactly one of report.errors / report.warnings."""
    report = make_report(*IssueCode)
    assert len(report.errors) + len(report.warnings) == len(report.issues)
    assert all(i.severity is Severity.ERROR for i in report.errors)
    assert all(i.severity is Severity.WARNING for i in report.warnings)
    assert {id(i) for i in report.errors}.isdisjoint(id(i) for i in report.warnings)


def test_ok_ignores_warnings():
    """A report with only warnings is ok; one error makes it not ok."""
    assert LoadReport().ok
    report = make_report(WARNING_CODE, IssueCode.EXACT_DUPLICATE)
    assert report.ok
    report.add(ERROR_CODE, "an error")
    assert not report.ok


def test_raise_if_errors_carries_full_report():
    """RulesetError.report holds every issue, not just the first error."""
    report = make_report(ERROR_CODE, WARNING_CODE, IssueCode.NODE_GAP)
    with pytest.raises(RulesetError) as excinfo:
        report.raise_if_errors()
    assert excinfo.value.report.issues == report.issues
    assert len(excinfo.value.report.errors) == 2


def test_raise_if_errors_silent_on_warnings_only():
    """No exception when the report has warnings but no errors."""
    make_report(WARNING_CODE, IssueCode.EXACT_DUPLICATE).raise_if_errors()
