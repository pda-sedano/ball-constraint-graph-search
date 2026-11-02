"""Issues found while loading or checking a ruleset, collected into a report.

Loading never stops at the first problem. Every check appends an ``Issue`` to a
``LoadReport``, and the caller decides what to do with it. ``strict`` loading
calls ``LoadReport.raise_if_errors``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """How serious an issue is.

    ERROR: the ruleset is not usable as written. Loading returns no ruleset.
    WARNING: the ruleset loads, but something is probably not what was meant.
    """

    ERROR = "error"
    WARNING = "warning"


class IssueCode(Enum):
    """Stable identifiers for every kind of issue. Tests assert on these, not on messages.

    Each code has one fixed severity, given by ``SEVERITY`` below.
    """

    # Document level
    MALFORMED_JSON = "malformed_json"
    MISSING_KEY = "missing_key"
    UNKNOWN_KEY = "unknown_key"
    BAD_FORMAT_VERSION = "bad_format_version"
    BAD_VALUE = "bad_value"  # wrong JSON type or out-of-range value (r, label, count, ...)
    EMPTY_RULESET = "empty_ruleset"

    # Single type, structural (checked by balltype.check_edge_list)
    MALFORMED_EDGE = "malformed_edge"  # not a pair of non-bool ints
    NEGATIVE_NODE = "negative_node"
    SELF_LOOP = "self_loop"
    REPEATED_EDGE = "repeated_edge"  # includes [1, 0] after [0, 1]
    NODE_GAP = "node_gap"  # nodes are not exactly 0..k-1
    DISCONNECTED = "disconnected"

    # Single type, relative to r (checked by ruleset.check_ruleset)
    BEYOND_RADIUS = "beyond_radius"
    SHALLOW_TYPE = "shallow_type"

    # Across types
    EXACT_DUPLICATE = "exact_duplicate"  # identical normalized edge lists
    ISOMORPHIC_DUPLICATE = "isomorphic_duplicate"  # step 2, CompiledRuleset


SEVERITY: dict[IssueCode, Severity] = {
    IssueCode.MALFORMED_JSON: Severity.ERROR,
    IssueCode.MISSING_KEY: Severity.ERROR,
    IssueCode.UNKNOWN_KEY: Severity.ERROR,
    IssueCode.BAD_FORMAT_VERSION: Severity.ERROR,
    IssueCode.BAD_VALUE: Severity.ERROR,
    IssueCode.EMPTY_RULESET: Severity.WARNING,
    IssueCode.MALFORMED_EDGE: Severity.ERROR,
    IssueCode.NEGATIVE_NODE: Severity.ERROR,
    IssueCode.SELF_LOOP: Severity.ERROR,
    IssueCode.REPEATED_EDGE: Severity.ERROR,
    IssueCode.NODE_GAP: Severity.ERROR,
    IssueCode.DISCONNECTED: Severity.ERROR,
    IssueCode.BEYOND_RADIUS: Severity.ERROR,
    IssueCode.SHALLOW_TYPE: Severity.WARNING,
    IssueCode.EXACT_DUPLICATE: Severity.WARNING,
    IssueCode.ISOMORPHIC_DUPLICATE: Severity.WARNING,
}


@dataclass(frozen=True)
class Issue:
    """One problem found during loading or checking.

    Attributes:
        code: What kind of problem this is.
        message: Human-readable detail, e.g. which edge or node.
        type_index: Position of the offending type in the file's ``types`` list,
            or None for document-level issues.
    """

    code: IssueCode
    message: str
    type_index: int | None = None

    @property
    def severity(self) -> Severity:
        """Severity looked up from ``SEVERITY``."""
        raise NotImplementedError

    def __str__(self) -> str:
        """One line, e.g. ``error [self_loop] type 2: node 1 is adjacent to itself``."""
        raise NotImplementedError


@dataclass
class LoadReport:
    """Every issue found while loading or checking one ruleset, in the order found.

    Mutable on purpose: checks append to it as they run.
    """

    issues: list[Issue] = field(default_factory=list)

    def add(self, code: IssueCode, message: str, type_index: int | None = None) -> None:
        """Append a new issue."""
        raise NotImplementedError

    def extend(self, issues: list[Issue]) -> None:
        """Append issues produced elsewhere, e.g. by ``check_edge_list``."""
        raise NotImplementedError

    @property
    def errors(self) -> list[Issue]:
        """Issues with severity ERROR."""
        raise NotImplementedError

    @property
    def warnings(self) -> list[Issue]:
        """Issues with severity WARNING."""
        raise NotImplementedError

    @property
    def ok(self) -> bool:
        """True when there are no errors. Warnings do not count."""
        raise NotImplementedError

    def codes(self) -> set[IssueCode]:
        """The set of codes present. Convenient for tests."""
        raise NotImplementedError

    def raise_if_errors(self) -> None:
        """Raise ``RulesetError`` carrying this report if there are any errors."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Multi-line summary: counts first, then one line per issue."""
        raise NotImplementedError


class RulesetError(ValueError):
    """Raised by strict loading. Carries the full report, not just the first error."""

    def __init__(self, report: LoadReport) -> None:
        raise NotImplementedError
