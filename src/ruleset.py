"""A ruleset as written: radius r plus a list of allowed ball types.

This object holds no canonical keys, does no pruning, and checks nothing about
the ball graphs. All of that happens in Rust when the ruleset is compiled,
which keeps both the list as written and the pruned list.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from balltype import BallType
from report import IssueCode, LoadReport


@dataclass(frozen=True)
class Ruleset:
    """Radius r and an ordered tuple of ball types, in the order written.

    Construct directly in code or via ``ruleset_io.load_ruleset``.
    Construction does not validate r; call ``check_ruleset``.

    Note: ``metadata`` is a dict, so instances compare with ``==`` but are not
    hashable. Hashing belongs on the compiled ruleset if it is ever needed.

    Attributes:
        name: Human-readable name.
        r: Ball radius, at least 1.
        types: Allowed ball types. Order is preserved through save and load.
        metadata: Free-form JSON-compatible data, e.g. the source graph of an
            extracted ruleset. Saved and loaded unchanged.
    """

    name: str
    r: int
    types: tuple[BallType, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.types)

    def __iter__(self) -> Iterator[BallType]:
        return iter(self.types)


def check_ruleset(ruleset: Ruleset, report: LoadReport | None = None) -> LoadReport:
    """Checks on the ruleset's own fields, not on its ball graphs.

    - ``r`` is an int of at least 1, not a bool (BAD_VALUE).
    - No types (EMPTY_RULESET, warning).

    Depth against r and duplicate types are checked in Rust on compiling.

    Args:
        ruleset: The ruleset to check.
        report: Report to append to. A new one is created if None.

    Returns:
        The report that was appended to.
    """
    if report is None:
        report = LoadReport()

    r = ruleset.r
    if not (isinstance(r, int) and not isinstance(r, bool) and r >= 1):
        report.add(IssueCode.BAD_VALUE, f"r must be an int of at least 1, got {r!r}")

    if not ruleset.types:
        report.add(IssueCode.EMPTY_RULESET, "ruleset has no types")

    return report
