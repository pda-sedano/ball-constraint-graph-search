"""A ruleset as written: radius r plus a list of allowed ball types.

This object holds no canonical keys and does no pruning. Those belong to a
``CompiledRuleset`` built from it in steps 2 and 4, which keeps both the list
as written and the pruned list.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .balltype import BallType
from .report import LoadReport


@dataclass(frozen=True)
class Ruleset:
    """Radius r and an ordered tuple of ball types, in the order written.

    Construct directly in code (enumeration, extraction) or via ``io.load_ruleset``.
    Construction does not validate against r; call ``check_ruleset``.

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
        raise NotImplementedError

    def __iter__(self) -> Iterator[BallType]:
        raise NotImplementedError

    def is_shallow(self, ball: BallType) -> bool:
        """True if ``ball.depth < self.r``.

        Shallow types are kept: one applies only when n equals its size, which
        is not known until search time.
        """
        raise NotImplementedError

    def shallow_indices(self) -> tuple[int, ...]:
        """Indices of shallow types, in order."""
        raise NotImplementedError

    @property
    def max_root_degree(self) -> int:
        """Largest root degree over all types. The sampler's hard degree cap."""
        raise NotImplementedError


def check_ruleset(ruleset: Ruleset, report: LoadReport | None = None) -> LoadReport:
    """Checks that need r or compare types with each other.

    - ``r`` is an int of at least 1, not a bool (BAD_VALUE).
    - No types (EMPTY_RULESET, warning).
    - Each type: depth greater than r (BEYOND_RADIUS) or less than r (SHALLOW_TYPE).
    - Pairs of types with the same ``structure_key`` (EXACT_DUPLICATE). Report
      the later index and name the earlier one in the message. Both are kept;
      merging happens on compilation, along with isomorphic duplicates.

    Args:
        ruleset: The ruleset to check. Its types are already structurally valid.
        report: Report to append to. A new one is created if None.

    Returns:
        The report that was appended to.
    """
    raise NotImplementedError
