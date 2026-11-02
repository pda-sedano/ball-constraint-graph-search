"""Layered drawings of ball types, to confirm a file says what was meant.

Optional: needs the ``draw`` extra (matplotlib). Import matplotlib inside the
functions so the rest of the package never depends on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .balltype import BallType
from .ruleset import Ruleset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def layered_positions(ball: BallType) -> dict[int, tuple[float, float]]:
    """Node positions with the root on top and one row per distance layer.

    Nodes within a row are spread evenly and centered. Pure function, no
    matplotlib, so it can be tested on its own.
    """
    raise NotImplementedError


def draw_ball(ball: BallType, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Draw one ball with ``layered_positions``. Highlight the root; title with the label."""
    raise NotImplementedError


def draw_ruleset(ruleset: Ruleset, ncols: int = 4) -> Figure:
    """Grid of ``draw_ball`` panels, one per type, shallow types marked."""
    raise NotImplementedError
