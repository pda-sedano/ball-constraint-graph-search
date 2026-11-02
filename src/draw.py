"""Layered drawings of ball types, to confirm a file says what was meant.

Optional: needs the ``draw`` extra (matplotlib). Import matplotlib inside the
functions so the rest of the package never depends on it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from balltype import BallType
from ruleset import Ruleset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def layered_positions(ball: BallType) -> dict[int, tuple[float, float]]:
    """Node positions with the root on top and one row per distance layer.

    Nodes within a row are spread evenly and centered. Pure function, no
    matplotlib, so it can be tested on its own.
    """
    positions: dict[int, tuple[float, float]] = {}
    for depth, layer in enumerate(ball.layers):
        # Unit spacing, centered on x = 0; row d sits at y = -d.
        offset = (len(layer) - 1) / 2
        for i, node in enumerate(layer):
            positions[node] = (i - offset, -float(depth))
    return positions


def draw_ball(ball: BallType, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Draw one ball with ``layered_positions``. Highlight the root; title with the label."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    graph = ball.to_networkx()
    options: dict[str, Any] = {
        "node_color": ["tab:red" if v == 0 else "tab:blue" for v in graph.nodes],
        "font_color": "white",
        "node_size": 400,
    }
    options.update(kwargs)
    nx.draw_networkx(graph, layered_positions(ball), ax=ax, **options)
    if ball.label is not None:
        ax.set_title(ball.label)
    ax.set_axis_off()
    return ax


def draw_ruleset(ruleset: Ruleset, ncols: int = 4) -> Figure:
    """Grid of ``draw_ball`` panels, one per type, shallow types marked."""
    import matplotlib.pyplot as plt

    if ncols < 1:
        raise ValueError(f"ncols must be at least 1, got {ncols}")
    n = len(ruleset)
    ncols = min(ncols, max(n, 1))
    nrows = max(math.ceil(n / ncols), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    shallow = set(ruleset.shallow_indices())
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_axis_off()
            continue
        ball = ruleset.types[i]
        draw_ball(ball, ax)
        title = f"{i}: {ball.label}" if ball.label is not None else f"type {i}"
        if i in shallow:
            ax.set_title(f"{title}\nshallow: depth {ball.depth} < r", color="tab:orange")
        else:
            ax.set_title(title)

    fig.suptitle(f"{ruleset.name} (r = {ruleset.r})")
    fig.tight_layout()
    return fig
