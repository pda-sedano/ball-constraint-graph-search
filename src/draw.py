"""Layered drawings of ball types, to confirm a file says what was meant.

Optional: needs the ``draw`` extra (matplotlib). Import matplotlib inside the
functions so the rest of the package never depends on it.

The layout computes its own distance layers rather than relying on a compiled
ruleset, so it can draw a ruleset that has not been compiled, including one
that would fail compiling. That is when a drawing helps most. Problems are
shown, not reported: nodes the root cannot reach are drawn in gray, and panel
titles carry notes. Compiling is what reports them as issues.
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

ROOT_COLOR = "tab:red"
REACHED_COLOR = "tab:blue"
UNREACHED_COLOR = "tab:gray"

COMPONENT_GAP = 1.0  # extra horizontal space between components, in node spacings
PADDING = 0.5  # space around the outermost nodes, in node spacings
SPACING_INCHES = 0.6  # one node spacing on the page, for draw_ruleset's figure size
TITLE_INCHES = 0.7  # room above each panel for up to three title lines
MIN_PANEL_INCHES = 1.5


def layout_components(ball: BallType) -> list[list[list[int]]]:
    """Each connected component as distance layers from its anchor, for drawing.

    A component's anchor is its lowest-numbered node, which is the root (node 0)
    for the root's component. Components are ordered by anchor, so the root's
    component comes first whenever the ball has a root.

    Returns:
        ``components[c][d]`` is the sorted list of nodes at distance d from
        component c's anchor.
    """
    graph = ball.to_networkx()
    anchors = sorted(min(component) for component in nx.connected_components(graph))
    return [[sorted(layer) for layer in nx.bfs_layers(graph, a)] for a in anchors]


def layered_positions(ball: BallType) -> dict[int, tuple[float, float]]:
    """Node positions: each component drawn as layers below its anchor, side by side.

    Layer d of every component sits at y = -d, so all anchors share the top row.
    Within a row, nodes are one unit apart and centered under the anchor.
    Components are placed left to right with ``COMPONENT_GAP`` between them.
    Pure function, no matplotlib, so it can be tested on its own.
    """
    positions: dict[int, tuple[float, float]] = {}
    left = 0.0
    for layers in layout_components(ball):
        width = max(len(row) for row in layers) - 1
        center = left + width / 2
        for depth, row in enumerate(layers):
            offset = (len(row) - 1) / 2
            for i, node in enumerate(row):
                positions[node] = (center + i - offset, -float(depth))
        left += width + 1 + COMPONENT_GAP
    return positions


def panel_notes(ball: BallType, r: int) -> list[str]:
    """Notes for a panel title, one per problem the drawing shows. Empty when there are none.

    - No node 0: ``"no root (node 0)"``. Depth is undefined without a root.
    - Depth of the root's component is not r: ``"depth d, r = r"``.
    - Nodes outside the root's component: ``"unreached: [...]"``.
    """
    components = layout_components(ball)
    if components[0][0] != [0]:
        return ["no root (node 0)"]

    notes = []
    depth = len(components[0]) - 1
    if depth != r:
        notes.append(f"depth {depth}, r = {r}")
    unreached = sorted(v for layers in components[1:] for row in layers for v in row)
    if unreached:
        notes.append(f"unreached: {unreached}")
    return notes


def draw_ball(ball: BallType, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Draw one ball with ``layered_positions``. Root in red, unreached nodes in gray.

    The axes get equal aspect and padded limits, so rows are evenly spaced in
    every panel and no node runs off the edge.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    graph = ball.to_networkx()
    components = layout_components(ball)
    reached = {v for row in components[0] for v in row} if 0 in graph else set()

    def color(v: int) -> str:
        if v == 0:
            return ROOT_COLOR
        return REACHED_COLOR if v in reached else UNREACHED_COLOR

    positions = layered_positions(ball)
    options: dict[str, Any] = {
        "node_color": [color(v) for v in graph.nodes],
        "font_color": "white",
        "node_size": 300,
        "font_size": 10,
    }
    options.update(kwargs)
    nx.draw_networkx(graph, positions, ax=ax, **options)

    _set_limits(ax, positions)
    if ball.label is not None:
        ax.set_title(ball.label)
    ax.set_axis_off()
    return ax


def _extent(positions: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """Width and height spanned by node centers."""
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    return max(xs) - min(xs), max(ys) - min(ys)


def _set_limits(
    ax: Axes,
    positions: dict[int, tuple[float, float]],
    width: float | None = None,
    height: float | None = None,
) -> None:
    """Equal-aspect limits around the nodes, padded so no node runs off the edge.

    ``width`` and ``height`` (between node centers) default to the drawing's own
    extent. Passing larger ones shared by several panels gives them all the same
    scale: the drawing is centered left to right and its top row stays at the top.
    """
    xs = [x for x, _ in positions.values()]
    own_width, own_height = _extent(positions)
    width = max(width or 0.0, own_width)
    height = max(height or 0.0, own_height)
    x_mid = (min(xs) + max(xs)) / 2
    ax.set_xlim(x_mid - width / 2 - PADDING, x_mid + width / 2 + PADDING)
    ax.set_ylim(-height - PADDING, PADDING)  # top row is always y = 0
    ax.set_aspect("equal")


def draw_ruleset(ruleset: Ruleset, ncols: int = 4) -> Figure:
    """Grid of ``draw_ball`` panels, one per type, with ``panel_notes`` in orange.

    Every panel uses the same scale, so node spacing is comparable across panels.
    The figure is sized from the largest drawing, about ``SPACING_INCHES`` per
    node spacing, so nodes keep their room however wide the widest type is.
    """
    import matplotlib.pyplot as plt

    if ncols < 1:
        raise ValueError(f"ncols must be at least 1, got {ncols}")
    n = len(ruleset)
    ncols = min(ncols, max(n, 1))
    nrows = max(math.ceil(n / ncols), 1)

    all_positions = [layered_positions(ball) for ball in ruleset.types]
    extents = [_extent(p) for p in all_positions] or [(0.0, 0.0)]
    width = max(w for w, _ in extents)
    height = max(h for _, h in extents)
    panel_width = max((width + 2 * PADDING) * SPACING_INCHES, MIN_PANEL_INCHES)
    panel_height = max((height + 2 * PADDING) * SPACING_INCHES, MIN_PANEL_INCHES) + TITLE_INCHES
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_width * ncols, panel_height * nrows), squeeze=False
    )

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_axis_off()
            continue
        ball = ruleset.types[i]
        draw_ball(ball, ax)
        _set_limits(ax, all_positions[i], width, height)
        title = f"{i}: {ball.label}" if ball.label is not None else f"type {i}"
        notes = panel_notes(ball, ruleset.r)
        if notes:
            ax.set_title("\n".join([title, *notes]), color="tab:orange")
        else:
            ax.set_title(title)

    fig.suptitle(f"{ruleset.name} (r = {ruleset.r})")
    fig.tight_layout()
    return fig
