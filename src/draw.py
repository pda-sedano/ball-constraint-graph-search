"""Drawings of ball types and of whole graphs.

Ball types are drawn in layers by distance from the root, to confirm a file
says what was meant. Whole graphs, such as the enumeration baseline's valid
graphs, get a spring layout, since they have no root to hang layers from.

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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx

from balltype import BallType
from ruleset import Ruleset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from enumeration import Baseline

ROOT_COLOR = "tab:red"
REACHED_COLOR = "tab:blue"
UNREACHED_COLOR = "tab:gray"

COMPONENT_GAP = 1.0  # extra horizontal space between components, in node spacings
PADDING = 0.5  # space around the outermost nodes, in node spacings
SPACING_INCHES = 0.6  # one node spacing on the page, for draw_ruleset's figure size
TITLE_INCHES = 0.7  # room above each panel for up to three title lines
MIN_PANEL_INCHES = 1.5

GRAPH_COLOR = "tab:blue"
GRAPH_PANEL_INCHES = 2.2  # each whole-graph panel, before its title
GRAPH_MARGIN = 0.15  # space around a whole-graph layout, which fills [-1, 1]


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


# --- Whole graphs ---------------------------------------------------------------


def edge_crossings(graph: nx.Graph, positions: dict[Any, tuple[float, float]]) -> int:
    """Number of pairs of edges, with no endpoint in common, whose segments cross.

    Pure function, no matplotlib. Touching or collinear segments do not count;
    with floating-point layouts they essentially never happen.
    """

    def turn(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    edges = list(graph.edges)
    count = 0
    for i, (a, b) in enumerate(edges):
        pa, pb = positions[a], positions[b]
        for c, d in edges[i + 1 :]:
            if len({a, b, c, d}) < 4:
                continue
            pc, pd = positions[c], positions[d]
            if turn(pa, pb, pc) * turn(pa, pb, pd) < 0 and turn(pc, pd, pa) * turn(pc, pd, pb) < 0:
                count += 1
    return count


def graph_positions(graph: nx.Graph, tries: int = 8) -> dict[Any, tuple[float, float]]:
    """A layout filling [-1, 1] with few edge crossings, the same every time for a graph.

    Candidates are ``tutte_layout`` when it applies, the circular layout, and
    spring layouts started from the circle and from seeds 0 to ``tries`` - 1.
    The one with the fewest crossings wins, the earliest on a tie. A single
    spring run often stops in a twisted local minimum (a hexagon drawn as a
    figure eight), and no spring run found the cube's drawing without
    crossings. Kamada-Kawai and spectral layouts were left out: on symmetric
    graphs such as the cube or K3,3 they put nodes on top of each other, and
    Kamada-Kawai needs scipy.
    """
    if graph.number_of_nodes() == 0:
        return {}
    circle = nx.circular_layout(graph)
    tutte = tutte_layout(graph)
    candidates = [] if tutte is None else [tutte]
    candidates += [circle, nx.spring_layout(graph, pos=circle, iterations=100)]
    candidates += [nx.spring_layout(graph, seed=seed, iterations=100) for seed in range(tries)]
    best: dict[Any, tuple[float, float]] = {}
    fewest = math.inf
    for layout in candidates:
        positions = _fill_unit_square(layout)
        crossings = edge_crossings(graph, positions)
        if crossings < fewest:
            best, fewest = positions, crossings
    return best


def tutte_layout(graph: nx.Graph) -> dict[Any, tuple[float, float]] | None:
    """Tutte's barycentric drawing, for 3-connected planar graphs; None for any other.

    The face with the most nodes goes on a circle, and every other node sits
    at the average of its neighbors' positions, found by one linear solve.
    Tutte (1963) proved this draws a 3-connected planar graph with straight
    edges, no crossings and convex faces: the cube becomes two nested squares,
    a prism two nested polygons. Graphs that are not 3-connected can get nodes
    on top of each other, so they get None and are left to the other layouts.
    """
    import numpy as np

    if graph.number_of_nodes() < 4:
        return None
    planar, embedding = nx.check_planarity(graph)
    if not planar or nx.node_connectivity(graph) < 3:
        return None

    seen: set[tuple[Any, Any]] = set()
    outer: list[Any] = []
    for u, v in embedding.edges:
        if (u, v) not in seen:
            face = embedding.traverse_face(u, v, mark_half_edges=seen)
            if len(face) > len(outer):
                outer = face

    positions: dict[Any, tuple[float, float]] = {}
    for i, v in enumerate(outer):
        angle = 2 * math.pi * i / len(outer)
        positions[v] = (math.cos(angle), math.sin(angle))
    inner = [v for v in graph if v not in positions]
    index = {v: i for i, v in enumerate(inner)}
    laplacian = np.zeros((len(inner), len(inner)))
    pull = np.zeros((len(inner), 2))
    for v in inner:
        laplacian[index[v], index[v]] = graph.degree(v)
        for u in graph[v]:
            if u in index:
                laplacian[index[v], index[u]] -= 1
            else:
                pull[index[v]] += positions[u]
    for v, (x, y) in zip(inner, np.linalg.solve(laplacian, pull)):
        positions[v] = (float(x), float(y))
    return positions


def _fill_unit_square(layout: dict[Any, Any]) -> dict[Any, tuple[float, float]]:
    """Center a layout on the origin and scale its longer side to span [-1, 1]."""
    xs = [float(p[0]) for p in layout.values()]
    ys = [float(p[1]) for p in layout.values()]
    x_mid, y_mid = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    return {
        v: ((float(p[0]) - x_mid) / half, (float(p[1]) - y_mid) / half) for v, p in layout.items()
    }


def draw_graph(graph: nx.Graph, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Draw a whole graph with ``graph_positions``, unlabelled, on equal-aspect axes.

    Keyword arguments go to ``networkx.draw_networkx``, for example
    ``with_labels=True`` to show node numbers.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    options: dict[str, Any] = {
        "node_color": GRAPH_COLOR,
        "node_size": 60,
        "with_labels": False,
        "font_color": "white",
        "font_size": 8,
    }
    options.update(kwargs)
    nx.draw_networkx(graph, graph_positions(graph), ax=ax, **options)
    edge = 1 + GRAPH_MARGIN
    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def draw_graphs(
    graphs: Sequence[nx.Graph],
    titles: Sequence[str] | None = None,
    ncols: int = 6,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Grid of ``draw_graph`` panels. Extra keyword arguments go to ``draw_graph``.

    With no graphs the figure has one blank panel saying so, rather than nothing.
    """
    import matplotlib.pyplot as plt

    if ncols < 1:
        raise ValueError(f"ncols must be at least 1, got {ncols}")
    if titles is not None and len(titles) != len(graphs):
        raise ValueError(f"{len(titles)} titles for {len(graphs)} graphs")
    n = len(graphs)
    ncols = min(ncols, max(n, 1))
    nrows = max(math.ceil(n / ncols), 1)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(GRAPH_PANEL_INCHES * ncols, (GRAPH_PANEL_INCHES + 0.3) * nrows + 0.4),
        squeeze=False,
    )
    for i, ax in enumerate(axes.flat):
        if i < n:
            draw_graph(graphs[i], ax, **kwargs)
            if titles is not None:
                ax.set_title(titles[i], fontsize=10)
        else:
            ax.set_axis_off()
    if n == 0:
        blank = axes[0][0]
        blank.text(0.5, 0.5, "no graphs", ha="center", va="center", transform=blank.transAxes)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def draw_baseline(
    baseline: Baseline,
    sizes: Iterable[int] | None = None,
    limit: int | None = 36,
    ncols: int = 6,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """The baseline's valid graphs in a grid, by n, each titled with n and |Aut|.

    Args:
        baseline: from ``enumeration.baseline``.
        sizes: node counts to show; all of the baseline's by default.
        limit: at most this many panels, the first ones by n; None for all. The
            title says when graphs were left out.
        ncols: panels per row.
        title: the figure's title, such as the ruleset name; "Valid closed
            graphs" by default. The radius is added after it.
        **kwargs: passed to ``draw_graph``, e.g. ``with_labels=True``.
    """
    wanted = set(baseline.levels) if sizes is None else set(sizes)
    found = [
        (n, valid)
        for n in sorted(baseline.levels)
        if n in wanted
        for valid in baseline.levels[n].valid
    ]
    shown = found if limit is None else found[:limit]

    heading = f"{title or 'Valid closed graphs'} (r = {baseline.r})"
    if not found:
        sizes_seen = sorted(wanted & set(baseline.levels))
        if not sizes_seen:
            span = "the sizes asked for"
        elif len(sizes_seen) == 1:
            span = f"n = {sizes_seen[0]}"
        else:
            span = f"n = {sizes_seen[0]} to {sizes_seen[-1]}"
        heading += f"\nnone for {span}"
    elif len(shown) < len(found):
        heading += f"\nfirst {len(shown)} of {len(found)}"

    return draw_graphs(
        [valid.to_networkx() for _, valid in shown],
        titles=[f"n = {n}, |Aut| = {valid.automorphisms}" for n, valid in shown],
        ncols=ncols,
        title=heading,
        **kwargs,
    )
