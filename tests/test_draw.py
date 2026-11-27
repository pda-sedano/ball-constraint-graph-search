import pytest

from graph_search.balltype import BallType
from graph_search.draw import PADDING, layered_positions, layout_components, panel_notes
from graph_search.ruleset import Ruleset

BALLS = {
    "single": [],
    "edge": [[0, 1]],
    "star": [[0, 1], [1, 2], [1, 3]],
    "path-ball": [[0, 1], [0, 2], [1, 3], [2, 4]],
    "uneven": [[0, 1], [0, 2], [0, 3], [1, 4], [4, 5], [4, 6]],
    "disconnected": [[0, 1], [0, 2], [2, 3], [4, 5], [5, 6], [6, 4]],
    "rootless": [[1, 2], [2, 3], [2, 4]],
    "three-components": [[0, 1], [2, 3], [5, 6], [6, 7]],
}

each_ball = pytest.mark.parametrize("edges", BALLS.values(), ids=BALLS.keys())


def placed_components(edges):
    """For each component, ``rows[d]`` is the (x, y) of each node at distance d from its anchor."""
    ball = BallType.from_edges(edges)
    pos = layered_positions(ball)
    return [[[pos[v] for v in row] for row in layers] for layers in layout_components(ball)]


# --- layout_components ------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("single", [[[0]]]),
        ("star", [[[0], [1], [2, 3]]]),
        ("disconnected", [[[0], [1, 2], [3]], [[4], [5, 6]]]),
        ("rootless", [[[1], [2], [3, 4]]]),
        ("three-components", [[[0], [1]], [[2], [3]], [[5], [6], [7]]]),
    ],
)
def test_layout_components(name, expected):
    assert layout_components(BallType.from_edges(BALLS[name])) == expected


def test_rootless_has_no_invented_root():
    """No node 0 is drawn when the edge list has none."""
    ball = BallType.from_edges(BALLS["rootless"])
    assert 0 not in layered_positions(ball)


# --- layered_positions ------------------------------------------------------------

@each_ball
def test_every_node_gets_its_own_position(edges):
    ball = BallType.from_edges(edges)
    pos = layered_positions(ball)
    nodes = set(ball.to_networkx())
    assert set(pos) == nodes
    assert len(set(pos.values())) == len(nodes)


@each_ball
def test_layer_d_is_at_height_minus_d(edges):
    """Every component's layers are rows, and all anchors share the top row."""
    for rows in placed_components(edges):
        for depth, row in enumerate(rows):
            assert {y for _, y in row} == {-depth}


@each_ball
def test_rows_are_unit_spaced(edges):
    for rows in placed_components(edges):
        for row in rows:
            xs = sorted(x for x, _ in row)
            assert [b - a for a, b in zip(xs, xs[1:])] == pytest.approx([1.0] * (len(xs) - 1))


@each_ball
def test_rows_are_centered_on_their_anchor(edges):
    for rows in placed_components(edges):
        anchor_x = rows[0][0][0]
        for row in rows:
            assert sum(x for x, _ in row) / len(row) == pytest.approx(anchor_x)


@each_ball
def test_components_sit_side_by_side_without_overlap(edges):
    """Each component lies wholly to the right of the one before, at least two spacings away."""
    spans = [
        (min(x for row in rows for x, _ in row), max(x for row in rows for x, _ in row))
        for rows in placed_components(edges)
    ]
    for (_, right), (left, _) in zip(spans, spans[1:]):
        assert left - right >= 2.0


# --- panel_notes ------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, r, expected",
    [
        ("star", 2, []),
        ("edge", 2, ["depth 1, r = 2"]),
        ("uneven", 2, ["depth 3, r = 2"]),
        ("single", 1, ["depth 0, r = 1"]),
        ("disconnected", 2, ["unreached: [4, 5, 6]"]),
        ("three-components", 2, ["depth 1, r = 2", "unreached: [2, 3, 5, 6, 7]"]),
        ("rootless", 2, ["no root (node 0)"]),
    ],
)
def test_panel_notes(name, r, expected):
    """Every applicable note is listed, so shallow and disconnected show together."""
    assert panel_notes(BallType.from_edges(BALLS[name]), r) == expected


# --- Drawing (needs matplotlib) ----------------------------------------------------

@pytest.fixture
def plt():
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.switch_backend("Agg")
    yield pyplot
    pyplot.close("all")


@each_ball
def test_draw_ball_keeps_every_node_inside_the_axes(plt, edges):
    """Limits leave PADDING around every node center, and aspect is equal."""
    from graph_search.draw import draw_ball

    ball = BallType.from_edges(edges)
    ax = draw_ball(ball)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for x, y in layered_positions(ball).values():
        assert x0 + PADDING <= x + 1e-9 and x <= x1 - PADDING + 1e-9
        assert y0 + PADDING <= y + 1e-9 and y <= y1 - PADDING + 1e-9
    assert ax.get_aspect() == 1.0


def test_draw_ruleset_panels_share_one_scale(plt):
    """Every panel spans the same data width and height, so spacing is comparable."""
    from graph_search.draw import draw_ruleset

    types = tuple(BallType.from_edges(edges, label=name) for name, edges in BALLS.items())
    fig = draw_ruleset(Ruleset(name="all", r=2, types=types))
    panels = [ax for ax in fig.axes if ax.has_data()]
    assert len(panels) == len(types)
    spans = {
        (round(ax.get_xlim()[1] - ax.get_xlim()[0], 9), round(ax.get_ylim()[1] - ax.get_ylim()[0], 9))
        for ax in panels
    }
    assert len(spans) == 1


def test_draw_ruleset_titles_carry_notes(plt):
    from graph_search.draw import draw_ruleset

    types = (BallType.from_edges(BALLS["star"], label="ok"), BallType.from_edges(BALLS["rootless"]))
    fig = draw_ruleset(Ruleset(name="x", r=2, types=types))
    titles = [ax.get_title() for ax in fig.axes if ax.has_data()]
    assert titles == ["0: ok", "type 1\nno root (node 0)"]


# --- Whole graphs -------------------------------------------------------------------

def g6(graph):
    import networkx as nx

    return nx.to_graph6_bytes(graph, header=False).decode().strip()


def small_baseline():
    """A hand-made baseline: two graphs at n = 6, one at n = 8, none at n = 7."""
    import networkx as nx

    from graph_search.enumeration import Baseline, Level, ValidGraph

    def level(n, graphs):
        return Level(n, 10, tuple(ValidGraph(g6(g), aut) for g, aut in graphs))

    return Baseline(
        r=2,
        degree_range=(2, 3),
        levels={
            6: level(6, [(nx.cycle_graph(6), 12), (nx.circular_ladder_graph(3), 12)]),
            7: level(7, []),
            8: level(8, [(nx.hypercube_graph(3), 48)]),
        },
    )


def test_edge_crossings():
    import networkx as nx

    from graph_search.draw import edge_crossings

    square = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    assert edge_crossings(nx.cycle_graph(4), square) == 0
    assert edge_crossings(nx.complete_graph(4), square) == 1  # the two diagonals
    bowtie = {0: (0, 0), 1: (1, 1), 2: (1, 0), 3: (0, 1)}  # 0-1 crosses 2-3
    assert edge_crossings(nx.cycle_graph(4), bowtie) == 1
    assert edge_crossings(nx.path_graph(3), {0: (0, 0), 1: (1, 0), 2: (0, 0.5)}) == 0  # shared node


def test_graph_positions_are_repeatable_and_fill_the_unit_square():
    import networkx as nx

    from graph_search.draw import graph_positions

    graph = nx.petersen_graph()
    first, second = graph_positions(graph), graph_positions(graph)
    assert first == second
    assert set(first) == set(graph)
    coordinates = [c for xy in first.values() for c in xy]
    assert all(-1 - 1e-9 <= c <= 1 + 1e-9 for c in coordinates)
    assert max(abs(c) for c in coordinates) == pytest.approx(1)
    assert graph_positions(nx.Graph()) == {}
    assert graph_positions(nx.empty_graph(1)) == {0: (0.0, 0.0)}


def test_tutte_layout_applies_only_to_3_connected_planar_graphs():
    import networkx as nx

    from graph_search.draw import edge_crossings, tutte_layout

    cube = nx.convert_node_labels_to_integers(nx.hypercube_graph(3))
    positions = tutte_layout(cube)
    assert positions is not None and set(positions) == set(cube)
    assert edge_crossings(cube, positions) == 0
    distinct = {(round(x, 6), round(y, 6)) for x, y in positions.values()}
    assert len(distinct) == 8  # no two nodes coincide
    assert tutte_layout(nx.cycle_graph(6)) is None  # 2-connected
    assert tutte_layout(nx.petersen_graph()) is None  # not planar
    assert tutte_layout(nx.complete_graph(3)) is None  # too small


@pytest.mark.parametrize("n", [5, 6, 9, 12])
def test_cycles_and_planar_graphs_are_drawn_without_crossings(n):
    """A single spring run drew the hexagon as a figure eight; picking by crossings fixes that."""
    import networkx as nx

    from graph_search.draw import edge_crossings, graph_positions

    for graph in (nx.cycle_graph(n), nx.circular_ladder_graph(n), nx.hypercube_graph(3)):
        graph = nx.convert_node_labels_to_integers(graph)
        assert edge_crossings(graph, graph_positions(graph)) == 0


def test_draw_graph_keeps_every_node_inside_the_axes(plt):
    import networkx as nx

    from graph_search.draw import draw_graph, graph_positions

    graph = nx.hypercube_graph(3)
    ax = draw_graph(graph)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for x, y in graph_positions(graph).values():
        assert x0 < x < x1 and y0 < y < y1
    assert ax.get_aspect() == 1.0


def test_draw_baseline_panels_in_order_with_aut(plt):
    from graph_search.draw import draw_baseline

    fig = draw_baseline(small_baseline(), title="toy")
    titles = [ax.get_title() for ax in fig.axes if ax.has_data()]
    assert titles == ["n = 6, |Aut| = 12", "n = 6, |Aut| = 12", "n = 8, |Aut| = 48"]
    assert fig._suptitle.get_text() == "toy (r = 2)"


def test_draw_baseline_limit_and_sizes(plt):
    from graph_search.draw import draw_baseline

    fig = draw_baseline(small_baseline(), limit=2)
    assert len([ax for ax in fig.axes if ax.has_data()]) == 2
    assert fig._suptitle.get_text() == "Valid closed graphs (r = 2)\nfirst 2 of 3"

    fig = draw_baseline(small_baseline(), sizes=[8])
    assert [ax.get_title() for ax in fig.axes if ax.has_data()] == ["n = 8, |Aut| = 48"]


def test_draw_baseline_with_nothing_to_show(plt):
    from graph_search.draw import draw_baseline

    fig = draw_baseline(small_baseline(), sizes=[7])
    assert not [ax for ax in fig.axes if ax.has_data()]
    assert fig._suptitle.get_text() == "Valid closed graphs (r = 2)\nnone for n = 7"
    assert "no graphs" in [t.get_text() for t in fig.axes[0].texts]

    fig = draw_baseline(small_baseline(), sizes=[7, 9])
    assert fig._suptitle.get_text().endswith("none for n = 7")


def test_draw_graphs_checks_its_arguments(plt):
    import networkx as nx

    from graph_search.draw import draw_graphs

    with pytest.raises(ValueError, match="ncols"):
        draw_graphs([nx.path_graph(3)], ncols=0)
    with pytest.raises(ValueError, match="titles"):
        draw_graphs([nx.path_graph(3)], titles=["a", "b"])
