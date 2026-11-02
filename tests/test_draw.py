import pytest

from balltype import BallType
from draw import layered_positions

BALLS = {
    "single": [],
    "edge": [[0, 1]],
    "star": [[0, 1], [1, 2], [1, 3]],
    "path-ball": [[0, 1], [0, 2], [1, 3], [2, 4]],
    "uneven": [[0, 1], [0, 2], [0, 3], [1, 4], [4, 5], [4, 6]],
}

each_ball = pytest.mark.parametrize("edges", BALLS.values(), ids=BALLS.keys())


def rows(edges):
    """Positions grouped by layer: ``rows[d]`` is the (x, y) of each node at depth d."""
    ball = BallType.from_edges(edges)
    pos = layered_positions(ball)
    return [[pos[v] for v in layer] for layer in ball.layers]


@each_ball
def test_every_node_gets_its_own_position(edges):
    ball = BallType.from_edges(edges)
    pos = layered_positions(ball)
    assert set(pos) == set(range(ball.num_nodes))
    assert len(set(pos.values())) == ball.num_nodes


@each_ball
def test_each_layer_is_one_row(edges):
    for row in rows(edges):
        assert len({y for _, y in row}) == 1


@each_ball
def test_root_on_top_and_layers_descend(edges):
    row_ys = [row[0][1] for row in rows(edges)]
    assert row_ys == sorted(row_ys, reverse=True)
    assert len(set(row_ys)) == len(row_ys)


@each_ball
def test_rows_are_evenly_spaced(edges):
    for row in rows(edges):
        xs = sorted(x for x, _ in row)
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        assert gaps == pytest.approx(gaps[:1] * len(gaps))


@each_ball
def test_rows_are_centered_on_the_root(edges):
    all_rows = rows(edges)
    root_x = all_rows[0][0][0]
    for row in all_rows:
        assert sum(x for x, _ in row) / len(row) == pytest.approx(root_x)
