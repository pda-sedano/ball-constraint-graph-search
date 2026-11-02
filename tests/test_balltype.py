import networkx as nx
import pytest

from balltype import BallType, check_edge_list
from report import IssueCode

STAR = [[0, 1], [1, 2], [1, 3]]

BAD_EDGE_LISTS = [
    ("not a list", IssueCode.BAD_VALUE),
    ([[0, 1], [1]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, 2.0]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, True]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, -2]], IssueCode.NEGATIVE_NODE),
    ([[0, 1], [1, 1]], IssueCode.SELF_LOOP),
    ([[0, 1], [0, 1]], IssueCode.REPEATED_EDGE),
    ([[0, 1], [1, 0]], IssueCode.REPEATED_EDGE),
    ([[0, 1], [1, 3]], IssueCode.NODE_GAP),
    ([[1, 2]], IssueCode.NODE_GAP),  # no root
    ([[0, 1], [2, 3]], IssueCode.DISCONNECTED),
]


# --- check_edge_list on raw input -------------------------------------------

@pytest.mark.parametrize("raw, code", BAD_EDGE_LISTS)
def test_check_edge_list_reports(raw, code):
    """Each bad edge list produces its code."""
    assert code in {issue.code for issue in check_edge_list(raw)}


def test_check_edge_list_reports_all_problems():
    """A list with a self-loop and a repeated edge reports both, not just the first."""
    codes = {issue.code for issue in check_edge_list([[0, 1], [1, 1], [1, 0]])}
    assert {IssueCode.SELF_LOOP, IssueCode.REPEATED_EDGE} <= codes


def test_check_edge_list_tags_type_index():
    """Every returned issue carries the type_index passed in."""
    issues = check_edge_list([[0, 1], [1, 1], [1, 0], [1]], type_index=5)
    assert len(issues) >= 2
    assert all(issue.type_index == 5 for issue in issues)


@pytest.mark.parametrize("raw", [[], [[0, 1]], [[0, 1], [1, 2], [1, 3]]])
def test_check_edge_list_accepts_valid(raw):
    """Valid lists, including the empty one (single-node ball), give no issues."""
    assert check_edge_list(raw) == []


# --- BallType construction ---------------------------------------------------

def test_from_edges_normalizes_and_sorts():
    """[[2, 1], [1, 0]] stores edges ((0, 1), (1, 2)) and num_nodes 3."""
    ball = BallType.from_edges([[2, 1], [1, 0]])
    assert ball.edges == ((0, 1), (1, 2))
    assert ball.num_nodes == 3


def test_from_edges_empty_is_single_node():
    """No edges: num_nodes 1, depth 0, layers ((0,),)."""
    ball = BallType.from_edges([])
    assert ball.num_nodes == 1
    assert ball.depth == 0
    assert ball.layers == ((0,),)


@pytest.mark.parametrize("raw", [raw for raw, _ in BAD_EDGE_LISTS])
def test_from_edges_rejects_invalid(raw):
    """Anything check_edge_list rejects raises ValueError here."""
    with pytest.raises(ValueError):
        BallType.from_edges(raw)


def test_direct_construction_rejects_unsorted_edges():
    """BallType(3, ((1, 2), (0, 1))) raises: __post_init__ checks normalization."""
    with pytest.raises(ValueError):
        BallType(3, ((1, 2), (0, 1)))


def test_from_networkx_puts_root_at_zero():
    """Root becomes node 0 and the edge count is preserved."""
    graph = nx.Graph([("a", "b"), ("b", "c"), ("b", "d"), ("c", "d")])
    ball = BallType.from_networkx(graph, "c")
    assert ball.num_nodes == graph.number_of_nodes()
    assert len(ball.edges) == graph.number_of_edges()
    assert ball.root_degree == graph.degree["c"]
    # Layer sizes depend on which node is the root, so they pin down the renumbering.
    distances = nx.single_source_shortest_path_length(graph, "c").values()
    assert ball.layer_sizes == tuple(
        sum(1 for d in distances if d == k) for k in range(max(distances) + 1)
    )


# --- Derived fields: the doc's star-only example ------------------------------

def test_star_only_layers():
    """edges [[0,1],[1,2],[1,3]] -> layers ((0,), (1,), (2, 3)), depth 2."""
    ball = BallType.from_edges(STAR)
    assert ball.layers == ((0,), (1,), (2, 3))
    assert ball.depth == 2


def test_star_only_degrees():
    """degrees (1, 3, 1, 1), root_degree 1, layer_sizes (1, 1, 2)."""
    ball = BallType.from_edges(STAR)
    assert ball.degrees == (1, 3, 1, 1)
    assert ball.root_degree == 1
    assert ball.layer_sizes == (1, 1, 2)


def test_to_networkx_sets_depth_attribute():
    """Every node has a "depth" attribute equal to ball.distance(node)."""
    ball = BallType.from_edges(STAR)
    graph = ball.to_networkx()
    assert set(graph.nodes) == set(range(ball.num_nodes))
    assert all(depth == ball.distance(v) for v, depth in graph.nodes(data="depth"))


# --- Equality -----------------------------------------------------------------

def test_equality_includes_label_and_count():
    """Same edges, different label: not ==, but same structure_key."""
    a = BallType.from_edges(STAR, label="a")
    b = BallType.from_edges(STAR, label="b")
    assert a != b
    assert a.structure_key == b.structure_key
    assert BallType.from_edges(STAR, count=1) != BallType.from_edges(STAR, count=2)


def test_ball_is_hashable():
    """BallType can go in a set (it is frozen and all compared fields are tuples)."""
    balls = {BallType.from_edges(STAR), BallType.from_edges(STAR), BallType.from_edges([])}
    assert len(balls) == 2


def test_to_dict_omits_unset_fields():
    """No "label" or "count" key when they are None."""
    assert BallType.from_edges(STAR).to_dict() == {"edges": STAR}
