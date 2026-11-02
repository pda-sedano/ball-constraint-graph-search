import pytest

from report import IssueCode

TODO = pytest.mark.skip(reason="not written yet")


# --- check_edge_list on raw input -------------------------------------------

@TODO
@pytest.mark.parametrize(
    "raw, code",
    [
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
    ],
)
def test_check_edge_list_reports(raw, code):
    """Each bad edge list produces its code."""


@TODO
def test_check_edge_list_reports_all_problems():
    """A list with a self-loop and a repeated edge reports both, not just the first."""


@TODO
def test_check_edge_list_tags_type_index():
    """Every returned issue carries the type_index passed in."""


@TODO
@pytest.mark.parametrize("raw", [[], [[0, 1]], [[0, 1], [1, 2], [1, 3]]])
def test_check_edge_list_accepts_valid(raw):
    """Valid lists, including the empty one (single-node ball), give no issues."""


# --- BallType construction ---------------------------------------------------

@TODO
def test_from_edges_normalizes_and_sorts():
    """[[2, 1], [1, 0]] stores edges ((0, 1), (1, 2)) and num_nodes 3."""


@TODO
def test_from_edges_empty_is_single_node():
    """No edges: num_nodes 1, depth 0, layers ((0,),)."""


@TODO
def test_from_edges_rejects_invalid():
    """Anything check_edge_list rejects raises ValueError here."""


@TODO
def test_direct_construction_rejects_unsorted_edges():
    """BallType(3, ((1, 2), (0, 1))) raises: __post_init__ checks normalization."""


@TODO
def test_from_networkx_puts_root_at_zero():
    """Root becomes node 0 and the edge count is preserved."""


# --- Derived fields: the doc's star-only example ------------------------------

@TODO
def test_star_only_layers():
    """edges [[0,1],[1,2],[1,3]] -> layers ((0,), (1,), (2, 3)), depth 2."""


@TODO
def test_star_only_degrees():
    """degrees (1, 3, 1, 1), root_degree 1, layer_sizes (1, 1, 2)."""


@TODO
def test_to_networkx_sets_depth_attribute():
    """Every node has a "depth" attribute equal to ball.distance(node)."""


# --- Equality -----------------------------------------------------------------

@TODO
def test_equality_includes_label_and_count():
    """Same edges, different label: not ==, but same structure_key."""


@TODO
def test_ball_is_hashable():
    """BallType can go in a set (it is frozen and all compared fields are tuples)."""


@TODO
def test_to_dict_omits_unset_fields():
    """No "label" or "count" key when they are None."""
