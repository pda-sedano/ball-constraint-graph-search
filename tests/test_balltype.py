import pytest

from balltype import BallType, InvalidBallError, check_edge_format
from report import IssueCode

STAR = [[0, 1], [1, 2], [1, 3]]

BAD_EDGE_FORMATS = [
    ("not a list", IssueCode.BAD_VALUE),
    ([[0, 1], [1]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, 2.0]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, True]], IssueCode.MALFORMED_EDGE),
    ([[0, 1], [1, -2]], IssueCode.NEGATIVE_NODE),
]

# Well-formed edge lists with graph problems. Python accepts them; Rust's
# compile rejects them (see test_io.COMPILE_FIXTURES for the codes).
BAD_STRUCTURE = [
    [[0, 1], [1, 1]],  # self-loop
    [[0, 1], [0, 1]],  # repeated edge
    [[0, 1], [1, 0]],  # repeated edge, reversed
    [[0, 1], [1, 3]],  # gap in numbering
    [[1, 2]],  # no root
    [[0, 1], [2, 3]],  # disconnected
]


# --- check_edge_format on raw input -------------------------------------------

@pytest.mark.parametrize("raw, code", BAD_EDGE_FORMATS)
def test_check_edge_format_reports(raw, code):
    """Each badly formatted edge list produces its code."""
    assert code in {issue.code for issue in check_edge_format(raw)}


def test_check_edge_format_reports_all_problems():
    """A list with a malformed edge and a negative node reports both, not just the first."""
    codes = {issue.code for issue in check_edge_format([[0, 1], [1], [1, -2]])}
    assert {IssueCode.MALFORMED_EDGE, IssueCode.NEGATIVE_NODE} <= codes


def test_check_edge_format_tags_type_index():
    """Every returned issue carries the type_index passed in."""
    issues = check_edge_format([[0, 1], [1], [1, -2]], type_index=5)
    assert len(issues) == 2
    assert all(issue.type_index == 5 for issue in issues)


@pytest.mark.parametrize("raw", [[], [[0, 1]], STAR, ((0, 1), (1, 2))])
def test_check_edge_format_accepts_valid(raw):
    """Valid lists, including the empty one (single-node ball) and tuples, give no issues."""
    assert check_edge_format(raw) == []


@pytest.mark.parametrize("raw", BAD_STRUCTURE)
def test_check_edge_format_ignores_structure(raw):
    """Graph problems are left to Rust: well-formed edges pass here."""
    assert check_edge_format(raw) == []
    BallType.from_edges(raw)


# --- BallType construction ---------------------------------------------------

def test_from_edges_keeps_edges_as_written():
    """Order and orientation are kept, so Rust's issue positions match the file."""
    ball = BallType.from_edges([[2, 1], [1, 0]])
    assert ball.edges == ((2, 1), (1, 0))


def test_from_edges_empty_is_single_node():
    """No edges: the single-node ball, just the root."""
    ball = BallType.from_edges([])
    assert ball.edges == ()
    assert set(ball.to_networkx()) == {0}


def test_from_edges_accepts_any_iterable():
    """A generator of pairs works the same as a list."""
    assert BallType.from_edges((e for e in STAR)) == BallType.from_edges(STAR)


@pytest.mark.parametrize("raw, code", BAD_EDGE_FORMATS)
def test_from_edges_rejects_bad_format(raw, code):
    """Anything check_edge_format rejects raises InvalidBallError carrying the issues."""
    with pytest.raises(InvalidBallError) as excinfo:
        BallType.from_edges(raw)
    assert excinfo.value.issues


def test_direct_construction_stores_tuples():
    """Lists passed to the constructor are stored as tuples, keeping the object hashable."""
    ball = BallType(([0, 1], [1, 2]))
    assert ball.edges == ((0, 1), (1, 2))
    hash(ball)


def test_direct_construction_rejects_bad_format():
    with pytest.raises(InvalidBallError):
        BallType(((0, -1),))


# --- NetworkX view --------------------------------------------------------------

def test_to_networkx_has_every_named_node():
    graph = BallType.from_edges(STAR).to_networkx()
    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.number_of_edges() == 3


@pytest.mark.parametrize("raw", BAD_STRUCTURE)
def test_to_networkx_tolerates_bad_structure(raw):
    """Safe on ball types that would fail compiling, so they can still be drawn.

    The nodes are exactly those named in the edges: no root is invented.
    """
    graph = BallType.from_edges(raw).to_networkx()
    assert set(graph.nodes) == {v for edge in raw for v in edge}


# --- Equality -----------------------------------------------------------------

def test_equality_includes_label_and_count():
    """Same edges, different label or count: not equal."""
    assert BallType.from_edges(STAR, label="a") != BallType.from_edges(STAR, label="b")
    assert BallType.from_edges(STAR, count=1) != BallType.from_edges(STAR, count=2)


def test_equality_is_as_written():
    """The same graph written in another order is a different BallType."""
    assert BallType.from_edges([[0, 1], [1, 2]]) != BallType.from_edges([[1, 2], [0, 1]])


def test_ball_is_hashable():
    """BallType can go in a set (it is frozen and all compared fields are tuples)."""
    balls = {BallType.from_edges(STAR), BallType.from_edges(STAR), BallType.from_edges([])}
    assert len(balls) == 2


def test_to_dict_omits_unset_fields():
    """No "label" or "count" key when they are None."""
    assert BallType.from_edges(STAR).to_dict() == {"edges": STAR}


def test_to_dict_keeps_edges_as_written():
    assert BallType.from_edges([[2, 1], [1, 0]]).to_dict() == {"edges": [[2, 1], [1, 0]]}
