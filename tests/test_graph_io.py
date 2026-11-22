"""Graph files: round trips and format checks. No extension needed."""

import json

import networkx as nx
import numpy as np
import pytest
from safetensors.numpy import save_file

from graph_io import GraphFile, load_graph, save_graph
from report import IssueCode, RulesetError


def write(path, tensors, metadata):
    save_file(tensors, str(path), metadata=metadata)


def good_tensors():
    return {
        "edges": np.array([[0, 1], [1, 2]], dtype=np.uint32),
        "open": np.array([0], dtype=np.uint32),
    }


GOOD_METADATA = {"format": "1", "num_nodes": "3", "metadata": "{}"}


# --- Round trips -------------------------------------------------------------------------

def test_round_trip(tmp_path):
    path = tmp_path / "g.safetensors"
    graph = nx.relabel_nodes(nx.cycle_graph(5), {i: f"v{i}" for i in range(5)})
    save_graph(path, graph, open_nodes=["v3", "v1", "v3"], metadata={"source": "test", "n": 5})
    loaded, report = load_graph(path)
    assert report.issues == []
    assert loaded == GraphFile(
        num_nodes=5,
        edges=((0, 1), (0, 4), (1, 2), (2, 3), (3, 4)),
        open_nodes=(1, 3),
        metadata={"source": "test", "n": 5},
    )
    assert nx.is_isomorphic(loaded.to_networkx(), graph)


def test_empty_graph_and_isolated_nodes(tmp_path):
    path = tmp_path / "g.safetensors"
    graph = nx.empty_graph(4)
    save_graph(path, graph)
    loaded, report = load_graph(path)
    assert report.ok
    assert (loaded.num_nodes, loaded.edges, loaded.open_nodes) == (4, (), ())
    assert len(loaded.to_networkx()) == 4


def test_save_rejects_bad_graphs(tmp_path):
    with pytest.raises(ValueError, match="open nodes not in the graph"):
        save_graph(tmp_path / "g.safetensors", nx.path_graph(3), open_nodes=[7])
    with pytest.raises(ValueError, match="self-loops"):
        save_graph(tmp_path / "g.safetensors", nx.Graph([(0, 0)]))
    with pytest.raises(TypeError):
        save_graph(tmp_path / "g.safetensors", nx.path_graph(3), metadata={"x": object()})


def test_file_is_plain_safetensors(tmp_path):
    """Readable by any safetensors reader: the header and arrays are as documented."""
    from safetensors import safe_open

    path = tmp_path / "g.safetensors"
    save_graph(path, nx.path_graph(3), open_nodes=[2])
    with safe_open(str(path), framework="numpy") as f:
        assert f.metadata() == {"format": "1", "num_nodes": "3", "metadata": "{}"}
        assert f.get_tensor("edges").tolist() == [[0, 1], [1, 2]]
        assert f.get_tensor("open").dtype == np.uint32


# --- Format checks ------------------------------------------------------------------------

def codes(path):
    graph_file, report = load_graph(path)
    assert graph_file is None
    return report.codes()


def test_not_safetensors(tmp_path):
    path = tmp_path / "g.safetensors"
    path.write_bytes(b"garbage")
    assert codes(path) == {IssueCode.MALFORMED_GRAPH_FILE}
    assert codes(tmp_path / "missing.safetensors") == {IssueCode.MALFORMED_GRAPH_FILE}


@pytest.mark.parametrize(
    "metadata, code",
    [
        ({"format": "1", "num_nodes": "3"}, IssueCode.MISSING_KEY),
        ({**GOOD_METADATA, "extra": "x"}, IssueCode.UNKNOWN_KEY),
        ({**GOOD_METADATA, "format": "2"}, IssueCode.BAD_FORMAT_VERSION),
        ({**GOOD_METADATA, "num_nodes": "-3"}, IssueCode.BAD_VALUE),
        ({**GOOD_METADATA, "num_nodes": "three"}, IssueCode.BAD_VALUE),
        ({**GOOD_METADATA, "metadata": "not json"}, IssueCode.BAD_VALUE),
        ({**GOOD_METADATA, "metadata": json.dumps([1, 2])}, IssueCode.BAD_VALUE),
        ({**GOOD_METADATA, "num_nodes": "2"}, IssueCode.NODE_OUT_OF_RANGE),
    ],
)
def test_bad_metadata(tmp_path, metadata, code):
    path = tmp_path / "g.safetensors"
    write(path, good_tensors(), metadata)
    assert code in codes(path)


def test_no_metadata_at_all(tmp_path):
    path = tmp_path / "g.safetensors"
    write(path, good_tensors(), None)
    assert codes(path) == {IssueCode.MISSING_KEY}


@pytest.mark.parametrize(
    "change",
    [
        {"edges": np.array([[0, 1]], dtype=np.int64)},
        {"edges": np.array([0, 1, 2], dtype=np.uint32)},
        {"edges": np.array([[0, 1, 2]], dtype=np.uint32)},
        {"open": np.array([[0]], dtype=np.uint32)},
        {"labels": np.array([0], dtype=np.uint32)},
    ],
)
def test_bad_tensors(tmp_path, change):
    path = tmp_path / "g.safetensors"
    write(path, {**good_tensors(), **change}, GOOD_METADATA)
    assert codes(path) == {IssueCode.BAD_TENSOR}


def test_missing_tensor(tmp_path):
    path = tmp_path / "g.safetensors"
    write(path, {"edges": good_tensors()["edges"]}, GOOD_METADATA)
    assert codes(path) == {IssueCode.BAD_TENSOR}


def test_all_problems_reported_together(tmp_path):
    path = tmp_path / "g.safetensors"
    write(path, {"edges": np.array([[0, 1]], dtype=np.int32)}, {"format": "9", "num_nodes": "x"})
    assert codes(path) == {
        IssueCode.MISSING_KEY,
        IssueCode.BAD_FORMAT_VERSION,
        IssueCode.BAD_VALUE,
        IssueCode.BAD_TENSOR,
    }


def test_strict_raises_with_the_report(tmp_path):
    path = tmp_path / "g.safetensors"
    write(path, good_tensors(), {**GOOD_METADATA, "format": "2"})
    with pytest.raises(RulesetError) as info:
        load_graph(path, strict=True)
    assert info.value.report.codes() == {IssueCode.BAD_FORMAT_VERSION}


def test_graph_checks_are_left_to_rust(tmp_path):
    """Self-loops and repeated edges are graph problems, not format problems."""
    path = tmp_path / "g.safetensors"
    tensors = {"edges": np.array([[1, 1], [0, 1], [0, 1]], dtype=np.uint32), "open": np.array([], dtype=np.uint32)}
    write(path, tensors, GOOD_METADATA)
    loaded, report = load_graph(path)
    assert report.ok and loaded.edges == ((1, 1), (0, 1), (0, 1))
