"""Graph files: safetensors, with format checks only.

One format for every graph the project saves, from checker inputs to
million-node samples (the design doc's "Graph file format")::

    tensor   edges      uint32, shape (m, 2): one row per edge, (min, max), sorted
    tensor   open       uint32, shape (k,): the open set, sorted; empty when closed
    metadata format     "1"
    metadata num_nodes  the node count, as a decimal string
    metadata metadata   a JSON object: ruleset name, source, and for samples |Aut|
                        and the chain step

As with rulesets, this module checks only the format: tensor names, dtypes
and shapes, metadata keys and values, and node ids below ``num_nodes``.
Self-loops, repeated edges and the degree cap are graph checks, made in Rust
when the graph is checked. Nodes are numbered 0..n-1; labels are not saved.

Example::

    import networkx as nx
    from graph_io import load_graph, save_graph

    save_graph("patch.safetensors", nx.path_graph(9), open_nodes=[0, 8],
               metadata={"source": "hand"})
    graph_file, report = load_graph("patch.safetensors")
    graph_file.to_networkx(), graph_file.open_nodes   # (path, (0, 8))
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import networkx as nx
import numpy as np

from extract import number_nodes
from report import IssueCode, LoadReport

FORMAT_VERSION = "1"
METADATA_KEYS = frozenset({"format", "num_nodes", "metadata"})
TENSORS = frozenset({"edges", "open"})


@dataclass(frozen=True)
class GraphFile:
    """A graph as stored: nodes 0..num_nodes-1, edges and open set as written."""

    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    open_nodes: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_networkx(self) -> nx.Graph:
        """The graph as a NetworkX graph on nodes 0..num_nodes-1.

        NetworkX merges repeated edges and keeps self-loops, so pass
        ``edges`` itself to anything that must see the file exactly.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.edges)
        return graph


def save_graph(
    path: str | PathLike[str],
    graph: nx.Graph,
    open_nodes: Iterable[Hashable] = (),
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write ``graph`` with its open set. Nodes are renumbered 0..n-1 in the
    graph's own order. ``metadata`` must be JSON-serializable.

    Raises ValueError for a directed graph, a multigraph, self-loops, or an
    open node not in the graph.
    """
    from safetensors.numpy import save_file

    labels, number, edges = number_nodes(graph)
    missing = [v for v in open_nodes if v not in number]
    if missing:
        raise ValueError(f"open nodes not in the graph: {missing[:5]}")
    normalized = sorted((min(a, b), max(a, b)) for a, b in edges)
    open_numbers = sorted({number[v] for v in open_nodes})
    save_file(
        {
            "edges": np.array(normalized, dtype=np.uint32).reshape(-1, 2),
            "open": np.array(open_numbers, dtype=np.uint32),
        },
        str(path),
        metadata={
            "format": FORMAT_VERSION,
            "num_nodes": str(len(labels)),
            "metadata": json.dumps(dict(metadata or {})),
        },
    )


def load_graph(
    path: str | PathLike[str], *, strict: bool = False
) -> tuple[GraphFile | None, LoadReport]:
    """Read a graph file and check its format, reporting every problem found.

    Returns ``(graph_file, report)``; ``graph_file`` is None whenever the
    report has errors. With ``strict``, raises ``RulesetError`` (the shared
    load error, carrying the report) instead.
    """
    from safetensors import safe_open

    report = LoadReport()
    try:
        with safe_open(str(path), framework="numpy") as f:
            header = f.metadata() or {}
            tensors = {name: f.get_tensor(name) for name in f.keys()}
    except Exception as exc:  # safetensors raises its own error type, plus OSError
        report.add(IssueCode.MALFORMED_GRAPH_FILE, f"not a readable safetensors file: {exc}")
        return _finish(None, report, strict)

    num_nodes = _check_metadata(header, report)
    extra = _check_metadata_value(header, report)
    edges = _check_tensor(tensors, "edges", 2, report)
    open_ = _check_tensor(tensors, "open", 1, report)
    for name in sorted(set(tensors) - TENSORS):
        report.add(IssueCode.BAD_TENSOR, f"unknown tensor {name!r}")

    if num_nodes is not None:
        for name, array in (("edges", edges), ("open", open_)):
            if array is not None and array.size and int(array.max()) >= num_nodes:
                bad = sorted({int(x) for x in array.ravel() if x >= num_nodes})
                report.add(
                    IssueCode.NODE_OUT_OF_RANGE,
                    f"{name} has nodes {bad[:5]} not below num_nodes = {num_nodes}",
                )
    if not report.ok:
        return _finish(None, report, strict)
    graph_file = GraphFile(
        num_nodes=num_nodes,
        edges=tuple((int(a), int(b)) for a, b in edges),
        open_nodes=tuple(int(v) for v in open_),
        metadata=extra,
    )
    return _finish(graph_file, report, strict)


def _finish(graph_file, report, strict):
    if strict:
        report.raise_if_errors()
    return graph_file, report


def _check_metadata(header: Mapping[str, str], report: LoadReport) -> int | None:
    """Keys present and no others, the format version, and num_nodes.
    Returns num_nodes if it is valid."""
    for key in sorted(METADATA_KEYS - set(header)):
        report.add(IssueCode.MISSING_KEY, f"metadata has no {key!r}")
    for key in sorted(set(header) - METADATA_KEYS):
        report.add(IssueCode.UNKNOWN_KEY, f"unknown metadata key {key!r}")
    if "format" in header and header["format"] != FORMAT_VERSION:
        report.add(
            IssueCode.BAD_FORMAT_VERSION,
            f"format is {header['format']!r}, expected {FORMAT_VERSION!r}",
        )
    text = header.get("num_nodes")
    if text is None:
        return None
    if not (text.isascii() and text.isdigit()):
        report.add(IssueCode.BAD_VALUE, f"num_nodes must be a non-negative integer, got {text!r}")
        return None
    return int(text)


def _check_metadata_value(header: Mapping[str, str], report: LoadReport) -> dict[str, Any]:
    """The free-form metadata: a JSON object."""
    text = header.get("metadata")
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        report.add(IssueCode.BAD_VALUE, f"metadata is not valid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        report.add(IssueCode.BAD_VALUE, f"metadata must be a JSON object, got {type(value).__name__}")
        return {}
    return value


def _check_tensor(tensors, name, ndim, report) -> np.ndarray | None:
    """Present, uint32, and shaped (m, 2) for edges or (k,) for open."""
    array = tensors.get(name)
    if array is None:
        report.add(IssueCode.BAD_TENSOR, f"no {name!r} tensor")
        return None
    problems = []
    if array.dtype != np.uint32:
        problems.append(f"dtype {array.dtype}, expected uint32")
    if array.ndim != ndim or (ndim == 2 and array.shape[1] != 2):
        expected = "(m, 2)" if ndim == 2 else "(k,)"
        problems.append(f"shape {array.shape}, expected {expected}")
    if problems:
        report.add(IssueCode.BAD_TENSOR, f"{name!r} has " + " and ".join(problems))
        return None
    return array
