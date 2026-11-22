//! Python bindings: the `graph_search_rs` module.
//!
//! A thin layer. It converts between Python and Rust types and does no checks
//! of its own beyond refusing r = 0, which `compile` treats as a bug. Issues
//! cross as plain `(code, message, type_index)` tuples; `src/compiled.py`
//! turns them into Python `Issue` objects.

use graph_search_core::{
    ball_census as census, compile, BallExtractor, CanonicalKey, CheckReport, Checker,
    CompiledBall, CompiledRuleset, Edge, EmbedQuery, EmbeddingChecker, Graph, Issue, IssueCode,
    Node, Outcome,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// One issue as Python receives it: (code, message, type_index).
type PyIssue = (&'static str, String, Option<usize>);

/// One pruning removal as Python receives it:
/// (index, round, node, depth, sub_ball_edges, lost_suppliers).
type PyRemoval = (usize, usize, Node, u32, Vec<Edge>, Vec<usize>);

/// A compiled ruleset, read-only from Python.
///
/// Ball indices count distinct types after duplicates are merged; use
/// `source_indices` to map one back to its position in the file.
#[pyclass(name = "CompiledRuleset", frozen, module = "graph_search_rs")]
struct PyCompiledRuleset {
    inner: CompiledRuleset,
}

impl PyCompiledRuleset {
    /// The ball at `index`, or IndexError.
    fn ball(&self, index: usize) -> PyResult<&CompiledBall> {
        self.inner.ball(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "ball index {index} out of range for {} ball types",
                self.inner.len()
            ))
        })
    }
}

#[pymethods]
impl PyCompiledRuleset {
    #[getter]
    fn r(&self) -> u32 {
        self.inner.r()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledRuleset(r={}, ball_types={}, kept={})",
            self.inner.r(),
            self.inner.len(),
            self.inner.pruning().kept().len()
        )
    }

    /// Position in the file of each distinct ball type.
    #[getter]
    fn source_indices(&self) -> Vec<usize> {
        self.inner.source_indices().to_vec()
    }

    /// Largest root degree over all types, pruned or not.
    #[getter]
    fn max_root_degree(&self) -> usize {
        self.inner.max_root_degree()
    }

    /// Indices of the types that survive pruning, ascending.
    #[getter]
    fn kept_indices(&self) -> Vec<usize> {
        self.inner.pruning().kept().to_vec()
    }

    /// Every type pruning removed, by round, then index: `(index, round,
    /// node, depth, sub_ball_edges, lost_suppliers)`. `node` (at `depth`) is
    /// the removed type's node whose radius-(r - depth) ball no remaining
    /// type has; `sub_ball_edges` is that ball, numbered from the node as 0;
    /// `lost_suppliers` are the types that had it, removed in earlier rounds.
    fn removals(&self) -> Vec<PyRemoval> {
        self.inner
            .pruning()
            .removed()
            .iter()
            .map(|removal| {
                (
                    removal.index,
                    removal.round,
                    removal.node,
                    removal.depth,
                    removal.sub_ball.edges().to_vec(),
                    removal.lost_suppliers.clone(),
                )
            })
            .collect()
    }

    /// Indices of types with depth less than r.
    fn shallow_indices(&self) -> Vec<usize> {
        self.inner.shallow_indices()
    }

    /// Nodes grouped by distance from the root, for ball `index`.
    fn layers(&self, index: usize) -> PyResult<Vec<Vec<u32>>> {
        Ok(self.ball(index)?.layers().to_vec())
    }

    /// Degree of each node of ball `index`.
    fn degrees(&self, index: usize) -> PyResult<Vec<usize>> {
        Ok(self.ball(index)?.degrees())
    }

    fn depth(&self, index: usize) -> PyResult<usize> {
        Ok(self.ball(index)?.depth())
    }

    /// Normalized, sorted edges of ball `index`.
    fn edges(&self, index: usize) -> PyResult<Vec<Edge>> {
        Ok(self.ball(index)?.edges().to_vec())
    }

    /// Canonical key of ball `index`, as `canonical_key` returns it.
    fn canonical_key<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyTuple>> {
        key_to_py(py, self.ball(index)?.canonical_key())
    }

    /// Index of the ball type isomorphic to the ball with these edges, or
    /// None. Pruned types are found too. Raises ValueError if the edges are
    /// not a valid ball.
    fn index_of(&self, edges: Vec<Edge>) -> PyResult<Option<usize>> {
        Ok(self
            .inner
            .index_of(ball_from_edges(&edges)?.canonical_key()))
    }

    /// The validity checker on the graph with nodes `0..num_nodes`, these
    /// edges and open nodes, and an optional cap `max_open` on the open set.
    /// Raises ValueError for an invalid graph or a node out of range.
    #[pyo3(signature = (num_nodes, edges, open_nodes, max_open=None))]
    fn check(
        &self,
        num_nodes: usize,
        edges: Vec<Edge>,
        open_nodes: Vec<Node>,
        max_open: Option<usize>,
    ) -> PyResult<PyCheckReport> {
        let graph = reference_graph(num_nodes, &edges)?;
        if let Some(v) = open_nodes.iter().find(|&&v| v as usize >= num_nodes) {
            return Err(PyValueError::new_err(format!(
                "open node {v} is out of range for {num_nodes} nodes"
            )));
        }
        let inner = Checker::new(&self.inner).check(&graph, &open_nodes, max_open);
        Ok(PyCheckReport { inner })
    }

    /// The embedding check, for each of `centers` in the patch on nodes
    /// `0..num_nodes` with these edges and open nodes: the indices of the
    /// kept types its visible ball fits, ascending. An empty list means the
    /// node fails. Raises ValueError for an invalid graph or a node out of
    /// range.
    fn compatible_types(
        &self,
        num_nodes: usize,
        edges: Vec<Edge>,
        open_nodes: Vec<Node>,
        centers: Vec<Node>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let graph = reference_graph(num_nodes, &edges)?;
        let out_of_range = open_nodes
            .iter()
            .chain(&centers)
            .find(|&&v| v as usize >= num_nodes);
        if let Some(v) = out_of_range {
            return Err(PyValueError::new_err(format!(
                "node {v} is out of range for {num_nodes} nodes"
            )));
        }
        let mut open = vec![false; num_nodes];
        for &v in &open_nodes {
            open[v as usize] = true;
        }
        let distance = graph.distances_from(&open_nodes);
        let r = self.inner.r();
        let mut extractor = BallExtractor::new();
        let mut checker = EmbeddingChecker::new(&self.inner);
        Ok(centers
            .iter()
            .map(|&v| {
                let ball = extractor.extract(&graph, v, r);
                checker.compatible(&EmbedQuery::new(&ball, &open, &distance, r))
            })
            .collect())
    }
}

/// The validity checker's report, read-only from Python. `check.py` wraps it
/// with node labels.
#[pyclass(name = "CheckReport", frozen, module = "graph_search_rs")]
struct PyCheckReport {
    inner: CheckReport,
}

#[pymethods]
impl PyCheckReport {
    /// Connected, open set within the cap, and every node passes.
    #[getter]
    fn valid(&self) -> bool {
        self.inner.is_valid()
    }

    #[getter]
    fn components(&self) -> usize {
        self.inner.components
    }

    #[getter]
    fn open_count(&self) -> usize {
        self.inner.open_count
    }

    #[getter]
    fn max_open(&self) -> Option<usize> {
        self.inner.max_open
    }

    /// Failing nodes plus (components - 1): the sampler's energy.
    #[getter]
    fn energy(&self) -> usize {
        self.inner.energy()
    }

    #[getter]
    fn core_fraction(&self) -> f64 {
        self.inner.core_fraction()
    }

    /// Largest distance to the open set minus r; None in closed mode.
    #[getter]
    fn core_depth(&self) -> Option<i64> {
        self.inner.core_depth()
    }

    /// Failing nodes, ascending.
    #[getter]
    fn failing(&self) -> Vec<Node> {
        self.inner.failing()
    }

    /// Each node's class: "open", "shadow" or "core".
    fn classes(&self) -> Vec<&'static str> {
        self.inner.nodes.iter().map(|n| n.class.as_str()).collect()
    }

    /// Each node's distance to the open set, None if none is reachable.
    fn distances(&self) -> Vec<Option<u32>> {
        self.inner
            .nodes
            .iter()
            .map(|n| n.distance_to_open)
            .collect()
    }

    /// Each node's outcome as (kind, ball indices): ("matched", [i]),
    /// ("pruned", [i]), ("unmatched", []) or ("compatible", [i, ...]).
    fn outcomes(&self) -> Vec<(&'static str, Vec<usize>)> {
        self.inner
            .nodes
            .iter()
            .map(|n| match &n.outcome {
                Outcome::Matched(i) => ("matched", vec![*i]),
                Outcome::Pruned(i) => ("pruned", vec![*i]),
                Outcome::Unmatched => ("unmatched", Vec::new()),
                Outcome::Compatible(types) => ("compatible", types.clone()),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckReport(valid={}, nodes={}, failing={}, components={})",
            self.inner.is_valid(),
            self.inner.num_nodes(),
            self.inner.failing().len(),
            self.inner.components
        )
    }
}

/// A canonical key as Python sees it: `(layer_sizes, edges)`, all tuples, so
/// it is hashable and can key a dict.
fn key_to_py<'py>(py: Python<'py>, key: &CanonicalKey) -> PyResult<Bound<'py, PyTuple>> {
    let layer_sizes = PyTuple::new(py, key.layer_sizes().iter().copied())?;
    let edges = PyTuple::new(py, key.edges().iter().copied())?;
    PyTuple::new(py, [layer_sizes.into_any(), edges.into_any()])
}

/// Build a ball from edges as written, or raise ValueError listing every
/// structural problem.
fn ball_from_edges(edges: &[Edge]) -> PyResult<CompiledBall> {
    CompiledBall::from_edges(edges).map_err(|issues| {
        let details: Vec<String> = issues
            .iter()
            .map(|issue| format!("{}: {}", issue.code.as_str(), issue.message))
            .collect();
        PyValueError::new_err(format!("invalid ball: {}", details.join("; ")))
    })
}

/// Convert one Rust issue to the tuple Python receives.
fn to_py_issue(issue: &Issue) -> PyIssue {
    (issue.code.as_str(), issue.message.clone(), issue.type_index)
}

/// Compile a ruleset from r and each type's edges, as written.
///
/// Returns `(compiled, issues)`: `compiled` is None whenever an issue is an
/// error. Raises ValueError if r is 0, instead of letting `compile` panic.
/// Node ids that do not fit in 32 bits raise OverflowError during argument
/// conversion; Python's `check_edge_format` rejects them first, so only code
/// that skips it can reach that.
#[pyfunction]
fn compile_ruleset(
    r: u32,
    types: Vec<Vec<Edge>>,
) -> PyResult<(Option<PyCompiledRuleset>, Vec<PyIssue>)> {
    if r == 0 {
        return Err(PyValueError::new_err("r must be at least 1, got 0"));
    }
    let outcome = compile(r, &types);
    let issues = outcome.issues.iter().map(to_py_issue).collect();
    Ok((
        outcome.ruleset.map(|inner| PyCompiledRuleset { inner }),
        issues,
    ))
}

/// Canonical key of the ball with these edges, as written: equal keys mean
/// isomorphic balls, root to root. Returns `(layer_sizes, edges)` as nested
/// tuples. Raises ValueError if the edges are not a valid ball.
#[pyfunction]
fn canonical_key<'py>(py: Python<'py>, edges: Vec<Edge>) -> PyResult<Bound<'py, PyTuple>> {
    key_to_py(py, ball_from_edges(&edges)?.canonical_key())
}

/// A reference graph on nodes `0..num_nodes`, degree cap from its own edges,
/// or ValueError naming the problem.
fn reference_graph(num_nodes: usize, edges: &[Edge]) -> PyResult<Graph> {
    Graph::from_edges_with_own_cap(num_nodes, edges)
        .map_err(|e| PyValueError::new_err(format!("invalid graph: {e}")))
}

/// The radius-r ball around `center` in the graph on nodes `0..num_nodes`
/// with these edges: `(nodes, depths, edges)`, where `nodes[i]` is the graph
/// node numbered i in the ball (the center is 0, then BFS order), `depths[i]`
/// its distance from the center, and `edges` the induced edges in ball
/// numbering. Raises ValueError for an invalid graph or center.
#[pyfunction]
fn extract_ball(
    num_nodes: usize,
    edges: Vec<Edge>,
    center: Node,
    r: u32,
) -> PyResult<(Vec<Node>, Vec<u32>, Vec<Edge>)> {
    let graph = reference_graph(num_nodes, &edges)?;
    if center as usize >= num_nodes {
        return Err(PyValueError::new_err(format!(
            "center {center} is out of range for {num_nodes} nodes"
        )));
    }
    let ball = BallExtractor::new().extract(&graph, center, r);
    Ok((
        ball.nodes().to_vec(),
        ball.depths().to_vec(),
        ball.edges().to_vec(),
    ))
}

/// Every node's radius-r ball, grouped by shape: `(canonical_edges, count,
/// example_center)` per shape, most common first, ties by canonical key.
/// Raises ValueError for an invalid graph.
#[pyfunction]
fn ball_census(
    num_nodes: usize,
    edges: Vec<Edge>,
    r: u32,
) -> PyResult<Vec<(Vec<Edge>, usize, Node)>> {
    let graph = reference_graph(num_nodes, &edges)?;
    Ok(census(&graph, r)
        .into_iter()
        .map(|entry| {
            let edges = entry.ball.canonical_key().edges().to_vec();
            (edges, entry.count, entry.example_center)
        })
        .collect())
}

/// Every code compiling can report, with its severity: `(code, severity)` pairs.
/// Python's tests compare these with `report.COMPILE_CODES` and `report.SEVERITY`.
#[pyfunction]
fn issue_codes() -> Vec<(&'static str, &'static str)> {
    IssueCode::ALL
        .iter()
        .map(|code| (code.as_str(), code.severity().as_str()))
        .collect()
}

#[pymodule]
fn graph_search_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompiledRuleset>()?;
    m.add_class::<PyCheckReport>()?;
    m.add_function(wrap_pyfunction!(compile_ruleset, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ball, m)?)?;
    m.add_function(wrap_pyfunction!(ball_census, m)?)?;
    m.add_function(wrap_pyfunction!(issue_codes, m)?)?;
    Ok(())
}
