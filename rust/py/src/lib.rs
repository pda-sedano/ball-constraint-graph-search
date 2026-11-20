//! Python bindings: the `graph_search_rs` module.
//!
//! A thin layer. It converts between Python and Rust types and does no checks
//! of its own beyond refusing r = 0, which `compile` treats as a bug. Issues
//! cross as plain `(code, message, type_index)` tuples; `src/compiled.py`
//! turns them into Python `Issue` objects.

use graph_search_core::{compile, CompiledBall, CompiledRuleset, Edge, Issue, IssueCode};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// One issue as Python receives it: (code, message, type_index).
type PyIssue = (&'static str, String, Option<usize>);

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
            "CompiledRuleset(r={}, ball_types={})",
            self.inner.r(),
            self.inner.len()
        )
    }

    /// Position in the file of each distinct ball type.
    #[getter]
    fn source_indices(&self) -> Vec<usize> {
        self.inner.source_indices().to_vec()
    }

    /// Largest root degree over all types.
    #[getter]
    fn max_root_degree(&self) -> usize {
        self.inner.max_root_degree()
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
    m.add_function(wrap_pyfunction!(compile_ruleset, m)?)?;
    m.add_function(wrap_pyfunction!(issue_codes, m)?)?;
    Ok(())
}
