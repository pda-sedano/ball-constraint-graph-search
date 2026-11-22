//! The enumeration baseline: which graphs in a list are valid in closed mode.
//!
//! The list is graph6, one graph per line, as geng writes it (Python runs
//! geng and caches its output). Every graph is checked with every node core,
//! stopping at the first failure, and each valid one comes back with its
//! automorphism count: the sampler's ground truth, since in equilibrium it
//! visits each unlabelled graph in proportion to n!/|Aut|.

use std::fmt;
use std::io::{self, BufRead};

use crate::canonical::automorphism_count;
use crate::check::Checker;
use crate::graph::Graph;
use crate::graph6::{parse_graph6, Graph6Error};
use crate::ruleset::CompiledRuleset;

/// A valid graph from the list.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidGraph {
    /// Its line number in the list, from 0.
    pub line: usize,
    /// The graph6 line itself.
    pub graph6: String,
    /// |Aut|, exact below 2^53.
    pub automorphisms: f64,
}

/// What enumerating a list found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Enumeration {
    /// Graphs read.
    pub candidates: usize,
    /// The valid ones, in list order.
    pub valid: Vec<ValidGraph>,
}

/// Why a list could not be read.
#[derive(Debug)]
pub enum EnumerationError {
    Io(io::Error),
    /// A line that is not graph6, by line number from 0.
    Graph6 {
        line: usize,
        error: Graph6Error,
    },
}

impl fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnumerationError::Io(e) => write!(f, "{e}"),
            EnumerationError::Graph6 { line, error } => write!(f, "line {line}: {error}"),
        }
    }
}

impl std::error::Error for EnumerationError {}

impl From<io::Error> for EnumerationError {
    fn from(e: io::Error) -> Self {
        EnumerationError::Io(e)
    }
}

/// Check every graph in `list` against `ruleset` in closed mode. Blank lines
/// are skipped.
pub fn valid_graphs(
    ruleset: &CompiledRuleset,
    list: impl BufRead,
) -> Result<Enumeration, EnumerationError> {
    let mut checker = Checker::new(ruleset);
    let mut found = Enumeration::default();
    for (number, line) in list.lines().enumerate() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let (n, edges) =
            parse_graph6(text.as_bytes()).map_err(|error| EnumerationError::Graph6 {
                line: number,
                error,
            })?;
        let graph = Graph::from_edges_with_own_cap(n, &edges)
            .expect("graph6 has no self-loops or repeated edges");
        found.candidates += 1;
        if checker.passes(&graph, &[], None) {
            found.valid.push(ValidGraph {
                line: number,
                graph6: text.to_string(),
                automorphisms: automorphism_count(&graph),
            });
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ruleset, PATH_BALL};

    #[test]
    fn cycles_are_the_valid_graphs_for_the_path_ball() {
        //! A hand-made list: a 6-cycle, a blank line, a path and a 5-cycle.
        use crate::ball::Edge;
        fn encode(n: usize, edges: &[Edge]) -> String {
            let mut bits = vec![false; n * (n - 1) / 2];
            for &(a, b) in edges {
                let (i, j) = (a.min(b) as usize, a.max(b) as usize);
                bits[j * (j - 1) / 2 + i] = true;
            }
            let mut out = vec![(n + 63) as u8];
            for chunk in bits.chunks(6) {
                let mut byte = 0u8;
                for (k, &b) in chunk.iter().enumerate() {
                    byte |= (b as u8) << (5 - k);
                }
                out.push(byte + 63);
            }
            String::from_utf8(out).expect("ascii")
        }
        let cycle =
            |n: usize| -> Vec<Edge> { (0..n as u32).map(|v| (v, (v + 1) % n as u32)).collect() };
        let path: Vec<Edge> = (1..7).map(|v| (v - 1, v)).collect();
        let list = format!(
            "{}\n\n{}\n{}\n",
            encode(6, &cycle(6)),
            encode(7, &path),
            encode(5, &cycle(5))
        );
        let found = valid_graphs(&ruleset(2, &[PATH_BALL]), list.as_bytes()).expect("readable");
        assert_eq!(found.candidates, 3);
        assert_eq!(found.valid.len(), 1);
        assert_eq!(
            (found.valid[0].line, found.valid[0].automorphisms),
            (0, 12.0)
        );
    }

    #[test]
    fn bad_lines_are_reported_with_their_number() {
        let error = valid_graphs(&ruleset(2, &[PATH_BALL]), "Bw\nB w\n".as_bytes()).unwrap_err();
        assert!(
            matches!(error, EnumerationError::Graph6 { line: 1, .. }),
            "{error}"
        );
    }
}
