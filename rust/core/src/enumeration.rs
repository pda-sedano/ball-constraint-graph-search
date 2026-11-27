//! The enumeration baseline: which graphs in a list are valid in closed mode,
//! and which (graph, open set) pairs are valid in open mode.
//!
//! The list is graph6, one graph per line, as geng writes it (Python runs
//! geng and caches its output). In closed mode every graph is checked with
//! every node core, stopping at the first failure, and each valid one comes
//! back with its automorphism count: the sampler's ground truth, since in
//! equilibrium it visits each unlabelled graph in proportion to n!/|Aut|. In
//! open mode each graph is checked with every set of m open nodes, and the
//! valid pairs come back once per isomorphism class, keyed as the sampler
//! keys them, with the number of symmetries that keep the open set in place.

use std::collections::HashSet;
use std::fmt;
use std::io::{self, BufRead};

use crate::ball::Node;
use crate::canonical::{automorphism_count, canonical_open_graph};
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

/// A valid graph with an open set, once per isomorphism class.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidOpenGraph {
    /// The canonical graph6 and the open nodes' canonical numbers, as in
    /// [`crate::CanonicalGraph`].
    pub graph6: String,
    pub open: Vec<Node>,
    /// Symmetries that keep the open set in place.
    pub automorphisms: f64,
}

/// What enumerating a list with open sets found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpenEnumeration {
    /// Graphs read.
    pub candidates: usize,
    /// (graph, open set) pairs checked.
    pub open_sets: usize,
    /// The valid classes, in order of first appearance.
    pub valid: Vec<ValidOpenGraph>,
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

/// Check every graph in `list` with every set of exactly `m` open nodes.
/// Blank lines are skipped, and graphs with fewer than m nodes contribute
/// nothing. Each valid class is kept once, keyed by its canonical form.
pub fn valid_open_graphs(
    ruleset: &CompiledRuleset,
    list: impl BufRead,
    m: usize,
) -> Result<OpenEnumeration, EnumerationError> {
    let mut checker = Checker::new(ruleset);
    let mut found = OpenEnumeration::default();
    let mut seen: HashSet<String> = HashSet::new();
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
        if m > n {
            continue;
        }
        // The m-subsets of 0..n in lexicographic order.
        let mut chosen: Vec<usize> = (0..m).collect();
        loop {
            found.open_sets += 1;
            let open: Vec<Node> = chosen.iter().map(|&v| v as Node).collect();
            if checker.passes(&graph, &open, Some(m)) {
                let mut is_open = vec![false; n];
                for &v in &chosen {
                    is_open[v] = true;
                }
                let form = canonical_open_graph(&graph, &is_open);
                if seen.insert(form.key()) {
                    found.valid.push(ValidOpenGraph {
                        graph6: form.graph6(),
                        open: form.open,
                        automorphisms: form.automorphisms,
                    });
                }
            }
            // Advance to the next subset: bump the rightmost index that can move.
            let Some(i) = (0..m).rev().find(|&i| chosen[i] < n - m + i) else {
                break;
            };
            chosen[i] += 1;
            for j in i + 1..m {
                chosen[j] = chosen[j - 1] + 1;
            }
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Edge;
    use crate::graph6::write_graph6;
    use crate::testing::{ruleset, PATH_BALL};

    fn cycle(n: usize) -> Vec<Edge> {
        (0..n as u32).map(|v| (v, (v + 1) % n as u32)).collect()
    }

    #[test]
    fn open_path_ball_states_on_six_nodes() {
        //! 6-cycle, 6-path, a relabeled 6-cycle and a star: with m = 2 the
        //! valid classes are the cycle with an open pair at distance 1, 2 or 3
        //! (|Aut| 2, 2, 4) and the path with both ends open (|Aut| 2), each
        //! once although the cycle appears twice. With m = 0 only the cycle.
        let path: Vec<Edge> = (0..5).map(|v| (v, v + 1)).collect();
        let relabeled: Vec<Edge> = [(0, 3), (3, 1), (1, 5), (5, 2), (2, 4), (4, 0)].to_vec();
        let star: Vec<Edge> = (1..6).map(|v| (0, v)).collect();
        let list: String = [cycle(6), path, relabeled, star]
            .iter()
            .map(|edges| write_graph6(6, edges) + "\n")
            .collect();
        let rules = ruleset(2, &[PATH_BALL]);
        let found = valid_open_graphs(&rules, list.as_bytes(), 2).expect("readable");
        assert_eq!((found.candidates, found.open_sets), (4, 4 * 15));
        let mut auts: Vec<f64> = found.valid.iter().map(|v| v.automorphisms).collect();
        auts.sort_by(f64::total_cmp);
        assert_eq!(auts, [2.0, 2.0, 2.0, 4.0]);
        assert!(found.valid.iter().all(|v| v.open == [4, 5]));
        let closed = valid_open_graphs(&rules, list.as_bytes(), 0).expect("readable");
        assert_eq!(closed.open_sets, 4);
        assert_eq!(closed.valid.len(), 1);
        assert_eq!(closed.valid[0].automorphisms, 12.0);
        let too_many = valid_open_graphs(&rules, list.as_bytes(), 7).expect("readable");
        assert_eq!((too_many.candidates, too_many.open_sets), (4, 0));
    }

    #[test]
    fn every_subset_is_tried_once() {
        //! The empty graph on 5 nodes with m = 0..=5: C(5, m) open sets each.
        let list = write_graph6(5, &[]) + "\n";
        let rules = ruleset(2, &[PATH_BALL]);
        for (m, expected) in [(0, 1), (1, 5), (2, 10), (3, 10), (4, 5), (5, 1)] {
            let found = valid_open_graphs(&rules, list.as_bytes(), m).expect("readable");
            assert_eq!(found.open_sets, expected, "m = {m}");
        }
    }

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
