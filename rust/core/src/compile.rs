//! Compiling: every check on the ball graphs, then a [`CompiledRuleset`].

use std::collections::HashMap;

use crate::ball::{check_structure, CompiledBall, Edge};
use crate::issue::{Issue, IssueCode};
use crate::ruleset::CompiledRuleset;

/// The result of compiling: every issue found, and the compiled ruleset if
/// none of them is an error.
#[derive(Debug, Clone)]
pub struct CompileOutcome {
    /// None whenever `issues` contains an error.
    pub ruleset: Option<CompiledRuleset>,
    /// In the order found: per-type structure, then radius, then duplicates.
    pub issues: Vec<Issue>,
}

impl CompileOutcome {
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(Issue::is_error)
    }
}

/// Compile a ruleset from its radius and its types' edge lists, as written.
///
/// Steps:
///   1. For each type, `check_structure`. Types that pass become
///      `CompiledBall`s; the rest are left out of later steps, so one bad
///      type does not hide problems in the others.
///   2. For each compiled ball, `check_radius`.
///   3. Across compiled balls, `merge_exact_duplicates`.
///   4. If any issue is an error, `ruleset` is None. Otherwise build the
///      `CompiledRuleset` from the merged balls and their source indices.
///
/// An empty `types` slice compiles to an empty ruleset with no issues; Python
/// already warned about it.
///
/// # Panics
///
/// If `r == 0`, with a message containing "r must be at least 1". Python's
/// `check_ruleset` reports a bad r for files, and the bindings refuse it
/// before calling here, so reaching this is a bug.
pub fn compile(r: u32, types: &[Vec<Edge>]) -> CompileOutcome {
    assert!(r >= 1, "r must be at least 1, got {r}");

    let mut issues = Vec::new();
    let mut compiled = Vec::new();
    for (index, edges) in types.iter().enumerate() {
        let found = check_structure(edges, index);
        if found.is_empty() {
            compiled.push((index, CompiledBall::from_checked_edges(edges)));
        }
        issues.extend(found);
    }

    issues.extend(
        compiled
            .iter()
            .filter_map(|(index, ball)| check_radius(r, *index, ball)),
    );

    let (kept, duplicate_issues) = merge_exact_duplicates(compiled);
    issues.extend(duplicate_issues);

    let ruleset = if issues.iter().any(Issue::is_error) {
        None
    } else {
        let (source_indices, balls) = kept.into_iter().unzip();
        Some(CompiledRuleset::new(r, balls, source_indices))
    };
    CompileOutcome { ruleset, issues }
}

/// `BeyondRadius` (error) if the ball's depth is greater than r,
/// `ShallowType` (warning) if less, None if equal.
fn check_radius(r: u32, type_index: usize, ball: &CompiledBall) -> Option<Issue> {
    let depth = ball.depth() as u64;
    let r = u64::from(r);
    if depth > r {
        Some(Issue::new(
            IssueCode::BeyondRadius,
            format!("depth {depth} is greater than r = {r}"),
            Some(type_index),
        ))
    } else if depth < r {
        Some(Issue::new(
            IssueCode::ShallowType,
            format!(
                "depth {depth} is less than r = {r}; it applies only when n = {}",
                ball.num_nodes()
            ),
            Some(type_index),
        ))
    } else {
        None
    }
}

/// Keep the first of each group of balls with identical normalized edges.
///
/// `balls` pairs each compiled ball with its position in the file, in file
/// order. Returns the kept pairs in the same order, plus one
/// `ExactDuplicate` warning per dropped copy, at the copy's index and naming
/// the index it duplicates.
fn merge_exact_duplicates(
    balls: Vec<(usize, CompiledBall)>,
) -> (Vec<(usize, CompiledBall)>, Vec<Issue>) {
    let mut first_index: HashMap<Vec<Edge>, usize> = HashMap::new();
    let mut kept = Vec::new();
    let mut issues = Vec::new();
    for (index, ball) in balls {
        if let Some(&first) = first_index.get(ball.edges()) {
            issues.push(Issue::new(
                IssueCode::ExactDuplicate,
                format!("same edges as type {first}; merged into it"),
                Some(index),
            ));
        } else {
            first_index.insert(ball.edges().to_vec(), index);
            kept.push((index, ball));
        }
    }
    (kept, issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAR: &[Edge] = &[(0, 1), (1, 2), (1, 3)];

    fn types(lists: &[&[Edge]]) -> Vec<Vec<Edge>> {
        lists.iter().map(|edges| edges.to_vec()).collect()
    }

    fn found(outcome: &CompileOutcome) -> Vec<(IssueCode, Option<usize>)> {
        outcome
            .issues
            .iter()
            .map(|i| (i.code, i.type_index))
            .collect()
    }

    #[test]
    fn clean_ruleset_has_no_issues() {
        let outcome = compile(2, &types(&[STAR]));
        assert_eq!(outcome.issues, []);
        let ruleset = outcome.ruleset.expect("no errors, so a ruleset");
        assert_eq!(ruleset.len(), 1);
        assert_eq!(ruleset.source_indices(), [0]);
        assert_eq!(ruleset.r(), 2);
    }

    #[test]
    fn beyond_radius_is_error_and_gives_no_ruleset() {
        let outcome = compile(2, &types(&[&[(0, 1), (1, 2), (2, 3)]]));
        assert_eq!(found(&outcome), [(IssueCode::BeyondRadius, Some(0))]);
        assert!(outcome.has_errors());
        assert!(outcome.ruleset.is_none());
    }

    #[test]
    fn shallow_is_warning_and_kept() {
        let outcome = compile(2, &types(&[STAR, &[(0, 1)]]));
        assert_eq!(found(&outcome), [(IssueCode::ShallowType, Some(1))]);
        assert!(!outcome.has_errors());
        let ruleset = outcome.ruleset.expect("warnings only");
        assert_eq!(ruleset.len(), 2);
        assert_eq!(ruleset.shallow_indices(), [1]);
    }

    #[test]
    fn exact_duplicates_merged_keeping_first() {
        let outcome = compile(
            2,
            &types(&[STAR, &[(3, 1), (2, 1), (1, 0)], &[(1, 3), (0, 1), (2, 1)]]),
        );
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::ExactDuplicate, Some(1)),
                (IssueCode::ExactDuplicate, Some(2))
            ]
        );
        let ruleset = outcome.ruleset.expect("warnings only");
        assert_eq!(ruleset.len(), 1);
        assert_eq!(ruleset.source_indices(), [0]);
    }

    #[test]
    fn structural_errors_in_several_types_all_reported() {
        let path3: &[Edge] = &[(0, 1), (1, 2), (2, 3)];
        let outcome = compile(2, &types(&[&[(0, 0)], path3, &[(0, 1), (1, 3)]]));
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::SelfLoop, Some(0)),
                (IssueCode::NodeGap, Some(2)),
                (IssueCode::BeyondRadius, Some(1)),
            ]
        );
        assert!(outcome.ruleset.is_none());
    }

    #[test]
    fn empty_ruleset_compiles_empty() {
        let outcome = compile(1, &[]);
        assert_eq!(outcome.issues, []);
        let ruleset = outcome.ruleset.expect("nothing wrong");
        assert!(ruleset.is_empty());
        assert_eq!(ruleset.max_root_degree(), 0);
    }

    #[test]
    fn max_root_degree_is_largest_over_types() {
        let outcome = compile(1, &types(&[&[(0, 1)], &[(0, 1), (0, 2), (0, 3)]]));
        assert_eq!(outcome.ruleset.expect("clean").max_root_degree(), 3);
    }

    #[test]
    #[should_panic(expected = "r must be at least 1")]
    fn zero_radius_panics() {
        compile(0, &types(&[STAR]));
    }
}
