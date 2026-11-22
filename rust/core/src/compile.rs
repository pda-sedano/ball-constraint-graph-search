//! Compiling: every check on the ball graphs, then a [`CompiledRuleset`].

use std::collections::HashMap;

use crate::ball::{check_structure, CompiledBall, Edge};
use crate::canonical::CanonicalKey;
use crate::issue::{Issue, IssueCode};
use crate::ruleset::CompiledRuleset;

/// The result of compiling: every issue found, and the compiled ruleset if
/// none of them is an error.
#[derive(Debug, Clone)]
pub struct CompileOutcome {
    /// None whenever `issues` contains an error.
    pub ruleset: Option<CompiledRuleset>,
    /// In the order found: per-type structure, then radius, then duplicates,
    /// then pruning.
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
///   3. Across compiled balls, `merge_duplicates`.
///   4. If any issue is an error, `ruleset` is None. Otherwise build the
///      `CompiledRuleset` from the merged balls and their source indices,
///      which runs visible sub-ball pruning.
///   5. `pruning_issues`: a warning per pruned type, and one more if none
///      are left. Pruned types stay in the ruleset, marked as removed.
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

    let (kept, duplicate_issues) = merge_duplicates(compiled);
    issues.extend(duplicate_issues);

    let ruleset = if issues.iter().any(Issue::is_error) {
        None
    } else {
        let (source_indices, balls) = kept.into_iter().unzip();
        let ruleset = CompiledRuleset::new(r, balls, source_indices);
        issues.extend(pruning_issues(&ruleset));
        Some(ruleset)
    };
    CompileOutcome { ruleset, issues }
}

/// One `PrunedType` warning per removed type, at its position in the file and
/// in order of removal, then `Unrealizable` if no type is left. Messages name
/// other types by file position too.
fn pruning_issues(ruleset: &CompiledRuleset) -> Vec<Issue> {
    let source = ruleset.source_indices();
    let pruning = ruleset.pruning();
    let mut issues: Vec<Issue> = pruning
        .removed()
        .iter()
        .map(|removal| {
            let suppliers: Vec<String> = removal
                .lost_suppliers
                .iter()
                .map(|&i| source[i].to_string())
                .collect();
            let whose = match suppliers.as_slice() {
                [] => "that no type has around its root".to_string(),
                [one] => format!(
                    "that only type {one} had around its root, removed in an earlier round"
                ),
                [rest @ .., last] => format!(
                    "that only types {} and {last} had around their roots, removed in earlier rounds",
                    rest.join(", ")
                ),
            };
            Issue::new(
                IssueCode::PrunedType,
                format!(
                    "round {}: node {} at depth {} sees a radius-{} ball {whose}",
                    removal.round, removal.node, removal.depth, removal.radius
                ),
                Some(source[removal.index]),
            )
        })
        .collect();
    if !ruleset.is_empty() && pruning.kept().is_empty() {
        let which = match ruleset.len() {
            1 => "the only type".to_string(),
            n => format!("all {n} distinct types"),
        };
        issues.push(Issue::new(
            IssueCode::Unrealizable,
            format!(
                "pruning removed {which}, so no graph can have every node's ball in this ruleset"
            ),
            None,
        ));
    }
    issues
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

/// Keep the first of each group of isomorphic balls (equal canonical keys).
///
/// `balls` pairs each compiled ball with its position in the file, in file
/// order. Returns the kept pairs in the same order, plus one warning per
/// dropped copy, at the copy's index and naming the index it duplicates:
/// `ExactDuplicate` when the normalized edges are identical too,
/// `IsomorphicDuplicate` when only the shape is.
fn merge_duplicates(balls: Vec<(usize, CompiledBall)>) -> (Vec<(usize, CompiledBall)>, Vec<Issue>) {
    let mut position_of: HashMap<CanonicalKey, usize> = HashMap::new();
    let mut kept: Vec<(usize, CompiledBall)> = Vec::new();
    let mut issues = Vec::new();
    for (index, ball) in balls {
        if let Some(&position) = position_of.get(ball.canonical_key()) {
            let (first, first_ball) = &kept[position];
            let issue = if first_ball.edges() == ball.edges() {
                Issue::new(
                    IssueCode::ExactDuplicate,
                    format!("same edges as type {first}; merged into it"),
                    Some(index),
                )
            } else {
                Issue::new(
                    IssueCode::IsomorphicDuplicate,
                    format!("isomorphic to type {first}; merged into it"),
                    Some(index),
                )
            };
            issues.push(issue);
        } else {
            position_of.insert(ball.canonical_key().clone(), kept.len());
            kept.push((index, ball));
        }
    }
    (kept, issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path seen from its middle: supplies its own sub-balls, so pruning
    /// keeps it and these tests see only the issues they are about.
    const PATH_BALL: &[Edge] = &[(0, 1), (0, 2), (1, 3), (2, 4)];
    /// The design doc's star example, which pruning removes.
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
        let outcome = compile(2, &types(&[PATH_BALL]));
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
        let outcome = compile(2, &types(&[PATH_BALL, &[(0, 1)]]));
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
            &types(&[
                PATH_BALL,
                &[(2, 4), (1, 3), (0, 2), (0, 1)],
                &[(1, 0), (4, 2), (3, 1), (2, 0)],
            ]),
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
    fn isomorphic_duplicates_merged_keeping_first() {
        //! The path ball with its arms numbered differently is the same shape.
        let relabeled: &[Edge] = &[(0, 3), (3, 4), (0, 1), (1, 2)];
        let outcome = compile(2, &types(&[PATH_BALL, relabeled, PATH_BALL]));
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::IsomorphicDuplicate, Some(1)),
                (IssueCode::ExactDuplicate, Some(2)),
            ]
        );
        let ruleset = outcome.ruleset.expect("warnings only");
        assert_eq!(ruleset.len(), 1);
        assert_eq!(ruleset.source_indices(), [0]);
    }

    #[test]
    fn index_of_finds_any_isomorphic_copy() {
        let ruleset = compile(2, &types(&[STAR, PATH_BALL]))
            .ruleset
            .expect("clean");
        let relabeled = CompiledBall::from_checked_edges(&[(0, 3), (3, 4), (0, 1), (1, 2)]);
        assert_eq!(ruleset.index_of(relabeled.canonical_key()), Some(1));
        let other = CompiledBall::from_checked_edges(&[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(ruleset.index_of(other.canonical_key()), None);
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
    fn pruned_types_are_warned_about_and_kept_in_the_ruleset() {
        //! Type 2 is a leaf beside an arm: its leaf needs a root of degree 1,
        //! which only the star (type 1) has, and round 1 removes the star.
        let leaf_and_arm: &[Edge] = &[(0, 1), (0, 2), (2, 3)];
        let outcome = compile(2, &types(&[PATH_BALL, STAR, leaf_and_arm]));
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::PrunedType, Some(1)),
                (IssueCode::PrunedType, Some(2))
            ]
        );
        assert_eq!(
            outcome.issues[1].message,
            "round 2: node 1 at depth 1 sees a radius-1 ball that only type 1 had around its \
             root, removed in an earlier round"
        );
        let ruleset = outcome.ruleset.expect("warnings only");
        assert_eq!(ruleset.len(), 3);
        assert_eq!(ruleset.pruning().kept(), [0]);
    }

    #[test]
    fn pruning_names_types_by_file_position() {
        //! A duplicate first shifts every later type's index by one.
        let leaf_and_arm: &[Edge] = &[(0, 1), (0, 2), (2, 3)];
        let outcome = compile(2, &types(&[PATH_BALL, PATH_BALL, STAR, leaf_and_arm]));
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::ExactDuplicate, Some(1)),
                (IssueCode::PrunedType, Some(2)),
                (IssueCode::PrunedType, Some(3))
            ]
        );
        assert!(outcome.issues[1].message.contains("that no type has"));
        assert!(outcome.issues[2].message.contains("only type 2 had"));
    }

    #[test]
    fn nothing_left_is_unrealizable() {
        let outcome = compile(2, &types(&[STAR]));
        assert_eq!(
            found(&outcome),
            [
                (IssueCode::PrunedType, Some(0)),
                (IssueCode::Unrealizable, None)
            ]
        );
        assert_eq!(
            outcome.issues[0].message,
            "round 1: node 1 at depth 1 sees a radius-1 ball that no type has around its root"
        );
        let ruleset = outcome.ruleset.expect("warnings only: the file is fine");
        assert_eq!(ruleset.pruning().kept(), [] as [usize; 0]);
    }

    #[test]
    #[should_panic(expected = "r must be at least 1")]
    fn zero_radius_panics() {
        compile(0, &types(&[STAR]));
    }
}
