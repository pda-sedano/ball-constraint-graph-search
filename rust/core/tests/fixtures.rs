//! The ruleset files shared with the Python tests, compiled directly in Rust.
//!
//! Python loads these fixtures without issues (they are well-formed); compiling
//! must report the listed code. The same table is `COMPILE_FIXTURES` in
//! `tests/test_io.py`, so the two suites cover the same files.

use std::path::{Path, PathBuf};

use graph_search_core::{compile, Edge, IssueCode, Severity};
use serde_json::Value;

const COMPILE_FIXTURES: &[(&str, IssueCode)] = &[
    ("self_loop.json", IssueCode::SelfLoop),
    ("repeated_edge.json", IssueCode::RepeatedEdge),
    ("reversed_edge.json", IssueCode::RepeatedEdge),
    ("node_gap.json", IssueCode::NodeGap),
    ("disconnected.json", IssueCode::Disconnected),
    ("rootless.json", IssueCode::NodeGap), // no node 0: a gap...
    ("rootless.json", IssueCode::Disconnected), // ...and nothing reachable
    ("beyond_radius.json", IssueCode::BeyondRadius),
    ("shallow.json", IssueCode::ShallowType),
    ("exact_duplicate.json", IssueCode::ExactDuplicate),
    // Needs step 3 (canonical form): skip this entry until then.
    ("isomorphic_duplicate.json", IssueCode::IsomorphicDuplicate),
    ("two_bad_types.json", IssueCode::SelfLoop), // and NodeGap at index 2
];

const CLEAN_RULESETS: &[&str] = &["star-only.json", "path-ball-r2.json", "dimer-r1.json"];

/// The repository root, two levels above this crate.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Read a ruleset file's r and each type's edge list, with serde_json.
///
/// No validation beyond what parsing needs: these files are known to be
/// well-formed. Panic with the path in the message on anything unexpected.
fn load(path: &Path) -> (u32, Vec<Vec<Edge>>) {
    let fail = |what: &str| -> ! { panic!("{}: {what}", path.display()) };
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&e.to_string()));
    let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| fail(&e.to_string()));

    let r = doc["r"]
        .as_u64()
        .unwrap_or_else(|| fail("r is not an unsigned int"));
    let node = |value: &Value| -> u32 {
        let id = value
            .as_u64()
            .unwrap_or_else(|| fail("node is not an unsigned int"));
        u32::try_from(id).unwrap_or_else(|_| fail("node does not fit in u32"))
    };
    let types = doc["types"]
        .as_array()
        .unwrap_or_else(|| fail("types is not a list"))
        .iter()
        .map(|t| {
            t["edges"]
                .as_array()
                .unwrap_or_else(|| fail("edges is not a list"))
                .iter()
                .map(|e| (node(&e[0]), node(&e[1])))
                .collect()
        })
        .collect();
    (
        u32::try_from(r).unwrap_or_else(|_| fail("r does not fit in u32")),
        types,
    )
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

#[test]
fn compile_fixtures_report_their_code() {
    for &(name, code) in COMPILE_FIXTURES {
        if code == IssueCode::IsomorphicDuplicate {
            continue; // needs step 3 (canonical form)
        }
        let (r, types) = load(&fixture(name));
        let outcome = compile(r, &types);
        assert!(
            outcome.issues.iter().any(|issue| issue.code == code),
            "{name}: expected {code:?}, got {:?}",
            outcome.issues
        );
        if code.severity() == Severity::Error {
            assert!(
                outcome.ruleset.is_none(),
                "{name}: errors must give no ruleset"
            );
        }
    }
}

#[test]
fn clean_rulesets_compile_without_issues() {
    for name in CLEAN_RULESETS {
        let (r, types) = load(&repo_root().join("rulesets").join(name));
        let outcome = compile(r, &types);
        assert_eq!(outcome.issues, [], "{name}");
        assert!(outcome.ruleset.is_some(), "{name}");
    }
}

#[test]
fn two_bad_types_reports_both() {
    let (r, types) = load(&fixture("two_bad_types.json"));
    let found: Vec<_> = compile(r, &types)
        .issues
        .iter()
        .map(|issue| (issue.code, issue.type_index))
        .collect();
    assert_eq!(
        found,
        [
            (IssueCode::SelfLoop, Some(0)),
            (IssueCode::NodeGap, Some(2))
        ]
    );
}
