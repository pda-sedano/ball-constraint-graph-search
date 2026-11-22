//! Issues found while compiling a ruleset.
//!
//! Only the codes for checks on the ball graphs live here. Format codes
//! (malformed edges, negative nodes, missing keys, ...) belong to Python alone.
//! Each code's string and severity must equal Python's `report.IssueCode` and
//! `report.SEVERITY`; `tests/test_compiled.py` checks this through the
//! bindings' `issue_codes()`.

/// How serious an issue is. An error means no compiled ruleset is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    /// The string Python uses: `"error"` or `"warning"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// Every kind of issue compiling can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueCode {
    // Single type, graph structure (ball::check_structure)
    SelfLoop,
    /// Includes `[1, 0]` after `[0, 1]`.
    RepeatedEdge,
    /// Nodes are not exactly 0..k-1. Includes a missing root.
    NodeGap,
    /// Some node is unreachable from the root. With no node 0, all of them.
    Disconnected,

    // Single type, relative to r (compile)
    BeyondRadius,
    ShallowType,

    // Across types (compile)
    /// Identical normalized edge lists.
    ExactDuplicate,
    /// Same ball up to root-preserving isomorphism, edges written differently.
    IsomorphicDuplicate,

    // Pruning (prune)
    /// Removed by visible sub-ball pruning: no graph can contain it.
    PrunedType,
    /// Pruning removed every type: no graph satisfies the ruleset. A warning,
    /// not an error, because the file itself is fine and the compiled
    /// ruleset still says which types were removed and why.
    Unrealizable,
}

impl IssueCode {
    /// Every code, for the parity test with Python.
    pub const ALL: [IssueCode; 10] = [
        IssueCode::SelfLoop,
        IssueCode::RepeatedEdge,
        IssueCode::NodeGap,
        IssueCode::Disconnected,
        IssueCode::BeyondRadius,
        IssueCode::ShallowType,
        IssueCode::ExactDuplicate,
        IssueCode::IsomorphicDuplicate,
        IssueCode::PrunedType,
        IssueCode::Unrealizable,
    ];

    /// The string value of the matching Python `IssueCode` member.
    pub fn as_str(self) -> &'static str {
        match self {
            IssueCode::SelfLoop => "self_loop",
            IssueCode::RepeatedEdge => "repeated_edge",
            IssueCode::NodeGap => "node_gap",
            IssueCode::Disconnected => "disconnected",
            IssueCode::BeyondRadius => "beyond_radius",
            IssueCode::ShallowType => "shallow_type",
            IssueCode::ExactDuplicate => "exact_duplicate",
            IssueCode::IsomorphicDuplicate => "isomorphic_duplicate",
            IssueCode::PrunedType => "pruned_type",
            IssueCode::Unrealizable => "unrealizable",
        }
    }

    /// Fixed severity per code, matching Python's `SEVERITY` table.
    pub fn severity(self) -> Severity {
        match self {
            IssueCode::SelfLoop
            | IssueCode::RepeatedEdge
            | IssueCode::NodeGap
            | IssueCode::Disconnected
            | IssueCode::BeyondRadius => Severity::Error,
            IssueCode::ShallowType
            | IssueCode::ExactDuplicate
            | IssueCode::IsomorphicDuplicate
            | IssueCode::PrunedType
            | IssueCode::Unrealizable => Severity::Warning,
        }
    }
}

/// One problem found while compiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    pub code: IssueCode,
    /// Human-readable detail, e.g. which edge, as positioned in the file.
    pub message: String,
    /// Position of the offending type in the ruleset as written, or None for
    /// issues about the ruleset as a whole.
    pub type_index: Option<usize>,
}

impl Issue {
    pub fn new(code: IssueCode, message: impl Into<String>, type_index: Option<usize>) -> Self {
        Self {
            code,
            message: message.into(),
            type_index,
        }
    }

    pub fn severity(&self) -> Severity {
        self.code.severity()
    }

    pub fn is_error(&self) -> bool {
        self.severity() == Severity::Error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_lists_every_code_once() {
        let codes: HashSet<IssueCode> = IssueCode::ALL.into_iter().collect();
        let names: HashSet<&str> = IssueCode::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(codes.len(), IssueCode::ALL.len());
        assert_eq!(names.len(), IssueCode::ALL.len());
    }

    #[test]
    fn is_error_follows_severity() {
        for code in IssueCode::ALL {
            let issue = Issue::new(code, "", None);
            assert_eq!(
                issue.is_error(),
                code.severity() == Severity::Error,
                "{code:?}"
            );
        }
    }
}
