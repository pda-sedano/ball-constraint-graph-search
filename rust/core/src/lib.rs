//! Rust core of the ball-constraint graph search.
//!
//! Python reads ruleset files and checks their format. Everything about the
//! ball graphs themselves is checked here, when a ruleset is compiled:
//!
//! - [`ball`]: structural checks on one ball type, and [`CompiledBall`].
//! - [`compile()`]: checks against r and across types, producing a [`CompiledRuleset`].
//! - [`issue`]: issue codes, which mirror Python's `report.IssueCode`.
//!
//! Later steps add canonical form, ball extraction, pruning, the embedding
//! check, the validity checker and the sampler.

pub mod ball;
pub mod compile;
pub mod issue;
pub mod ruleset;

pub use ball::{check_structure, CompiledBall, Edge, Node};
pub use compile::{compile, CompileOutcome};
pub use issue::{Issue, IssueCode, Severity};
pub use ruleset::CompiledRuleset;
