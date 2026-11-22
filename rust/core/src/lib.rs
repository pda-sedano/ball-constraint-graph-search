//! Rust core of the ball-constraint graph search.
//!
//! Python reads ruleset files and checks their format. Everything about the
//! ball graphs themselves is checked here, when a ruleset is compiled:
//!
//! - [`ball`]: structural checks on one ball type, and [`CompiledBall`].
//! - [`canonical`]: canonical keys, equal exactly for isomorphic rooted balls.
//! - [`compile()`]: checks against r and across types, merging duplicates,
//!   producing a [`CompiledRuleset`] that looks types up by canonical key.
//! - [`issue`]: issue codes, which mirror Python's `report.IssueCode`.
//!
//! And the graph the search works on:
//!
//! - [`graph`]: [`Graph`], fixed-size adjacency rows edited in place.
//! - [`extract`]: the radius-r ball around a node ([`BallExtractor`]), and
//!   [`ball_census()`], every ball shape in a reference graph with counts.
//!
//! Later steps add pruning, the embedding check, the validity checker and
//! the sampler.

pub mod ball;
pub mod canonical;
pub mod compile;
pub mod extract;
pub mod graph;
pub mod issue;
pub mod ruleset;

pub use ball::{check_structure, CompiledBall, Edge, Node};
pub use canonical::{CanonicalForm, CanonicalKey};
pub use compile::{compile, CompileOutcome};
pub use extract::{ball_census, BallExtractor, CensusEntry, VisibleBall};
pub use graph::{Graph, GraphError};
pub use issue::{Issue, IssueCode, Severity};
pub use ruleset::CompiledRuleset;
