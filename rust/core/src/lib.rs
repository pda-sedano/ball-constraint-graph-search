//! Rust core of the ball-constraint graph search.
//!
//! Python reads ruleset files and checks their format. Everything about the
//! ball graphs themselves is checked here, when a ruleset is compiled:
//!
//! - [`ball`]: structural checks on one ball type, and [`CompiledBall`].
//! - [`canonical`]: canonical keys, equal exactly for isomorphic rooted balls.
//! - [`compile()`]: checks against r and across types, merging duplicates,
//!   producing a [`CompiledRuleset`] that looks types up by canonical key.
//! - [`prune`]: visible sub-ball pruning, run by every compile, which removes
//!   types no graph can contain and says why.
//! - [`issue`]: issue codes, which mirror Python's `report.IssueCode`.
//!
//! And the graph the search works on:
//!
//! - [`graph`]: [`Graph`], fixed-size adjacency rows edited in place.
//! - [`extract`]: the radius-r ball around a node ([`BallExtractor`]), and
//!   [`ball_census()`], every ball shape in a reference graph with counts.
//! - [`embed`]: the embedding check for open and shadow nodes, whose visible
//!   ball must fit inside some kept type ([`EmbeddingChecker`]).
//! - [`check`]: the validity checker ([`Checker`]): classify every node by its
//!   distance to the open set, check each, and report.
//! - [`graph6`] and [`enumeration`]: read geng's output and find the valid
//!   graphs in it, with automorphism counts, for the enumeration baseline.
//!
//! The sampler comes later.

pub mod ball;
pub mod canonical;
pub mod check;
pub mod compile;
pub mod embed;
pub mod enumeration;
pub mod extract;
pub mod graph;
pub mod graph6;
pub mod issue;
pub mod prune;
pub mod ruleset;
#[cfg(test)]
mod testing;

pub use ball::{check_structure, CompiledBall, Edge, Node};
pub use canonical::{automorphism_count, CanonicalForm, CanonicalKey};
pub use check::{CheckReport, Checker, NodeClass, NodeReport, Outcome};
pub use compile::{compile, CompileOutcome};
pub use embed::{EmbedQuery, EmbeddingChecker, STEP_BUDGET};
pub use enumeration::{valid_graphs, Enumeration, EnumerationError, ValidGraph};
pub use extract::{ball_census, BallExtractor, CensusEntry, VisibleBall};
pub use graph::{Graph, GraphError};
pub use graph6::{parse_graph6, Graph6Error};
pub use issue::{Issue, IssueCode, Severity};
pub use prune::{Pruning, Removal};
pub use ruleset::CompiledRuleset;
