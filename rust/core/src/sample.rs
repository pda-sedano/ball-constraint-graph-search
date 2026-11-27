//! The sampler: one Metropolis-Hastings chain over numbered graphs on n
//! nodes, each with an open set of exactly m nodes (m = 0 is closed mode),
//! at a fixed β, with a full recheck after every move.
//!
//! # The chain
//!
//! Each step proposes one change. In open mode (0 < m < n), with probability
//! `swap_rate` it swaps the flags of a uniformly random open node and a
//! uniformly random non-open node. Otherwise it proposes an edge change: with
//! probability 1/2 it removes a uniformly random edge, else it adds a
//! uniformly random non-edge. An addition that would push a node past the
//! degree cap is rejected at once, which is the same as giving those graphs
//! infinite energy. A proposed move is accepted with probability
//!
//! ```text
//! remove: min(1, exp(-β ΔU) · E / (N - E + 1))
//! add:    min(1, exp(-β ΔU) · (N - E) / (E + 1))
//! swap:   min(1, exp(-β ΔU))
//! ```
//!
//! where E is the current edge count and N = n(n-1)/2 the number of node
//! pairs. The edge factors correct for that proposal not being symmetric; the
//! swap is symmetric already (m(n - m) choices either way). The choice
//! between kinds of move does not depend on the state, so the chain's
//! stationary distribution gives every numbered graph with an open set of m
//! nodes, under the cap, weight exp(-βU). With an energy that is 0 exactly on
//! valid states, every valid numbered state gets the same weight at any β.
//!
//! # Numbered, not unlabelled
//!
//! The chain is uniform over *numbered* states. An unlabelled state with
//! automorphism group Aut (symmetries keeping the open set in place) has
//! n!/|Aut| distinct numberings, so the chain gives it time in proportion to
//! 1/|Aut|: symmetric graphs get less. The numbering is arbitrary, so the
//! analysis (Python's `sampler` module) counts each unlabelled state once by
//! default, weighting every sample by its |Aut|, which every snapshot and
//! visit count carries. That reweighting is exact. Building |Aut| into the
//! chain's target instead would need a nauty call per proposal.
//!
//! In closed mode no swap is ever drawn, so a seed gives the same run as
//! before open mode existed.
//!
//! # Recording
//!
//! [`Recorder`] runs a sampler and keeps what the design doc asks for. Two
//! rules keep it unbiased. The state after every step counts, including steps
//! whose move was rejected, since the chain's time at a state is its weight.
//! Snapshots are taken at fixed step numbers (every `snapshot_every` steps,
//! kept if U = 0 then), never "at the first valid state after a gap", which
//! would favour valid states that are easy to walk into.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ball::{Edge, Node};
use crate::canonical::{canonical_open_graph, open_automorphism_count, CanonicalGraph};
use crate::check::Checker;
use crate::graph::{Graph, GraphError};
use crate::rng::Rng;
use crate::ruleset::CompiledRuleset;

/// What the chain minimizes, given the graph and which nodes are open (one
/// entry per node). Any energy that is 0 exactly on the states to be sampled
/// gives the same uniform result, so it can be tuned for speed.
pub trait Energy {
    fn energy(&mut self, graph: &Graph, is_open: &[bool]) -> f64;
}

/// The design doc's energy: the number of nodes failing their class's check
/// (core nodes need a kept type, open and shadow nodes an embedding into
/// one), plus (components - 1).
pub struct RulesetEnergy<'a> {
    checker: Checker<'a>,
    open: Vec<Node>,
    passing: Vec<bool>,
}

impl<'a> RulesetEnergy<'a> {
    pub fn new(ruleset: &'a CompiledRuleset) -> Self {
        RulesetEnergy {
            checker: Checker::new(ruleset),
            open: Vec::new(),
            passing: Vec::new(),
        }
    }

    /// Whether each node passed at the last [`Energy::energy`] call. Kept so
    /// an incremental recheck can later update only the nodes near a move.
    pub fn passing(&self) -> &[bool] {
        &self.passing
    }
}

impl Energy for RulesetEnergy<'_> {
    fn energy(&mut self, graph: &Graph, is_open: &[bool]) -> f64 {
        let n = graph.num_nodes();
        self.open.clear();
        self.open
            .extend((0..n).filter(|&v| is_open[v]).map(|v| v as Node));
        // Distance to the nearest open node classifies each node; with no
        // open nodes every distance is infinite and every node is core.
        let distance = graph.distances_from(&self.open);
        self.passing.clear();
        for v in 0..n as Node {
            let report = self.checker.check_node(graph, is_open, &distance, v);
            self.passing.push(report.outcome.passes());
        }
        let failing = self.passing.iter().filter(|&&p| !p).count();
        (failing + graph.component_count().saturating_sub(1)) as f64
    }
}

/// Energy 0 everywhere: the chain is then uniform over every numbered graph
/// under the degree cap, with every open set of size m. For testing the
/// chain apart from the checker.
pub struct ZeroEnergy;

impl Energy for ZeroEnergy {
    fn energy(&mut self, _graph: &Graph, _is_open: &[bool]) -> f64 {
        0.0
    }
}

/// How often an open-mode step proposes a flag swap rather than an edge change.
pub const DEFAULT_SWAP_RATE: f64 = 0.2;

/// A chain's fixed settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub num_nodes: usize,
    /// No node's degree may exceed this.
    pub max_degree: usize,
    pub beta: f64,
    pub seed: u64,
    /// m: the open set always has exactly this many nodes. 0 is closed mode.
    pub open_nodes: usize,
    /// Probability that a step proposes a flag swap, when 0 < m < n.
    pub swap_rate: f64,
}

impl Settings {
    /// Closed mode.
    pub fn closed(num_nodes: usize, max_degree: usize, beta: f64, seed: u64) -> Self {
        Settings {
            num_nodes,
            max_degree,
            beta,
            seed,
            open_nodes: 0,
            swap_rate: DEFAULT_SWAP_RATE,
        }
    }

    /// Open mode with m open nodes and the default swap rate.
    pub fn open(num_nodes: usize, max_degree: usize, beta: f64, seed: u64, m: usize) -> Self {
        Settings {
            open_nodes: m,
            ..Settings::closed(num_nodes, max_degree, beta, seed)
        }
    }
}

/// The chain's first graph.
#[derive(Debug, Clone)]
pub enum Start {
    Empty,
    /// Random distinct edges under the cap, up to this many. Fewer if the
    /// cap makes that many hard to place.
    Random {
        edges: usize,
    },
    Edges(Vec<Edge>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplerError {
    /// β must be finite and not negative.
    Beta(f64),
    /// The swap rate must be in [0, 1].
    SwapRate(f64),
    /// More open nodes than nodes.
    OpenCount { open: usize, nodes: usize },
    /// A starting edge could not be added.
    Start(GraphError),
    /// The starting open set is the wrong size or names a bad node.
    StartOpen(String),
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerError::Beta(beta) => write!(f, "beta must be finite and >= 0, got {beta}"),
            SamplerError::SwapRate(rate) => write!(f, "swap_rate must be in [0, 1], got {rate}"),
            SamplerError::OpenCount { open, nodes } => {
                write!(f, "{open} open nodes is more than the {nodes} nodes")
            }
            SamplerError::Start(e) => write!(f, "bad starting graph: {e}"),
            SamplerError::StartOpen(why) => write!(f, "bad starting open set: {why}"),
        }
    }
}

impl std::error::Error for SamplerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Remove,
    Add,
    /// Exchange the flags of one open and one non-open node.
    Swap,
}

/// What one step did. Every variant except `Accepted` leaves the state as
/// it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Accepted(Move),
    Rejected(Move),
    /// An addition that would exceed the degree cap.
    Capped,
    /// Nothing of the chosen kind to propose: no edges to remove, or no
    /// non-edges to add.
    NoCandidate(Move),
}

/// Counts of what the steps did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoveStats {
    pub proposed_remove: u64,
    pub accepted_remove: u64,
    pub proposed_add: u64,
    pub accepted_add: u64,
    pub proposed_swap: u64,
    pub accepted_swap: u64,
    /// Additions rejected by the degree cap (counted in `proposed_add`).
    pub capped: u64,
    /// Steps with nothing to propose (counted in the proposals).
    pub no_candidate: u64,
}

/// The Metropolis-Hastings acceptance probability for a move from a graph
/// with `edges` edges, out of `pairs` node pairs, changing the energy by
/// `delta`. A swap ignores `edges` and `pairs`.
pub fn acceptance(mv: Move, edges: usize, pairs: usize, delta: f64, beta: f64) -> f64 {
    let proposal_ratio = match mv {
        Move::Remove => edges as f64 / (pairs - edges + 1) as f64,
        Move::Add => (pairs - edges) as f64 / (edges + 1) as f64,
        Move::Swap => 1.0,
    };
    ((-beta * delta).exp() * proposal_ratio).min(1.0)
}

/// Gives every sampler a distinct id, so a recorder that sees several
/// samplers (one per temperature, in tempering) never mixes up their states.
static NEXT_SAMPLER_ID: AtomicU64 = AtomicU64::new(0);

/// One chain.
pub struct Sampler<E> {
    /// Unique in this process; see [`Sampler::id`].
    id: u64,
    energy_fn: E,
    graph: Graph,
    /// Every edge once, in no order, so a uniformly random one is one draw.
    edges: Vec<Edge>,
    /// Per node, and the open and non-open nodes as lists in no order, so a
    /// uniformly random one of each is one draw.
    is_open: Vec<bool>,
    open: Vec<Node>,
    closed: Vec<Node>,
    energy: f64,
    beta: f64,
    /// 0 unless swaps are possible (0 < m < n).
    swap_rate: f64,
    rng: Rng,
    pairs: usize,
    steps: u64,
    stats: MoveStats,
}

impl<E: Energy> Sampler<E> {
    /// A chain with these settings. `start_open` gives the first open set
    /// (exactly `settings.open_nodes` distinct nodes); if None it is a
    /// uniformly random one. Random starts draw from the seeded generator,
    /// edges first.
    pub fn new(
        mut energy_fn: E,
        settings: Settings,
        start: Start,
        start_open: Option<Vec<Node>>,
    ) -> Result<Self, SamplerError> {
        let Settings {
            num_nodes: n,
            max_degree,
            beta,
            seed,
            open_nodes: m,
            swap_rate,
        } = settings;
        if !(beta.is_finite() && beta >= 0.0) {
            return Err(SamplerError::Beta(beta));
        }
        if !(0.0..=1.0).contains(&swap_rate) {
            return Err(SamplerError::SwapRate(swap_rate));
        }
        if m > n {
            return Err(SamplerError::OpenCount { open: m, nodes: n });
        }
        let mut rng = Rng::new(seed);
        let pairs = n * n.saturating_sub(1) / 2;
        let graph = match start {
            Start::Empty => Graph::new(n, max_degree),
            Start::Edges(edges) => {
                Graph::from_edges(n, max_degree, &edges).map_err(SamplerError::Start)?
            }
            Start::Random { edges } => {
                let mut graph = Graph::new(n, max_degree);
                let target = edges.min(pairs);
                let mut attempts = 0;
                while graph.num_edges() < target && attempts < 100 * target {
                    attempts += 1;
                    let (u, v) = random_pair(&mut rng, n);
                    let _ = graph.add_edge(u, v); // skips repeats and capped pairs
                }
                graph
            }
        };

        let mut is_open = vec![false; n];
        match start_open {
            Some(nodes) => {
                if nodes.len() != m {
                    return Err(SamplerError::StartOpen(format!(
                        "{} nodes given for {m} open",
                        nodes.len()
                    )));
                }
                for v in nodes {
                    let i = v as usize;
                    if i >= n {
                        return Err(SamplerError::StartOpen(format!("node {v} of {n}")));
                    }
                    if is_open[i] {
                        return Err(SamplerError::StartOpen(format!("node {v} given twice")));
                    }
                    is_open[i] = true;
                }
            }
            None => {
                // A uniform m-subset: the first m places of a partial shuffle.
                // Draws nothing when m = 0, so closed runs are unchanged.
                let mut nodes: Vec<usize> = (0..n).collect();
                for i in 0..m {
                    let j = i + rng.below(n - i);
                    nodes.swap(i, j);
                    is_open[nodes[i]] = true;
                }
            }
        }
        let open: Vec<Node> = (0..n).filter(|&v| is_open[v]).map(|v| v as Node).collect();
        let closed: Vec<Node> = (0..n).filter(|&v| !is_open[v]).map(|v| v as Node).collect();

        let edges = graph.edges();
        let energy = energy_fn.energy(&graph, &is_open);
        Ok(Sampler {
            id: NEXT_SAMPLER_ID.fetch_add(1, Ordering::Relaxed),
            energy_fn,
            graph,
            edges,
            is_open,
            open,
            closed,
            energy,
            beta,
            swap_rate: if m > 0 && m < n { swap_rate } else { 0.0 },
            rng,
            pairs,
            steps: 0,
            stats: MoveStats::default(),
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Whether each node is open.
    pub fn is_open(&self) -> &[bool] {
        &self.is_open
    }

    /// The open nodes, sorted.
    pub fn open_nodes(&self) -> Vec<Node> {
        let mut open = self.open.clone();
        open.sort_unstable();
        open
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Change β, as tempering does when two replicas trade temperatures.
    /// The energy does not depend on β, so nothing is recomputed.
    ///
    /// # Panics
    ///
    /// If `beta` is not finite and at least 0.
    pub fn set_beta(&mut self, beta: f64) {
        assert!(
            beta.is_finite() && beta >= 0.0,
            "beta must be finite and >= 0, got {beta}"
        );
        self.beta = beta;
    }

    /// An id no other sampler in this process has. With [`Sampler::version`]
    /// it identifies a state, which the recorder uses to skip recomputing a
    /// canonical form.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn stats(&self) -> MoveStats {
        self.stats
    }

    /// The number of accepted moves so far: changes exactly when the state does.
    pub fn version(&self) -> u64 {
        self.stats.accepted_add + self.stats.accepted_remove + self.stats.accepted_swap
    }

    pub fn energy_fn(&self) -> &E {
        &self.energy_fn
    }

    /// One proposal, accepted or not.
    pub fn step(&mut self) -> StepResult {
        self.steps += 1;
        // Draw for a swap only when one is possible, so closed mode uses the
        // generator exactly as it did before open mode.
        if self.swap_rate > 0.0 && self.rng.next_f64() < self.swap_rate {
            return self.swap();
        }
        let edges = self.edges.len();
        if self.rng.next_f64() < 0.5 {
            self.stats.proposed_remove += 1;
            if edges == 0 {
                self.stats.no_candidate += 1;
                return StepResult::NoCandidate(Move::Remove);
            }
            let i = self.rng.below(edges);
            let (u, v) = self.edges[i];
            self.graph.remove_edge(u, v);
            let new = self.energy_fn.energy(&self.graph, &self.is_open);
            let p = acceptance(
                Move::Remove,
                edges,
                self.pairs,
                new - self.energy,
                self.beta,
            );
            if self.rng.next_f64() < p {
                self.edges.swap_remove(i);
                self.energy = new;
                self.stats.accepted_remove += 1;
                StepResult::Accepted(Move::Remove)
            } else {
                self.graph
                    .add_edge(u, v)
                    .expect("an edge just removed can be put back");
                StepResult::Rejected(Move::Remove)
            }
        } else {
            self.stats.proposed_add += 1;
            if edges == self.pairs {
                self.stats.no_candidate += 1;
                return StepResult::NoCandidate(Move::Add);
            }
            // A uniform non-edge: uniform pairs until one is absent.
            let (u, v) = loop {
                let (u, v) = random_pair(&mut self.rng, self.graph.num_nodes());
                if !self.graph.has_edge(u, v) {
                    break (u, v);
                }
            };
            let cap = self.graph.max_degree();
            if self.graph.degree(u) == cap || self.graph.degree(v) == cap {
                self.stats.capped += 1;
                return StepResult::Capped;
            }
            self.graph
                .add_edge(u, v)
                .expect("a non-edge under the cap can be added");
            let new = self.energy_fn.energy(&self.graph, &self.is_open);
            let p = acceptance(Move::Add, edges, self.pairs, new - self.energy, self.beta);
            if self.rng.next_f64() < p {
                self.edges.push((u, v));
                self.energy = new;
                self.stats.accepted_add += 1;
                StepResult::Accepted(Move::Add)
            } else {
                self.graph.remove_edge(u, v);
                StepResult::Rejected(Move::Add)
            }
        }
    }

    /// Propose exchanging the flags of a uniformly random open node and a
    /// uniformly random non-open node.
    fn swap(&mut self) -> StepResult {
        self.stats.proposed_swap += 1;
        let i = self.rng.below(self.open.len());
        let j = self.rng.below(self.closed.len());
        let (a, b) = (self.open[i], self.closed[j]);
        self.is_open[a as usize] = false;
        self.is_open[b as usize] = true;
        let new = self.energy_fn.energy(&self.graph, &self.is_open);
        let p = acceptance(Move::Swap, 0, 0, new - self.energy, self.beta);
        if self.rng.next_f64() < p {
            self.open[i] = b;
            self.closed[j] = a;
            self.energy = new;
            self.stats.accepted_swap += 1;
            StepResult::Accepted(Move::Swap)
        } else {
            self.is_open[a as usize] = true;
            self.is_open[b as usize] = false;
            StepResult::Rejected(Move::Swap)
        }
    }
}

/// A uniformly random unordered pair of distinct nodes, as `(min, max)`.
fn random_pair(rng: &mut Rng, n: usize) -> (Node, Node) {
    let u = rng.below(n);
    let mut v = rng.below(n - 1);
    if v >= u {
        v += 1;
    }
    (u.min(v) as Node, u.max(v) as Node)
}

/// What [`Recorder`] keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordOptions {
    /// Steps run before anything is recorded.
    pub burn_in: u64,
    /// Steps recorded after the burn-in.
    pub steps: u64,
    /// Snapshot the state at every multiple of this many recorded steps, if
    /// it is valid then.
    pub snapshot_every: Option<u64>,
    /// Count the steps spent on each valid state up to isomorphism, keyed by
    /// [`CanonicalGraph::key`]. Only useful at small n, where states recur.
    pub count_graphs: bool,
    /// Record the energy at every multiple of this many recorded steps.
    pub trace_every: Option<u64>,
}

/// A valid state seen at a snapshot step.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// The chain's step number, burn-in included.
    pub step: u64,
    pub edges: Vec<Edge>,
    /// The open nodes, sorted; empty in closed mode.
    pub open: Vec<Node>,
    /// Symmetries that keep the open set in place.
    pub automorphisms: f64,
}

/// Time spent on one valid state up to isomorphism.
#[derive(Debug, Clone, PartialEq)]
pub struct Visits {
    pub steps: u64,
    /// The canonical graph6 and canonical open nodes, as in [`CanonicalGraph`].
    pub graph6: String,
    pub open: Vec<Node>,
    pub automorphisms: f64,
}

/// Runs a sampler for a burn-in and then a number of recorded steps, in as
/// many calls to [`Recorder::advance`] as the caller likes (so Python can
/// check for Ctrl-C between them).
#[derive(Debug, Clone)]
pub struct Recorder {
    options: RecordOptions,
    done: u64,
    /// Recorded steps at which the state was valid.
    pub valid_steps: u64,
    /// By [`CanonicalGraph::key`], when `count_graphs` is on.
    pub visits: HashMap<String, Visits>,
    pub snapshots: Vec<Snapshot>,
    /// `(step, energy)` pairs, when `trace_every` is set.
    pub trace: Vec<(u64, f64)>,
    /// What the recorded steps did, burn-in included.
    pub moves: MoveStats,
    /// The sampler id and version and the canonical form of the last valid
    /// state keyed, so a state that has not changed is not canonicalized
    /// again. The id matters in tempering, where one recorder sees several
    /// samplers.
    cached: Option<(u64, u64, String, CanonicalGraph)>,
}

impl Recorder {
    pub fn new(options: RecordOptions) -> Self {
        Recorder {
            options,
            done: 0,
            valid_steps: 0,
            visits: HashMap::new(),
            snapshots: Vec::new(),
            trace: Vec::new(),
            moves: MoveStats::default(),
            cached: None,
        }
    }

    pub fn options(&self) -> RecordOptions {
        self.options
    }

    /// Steps still to run, burn-in included.
    pub fn remaining(&self) -> u64 {
        self.options.burn_in + self.options.steps - self.done
    }

    /// Run up to `max_steps` more steps. Returns how many it ran.
    pub fn advance<E: Energy>(&mut self, sampler: &mut Sampler<E>, max_steps: u64) -> u64 {
        let count = max_steps.min(self.remaining());
        for _ in 0..count {
            let result = sampler.step();
            self.record(sampler, result);
        }
        count
    }

    /// Record one step that `sampler` just took, with its result. Use this
    /// when something else drives the sampler, as tempering does (one
    /// recorder per temperature, fed by whichever replica holds it).
    pub fn record<E: Energy>(&mut self, sampler: &Sampler<E>, result: StepResult) {
        self.count_move(result);
        self.done += 1;
        if self.done <= self.options.burn_in {
            return;
        }
        let t = self.done - self.options.burn_in;
        if self.options.trace_every.is_some_and(|k| t % k == 0) {
            self.trace.push((sampler.steps(), sampler.energy()));
        }
        if sampler.energy() != 0.0 {
            return;
        }
        self.valid_steps += 1;
        if self.options.count_graphs {
            self.refresh(sampler);
            let (_, _, key, form) = self.cached.as_ref().expect("just refreshed");
            self.visits
                .entry(key.clone())
                .or_insert_with(|| Visits {
                    steps: 0,
                    graph6: form.graph6(),
                    open: form.open.clone(),
                    automorphisms: form.automorphisms,
                })
                .steps += 1;
        }
        if self.options.snapshot_every.is_some_and(|k| t % k == 0) {
            let automorphisms = if self.options.count_graphs {
                self.cached
                    .as_ref()
                    .expect("refreshed above")
                    .3
                    .automorphisms
            } else {
                open_automorphism_count(sampler.graph(), sampler.is_open())
            };
            self.snapshots.push(Snapshot {
                step: sampler.steps(),
                edges: sampler.graph().edges(),
                open: sampler.open_nodes(),
                automorphisms,
            });
        }
    }

    fn count_move(&mut self, result: StepResult) {
        let m = &mut self.moves;
        let (kind, accepted) = match result {
            StepResult::Accepted(kind) => (kind, true),
            StepResult::Rejected(kind) => (kind, false),
            StepResult::Capped => {
                m.capped += 1;
                (Move::Add, false)
            }
            StepResult::NoCandidate(kind) => {
                m.no_candidate += 1;
                (kind, false)
            }
        };
        let (proposed, taken) = match kind {
            Move::Remove => (&mut m.proposed_remove, &mut m.accepted_remove),
            Move::Add => (&mut m.proposed_add, &mut m.accepted_add),
            Move::Swap => (&mut m.proposed_swap, &mut m.accepted_swap),
        };
        *proposed += 1;
        if accepted {
            *taken += 1;
        }
    }

    /// Make the cached canonical form match the sampler's current state.
    fn refresh<E: Energy>(&mut self, sampler: &Sampler<E>) {
        if self
            .cached
            .as_ref()
            .is_some_and(|(id, version, _, _)| *id == sampler.id() && *version == sampler.version())
        {
            return;
        }
        let form = canonical_open_graph(sampler.graph(), sampler.is_open());
        self.cached = Some((sampler.id(), sampler.version(), form.key(), form));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::canonical_graph;
    use crate::testing::{random_graph, ruleset, PATH_BALL};

    /// Index of each node pair of `0..n` in a bitmask, so a graph is a number.
    fn pair_index(n: usize) -> HashMap<Edge, usize> {
        let mut index = HashMap::new();
        for u in 0..n as Node {
            for v in u + 1..n as Node {
                let i = index.len();
                index.insert((u, v), i);
            }
        }
        index
    }

    fn pairs_of(index: &HashMap<Edge, usize>) -> Vec<Edge> {
        let mut pairs: Vec<Edge> = vec![(0, 0); index.len()];
        for (&e, &i) in index {
            pairs[i] = e;
        }
        pairs
    }

    /// The graph as a bitmask over `pairs`, read straight from the adjacency
    /// rows (no allocation: the stationary tests call this millions of times).
    fn mask_of(graph: &Graph, pairs: &[Edge]) -> usize {
        pairs
            .iter()
            .enumerate()
            .filter(|(_, &(u, v))| graph.has_edge(u, v))
            .map(|(i, _)| 1 << i)
            .sum()
    }

    fn open_mask(is_open: &[bool]) -> usize {
        is_open
            .iter()
            .enumerate()
            .filter(|(_, &o)| o)
            .map(|(v, _)| 1 << v)
            .sum()
    }

    /// States on 4 nodes are (edge mask, open mask): index mask * 16 + open.
    const N: usize = 4;
    const OPEN_MASKS: usize = 1 << N;

    /// An arbitrary fixed energy for every state on 4 nodes.
    struct TableEnergy {
        pairs: Vec<Edge>,
        table: Vec<f64>,
    }

    impl Energy for TableEnergy {
        fn energy(&mut self, graph: &Graph, is_open: &[bool]) -> f64 {
            self.table[mask_of(graph, &self.pairs) * OPEN_MASKS + open_mask(is_open)]
        }
    }

    fn random_table(seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..(1usize << (N * (N - 1) / 2)) * OPEN_MASKS)
            .map(|_| rng.below(4) as f64)
            .collect()
    }

    fn degrees_ok(mask: usize, pairs: &[Edge], cap: usize) -> bool {
        let mut degree = [0; N];
        for (i, &(u, v)) in pairs.iter().enumerate() {
            if mask >> i & 1 == 1 {
                degree[u as usize] += 1;
                degree[v as usize] += 1;
            }
        }
        degree.iter().all(|&d| d <= cap)
    }

    #[test]
    fn transition_rule_satisfies_detailed_balance_exactly() {
        //! Every pair of states on 4 nodes one move apart, for m = 0, 1 and 2
        //! open nodes, two caps and arbitrary energies: π(x) P(x→y) =
        //! π(y) P(y→x). P is built from the proposal (a swap with probability
        //! s when 0 < m < n, else ½ remove a uniform edge or ½ add a uniform
        //! non-edge, capped additions rejected) and `acceptance`.
        let index = pair_index(N);
        let pairs = pairs_of(&index);
        let total = pairs.len();
        for m in [0usize, 1, 2] {
            let s = if m > 0 { 0.3 } else { 0.0 };
            for cap in [2, 3] {
                for (seed, beta) in [(1, 0.7), (2, 2.0), (3, 0.0)] {
                    let energy = random_table(seed + 10 * m as u64);
                    let u = |x: usize, o: usize| energy[x * OPEN_MASKS + o];
                    let weight = |x: usize, o: usize| (-beta * u(x, o)).exp();
                    let edge_move = |x: usize, o: usize, i: usize| -> f64 {
                        let e = x.count_ones() as usize;
                        let y = x ^ (1 << i);
                        let delta = u(y, o) - u(x, o);
                        (1.0 - s)
                            * if x >> i & 1 == 1 {
                                0.5 / e as f64 * acceptance(Move::Remove, e, total, delta, beta)
                            } else if degrees_ok(y, &pairs, cap) {
                                0.5 / (total - e) as f64
                                    * acceptance(Move::Add, e, total, delta, beta)
                            } else {
                                0.0
                            }
                    };
                    let swap_move = |x: usize, o: usize, o2: usize| -> f64 {
                        let delta = u(x, o2) - u(x, o);
                        s / (m * (N - m)) as f64 * acceptance(Move::Swap, 0, 0, delta, beta)
                    };
                    let opens = (0..OPEN_MASKS).filter(|o: &usize| o.count_ones() as usize == m);
                    for o in opens {
                        for x in (0..1 << total).filter(|&x| degrees_ok(x, &pairs, cap)) {
                            for i in 0..total {
                                let y = x ^ (1 << i);
                                if !degrees_ok(y, &pairs, cap) {
                                    assert_eq!(edge_move(x, o, i), 0.0);
                                    continue;
                                }
                                let forward = weight(x, o) * edge_move(x, o, i);
                                let backward = weight(y, o) * edge_move(y, o, i);
                                assert!(
                                    (forward - backward).abs() <= 1e-12 * forward.max(backward),
                                    "m {m}, cap {cap}: {x:06b} <-> {y:06b}"
                                );
                            }
                            for a in (0..N).filter(|a| o >> a & 1 == 1) {
                                for b in (0..N).filter(|b| o >> b & 1 == 0) {
                                    let o2 = o ^ (1 << a) ^ (1 << b);
                                    let forward = weight(x, o) * swap_move(x, o, o2);
                                    let backward = weight(x, o2) * swap_move(x, o2, o);
                                    assert!(
                                        (forward - backward).abs() <= 1e-12 * forward.max(backward),
                                        "m {m}: swap {o:04b} <-> {o2:04b}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run the real chain on 4 nodes and compare its time at each state with
    /// the exact stationary distribution exp(-βU) under the cap, over open
    /// sets of size m.
    fn check_stationary(table: Vec<f64>, cap: usize, beta: f64, seed: u64, m: usize, steps: u64) {
        let index = pair_index(N);
        let pairs = pairs_of(&index);
        let allowed: Vec<usize> = (0..1 << pairs.len())
            .filter(|&x| degrees_ok(x, &pairs, cap))
            .flat_map(|x| {
                (0..OPEN_MASKS)
                    .filter(move |o: &usize| o.count_ones() as usize == m)
                    .map(move |o| x * OPEN_MASKS + o)
            })
            .collect();
        let z: f64 = allowed.iter().map(|&k| (-beta * table[k]).exp()).sum();
        let settings = Settings {
            swap_rate: 0.3,
            ..Settings::open(N, cap, beta, seed, m)
        };
        let energy = TableEnergy {
            pairs: pairs.clone(),
            table: table.clone(),
        };
        let mut sampler = Sampler::new(energy, settings, Start::Empty, None).expect("valid");
        let mut time = vec![0u64; table.len()];
        for _ in 0..steps {
            sampler.step();
            let k = mask_of(sampler.graph(), &pairs) * OPEN_MASKS + open_mask(sampler.is_open());
            time[k] += 1;
        }
        let distance: f64 = allowed
            .iter()
            .map(|&k| (time[k] as f64 / steps as f64 - (-beta * table[k]).exp() / z).abs())
            .sum::<f64>()
            / 2.0;
        let limit = if m == 0 { 0.01 } else { 0.015 };
        assert!(distance < limit, "m {m}: total variation {distance}");
        let outside: u64 = (0..time.len())
            .filter(|k| !allowed.contains(k))
            .map(|k| time[k])
            .sum();
        assert_eq!(outside, 0, "the chain left the allowed states");
    }

    #[test]
    fn zero_energy_chain_is_uniform_under_the_cap() {
        let zero = || vec![0.0; 64 * OPEN_MASKS];
        check_stationary(zero(), 3, 1.0, 11, 0, 2_000_000);
        check_stationary(zero(), 2, 1.0, 12, 0, 2_000_000);
        check_stationary(zero(), 2, 1.0, 15, 2, 3_000_000);
    }

    #[test]
    fn chain_matches_exp_minus_beta_u() {
        check_stationary(random_table(5), 3, 0.8, 13, 0, 2_000_000);
        check_stationary(random_table(6), 2, 1.5, 14, 0, 2_000_000);
        check_stationary(random_table(7), 3, 0.8, 16, 1, 3_000_000);
    }

    #[test]
    fn ruleset_energy_matches_the_checker() {
        //! With and without open sets.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut energy = RulesetEnergy::new(&rules);
        let mut rng = crate::testing::Rng(4);
        for round in 0..200 {
            let n = 3 + rng.below(12);
            let m = rng.below(2 * n).min(n * (n - 1) / 2);
            let g = random_graph(n, m, &mut rng);
            let open: Vec<Node> = if round % 2 == 0 {
                Vec::new()
            } else {
                (0..n as Node).filter(|_| rng.below(4) == 0).collect()
            };
            let mut is_open = vec![false; n];
            for &v in &open {
                is_open[v as usize] = true;
            }
            let report = Checker::new(&rules).check(&g, &open, None);
            assert_eq!(energy.energy(&g, &is_open), report.energy() as f64);
            let passing: Vec<bool> = report.nodes.iter().map(|r| r.outcome.passes()).collect();
            assert_eq!(energy.passing(), passing);
        }
    }

    #[test]
    fn finds_the_cycle_and_counts_it() {
        //! The path ball at r = 2 allows only cycles of 6 or more nodes.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut sampler = Sampler::new(
            RulesetEnergy::new(&rules),
            Settings::closed(8, 2, 1.0, 3),
            Start::Random { edges: 8 },
            None,
        )
        .expect("valid");
        let mut recorder = Recorder::new(RecordOptions {
            burn_in: 2_000,
            steps: 20_000,
            snapshot_every: Some(100),
            count_graphs: true,
            trace_every: Some(1_000),
        });
        while recorder.remaining() > 0 {
            recorder.advance(&mut sampler, 3_000);
        }
        assert!(recorder.valid_steps > 0, "never valid");
        let cycle: Vec<Edge> = (0..8).map(|v| (v, (v + 1) % 8)).collect();
        let key = canonical_graph(&Graph::from_edges(8, 2, &cycle).expect("valid")).key();
        let keys: Vec<&String> = recorder.visits.keys().collect();
        assert_eq!(keys, [&key]);
        let visit = &recorder.visits[&key];
        assert_eq!(visit.steps, recorder.valid_steps);
        assert_eq!(visit.automorphisms, 16.0);
        assert_eq!(visit.graph6, key);
        assert!(visit.open.is_empty());
        assert_eq!(recorder.trace.len(), 20);
        assert_eq!(recorder.trace[0].0, 3_000);
        for s in &recorder.snapshots {
            assert_eq!(s.step % 100, 0);
            assert_eq!(s.automorphisms, 16.0);
            assert_eq!(s.edges.len(), 8);
            assert!(s.open.is_empty());
        }
        assert_eq!(sampler.steps(), 22_000);
    }

    /// The valid states of the path ball (r = 2) on 6 nodes with 2 open
    /// nodes: the 6-cycle with an open pair at distance 1, 2 or 3, and the
    /// 6-path with both ends open. Keys with their |Aut|.
    fn path_ball_open_states() -> HashMap<String, f64> {
        let cycle: Vec<Edge> = (0..6).map(|v| (v, (v + 1) % 6)).collect();
        let path: Vec<Edge> = (0..5).map(|v| (v, v + 1)).collect();
        let open = |nodes: &[usize]| -> Vec<bool> { (0..6).map(|v| nodes.contains(&v)).collect() };
        let c6 = Graph::from_edges(6, 2, &cycle).expect("valid");
        let p6 = Graph::from_edges(6, 2, &path).expect("valid");
        [
            (&c6, open(&[0, 1])),
            (&c6, open(&[0, 2])),
            (&c6, open(&[0, 3])),
            (&p6, open(&[0, 5])),
        ]
        .into_iter()
        .map(|(g, o)| {
            let form = canonical_open_graph(g, &o);
            (form.key(), form.automorphisms)
        })
        .collect()
    }

    #[test]
    fn open_mode_visits_only_valid_open_states_in_proportion() {
        //! Path ball, n = 6, m = 2: four valid states up to isomorphism, with
        //! |Aut| 2, 2, 4 and 2, so expected shares 2/7, 2/7, 1/7 and 2/7.
        let rules = ruleset(2, &[PATH_BALL]);
        let expected = path_ball_open_states();
        assert_eq!(expected.len(), 4);
        let mut sampler = Sampler::new(
            RulesetEnergy::new(&rules),
            Settings::open(6, 2, 1.0, 5, 2),
            Start::Random { edges: 6 },
            None,
        )
        .expect("valid");
        let mut recorder = Recorder::new(RecordOptions {
            burn_in: 5_000,
            steps: 200_000,
            snapshot_every: Some(1_000),
            count_graphs: true,
            trace_every: None,
        });
        while recorder.remaining() > 0 {
            recorder.advance(&mut sampler, 50_000);
        }
        assert!(sampler.stats().accepted_swap > 0);
        for key in recorder.visits.keys() {
            assert!(expected.contains_key(key), "visited an invalid state {key}");
        }
        let weight_total: f64 = expected.values().map(|aut| 1.0 / aut).sum();
        let distance: f64 = expected
            .iter()
            .map(|(key, aut)| {
                let observed = recorder.visits.get(key).map_or(0, |v| v.steps) as f64
                    / recorder.valid_steps as f64;
                (observed - 1.0 / aut / weight_total).abs()
            })
            .sum::<f64>()
            / 2.0;
        assert!(distance < 0.05, "total variation {distance}");
        for s in &recorder.snapshots {
            assert_eq!(s.open.len(), 2);
            let mut is_open = vec![false; 6];
            for &v in &s.open {
                is_open[v as usize] = true;
            }
            let g = Graph::from_edges(6, 2, &s.edges).expect("valid");
            assert!(Checker::new(&rules).passes(&g, &s.open, Some(2)));
            assert_eq!(s.automorphisms, open_automorphism_count(&g, &is_open));
        }
    }

    #[test]
    fn recorder_counts_every_step_including_rejections() {
        //! Replays the same seeded chain by hand: the recorder must count
        //! every recorded step at U = 0, not only those right after an
        //! accepted move, and snapshot exactly the valid multiples. Run in
        //! open mode, so swaps are replayed too.
        let rules = ruleset(2, &[PATH_BALL]);
        let new_sampler = || {
            Sampler::new(
                RulesetEnergy::new(&rules),
                Settings::open(7, 2, 0.7, 21, 2),
                Start::Random { edges: 7 },
                None,
            )
            .expect("valid")
        };
        let options = RecordOptions {
            burn_in: 500,
            steps: 20_000,
            snapshot_every: Some(7),
            count_graphs: true,
            trace_every: None,
        };
        let mut recorded = new_sampler();
        let mut recorder = Recorder::new(options);
        while recorder.remaining() > 0 {
            recorder.advance(&mut recorded, 999);
        }

        let mut replay = new_sampler();
        let (mut valid, mut after_rejection, mut snapshot_steps) = (0, 0, Vec::new());
        for t in 1..=options.burn_in + options.steps {
            let result = replay.step();
            if t > options.burn_in && replay.energy() == 0.0 {
                valid += 1;
                if !matches!(result, StepResult::Accepted(_)) {
                    after_rejection += 1;
                }
                if (t - options.burn_in) % 7 == 0 {
                    snapshot_steps.push(t);
                }
            }
        }
        assert!(
            after_rejection > 0,
            "the test needs valid steps after rejected moves"
        );
        assert_eq!(recorder.valid_steps, valid);
        assert_eq!(recorder.moves, recorded.stats());
        assert_eq!(
            recorder.visits.values().map(|v| v.steps).sum::<u64>(),
            valid
        );
        let taken: Vec<u64> = recorder.snapshots.iter().map(|s| s.step).collect();
        assert_eq!(taken, snapshot_steps);
    }

    #[test]
    fn same_seed_same_run() {
        let rules = ruleset(2, &[PATH_BALL]);
        let run = |seed, m| {
            let mut s = Sampler::new(
                RulesetEnergy::new(&rules),
                Settings::open(9, 2, 0.5, seed, m),
                Start::Random { edges: 9 },
                None,
            )
            .expect("valid");
            for _ in 0..3_000 {
                s.step();
            }
            (s.graph().edges(), s.open_nodes(), s.stats(), s.energy())
        };
        assert_eq!(run(1, 0), run(1, 0));
        assert_eq!(run(1, 2), run(1, 2));
        assert_ne!(run(1, 0).0, run(2, 0).0);
    }

    #[test]
    fn stats_add_up_and_the_open_set_keeps_its_size() {
        let settings = Settings::open(6, 2, 1.0, 9, 2);
        let mut s = Sampler::new(ZeroEnergy, settings, Start::Empty, None).expect("valid");
        let mut accepted = 0;
        for _ in 0..10_000 {
            if let StepResult::Accepted(_) = s.step() {
                accepted += 1;
            }
            assert!((0..6).all(|v| s.graph().degree(v) <= 2));
            assert_eq!(s.is_open().iter().filter(|&&o| o).count(), 2);
            assert_eq!(s.open.len() + s.closed.len(), 6);
            assert!(s.open.iter().all(|&v| s.is_open[v as usize]));
            assert!(s.closed.iter().all(|&v| !s.is_open[v as usize]));
        }
        let st = s.stats();
        assert_eq!(
            st.proposed_add + st.proposed_remove + st.proposed_swap,
            10_000
        );
        assert_eq!(
            st.accepted_add + st.accepted_remove + st.accepted_swap,
            accepted
        );
        assert_eq!(
            st.accepted_swap, st.proposed_swap,
            "zero energy accepts every swap"
        );
        assert!((st.proposed_swap as f64 / 10_000.0 - DEFAULT_SWAP_RATE).abs() < 0.02);
        assert_eq!(s.version(), accepted);
        assert!(st.capped > 0);
        assert_eq!(s.graph().edges().len(), s.edges.len());
    }

    #[test]
    fn swaps_only_when_possible() {
        //! No swaps in closed mode or with every node open.
        for m in [0, 5] {
            let mut s = Sampler::new(
                ZeroEnergy,
                Settings::open(5, 2, 1.0, 1, m),
                Start::Empty,
                None,
            )
            .expect("valid");
            for _ in 0..2_000 {
                s.step();
            }
            assert_eq!(s.stats().proposed_swap, 0, "m = {m}");
        }
    }

    #[test]
    fn given_open_set_is_used() {
        let s = Sampler::new(
            ZeroEnergy,
            Settings::open(6, 2, 1.0, 1, 2),
            Start::Empty,
            Some(vec![4, 1]),
        )
        .expect("valid");
        assert_eq!(s.open_nodes(), vec![1, 4]);
    }

    #[test]
    fn rejects_bad_settings() {
        let new = |settings: Settings, start: Start, open: Option<Vec<Node>>| {
            Sampler::new(ZeroEnergy, settings, start, open).err()
        };
        assert_eq!(
            new(Settings::closed(4, 2, -1.0, 0), Start::Empty, None),
            Some(SamplerError::Beta(-1.0))
        );
        assert!(new(Settings::closed(4, 2, f64::NAN, 0), Start::Empty, None).is_some());
        assert!(matches!(
            new(
                Settings::closed(4, 1, 1.0, 0),
                Start::Edges(vec![(0, 1), (0, 2)]),
                None
            ),
            Some(SamplerError::Start(GraphError::DegreeCap { .. }))
        ));
        assert_eq!(
            new(Settings::open(4, 2, 1.0, 0, 5), Start::Empty, None),
            Some(SamplerError::OpenCount { open: 5, nodes: 4 })
        );
        let bad_rate = Settings {
            swap_rate: 1.5,
            ..Settings::open(4, 2, 1.0, 0, 1)
        };
        assert_eq!(
            new(bad_rate, Start::Empty, None),
            Some(SamplerError::SwapRate(1.5))
        );
        for open in [vec![1], vec![1, 1], vec![1, 9]] {
            assert!(matches!(
                new(Settings::open(4, 2, 1.0, 0, 2), Start::Empty, Some(open)),
                Some(SamplerError::StartOpen(_))
            ));
        }
    }

    #[test]
    fn tiny_graphs_do_not_panic() {
        for n in 0..3 {
            for m in 0..=n {
                let settings = Settings::open(n, 2, 1.0, 0, m);
                let mut s = Sampler::new(ZeroEnergy, settings, Start::Random { edges: 5 }, None)
                    .expect("valid");
                for _ in 0..100 {
                    s.step();
                }
            }
        }
    }
}
