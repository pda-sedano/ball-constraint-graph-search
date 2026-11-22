//! The sampler, version 1: one Metropolis-Hastings chain over numbered graphs
//! on n nodes, closed mode, fixed β, full recheck after every move.
//!
//! # The chain
//!
//! Each step proposes one edge change. With probability 1/2 it removes a
//! uniformly random edge; otherwise it adds a uniformly random non-edge. An
//! addition that would push a node past the degree cap is rejected at once,
//! which is the same as giving those graphs infinite energy. A proposed move
//! is accepted with probability
//!
//! ```text
//! remove: min(1, exp(-β ΔU) · E / (N - E + 1))
//! add:    min(1, exp(-β ΔU) · (N - E) / (E + 1))
//! ```
//!
//! where E is the current edge count and N = n(n-1)/2 the number of node
//! pairs. The second factor corrects for the proposal not being symmetric, so
//! the chain's stationary distribution gives every numbered graph under the
//! cap weight exp(-βU). With an energy that is 0 exactly on valid graphs, the
//! valid graphs all get the same weight at any β: sampling the chain at U = 0
//! is uniform over valid numbered graphs.
//!
//! # Recording
//!
//! [`Recorder`] runs a sampler and keeps what the design doc asks for. Two
//! rules keep it unbiased. The state after every step counts, including steps
//! whose move was rejected, since the chain's time at a state is its weight.
//! Snapshots are taken at fixed step numbers (every `snapshot_every` steps,
//! kept if U = 0 then), never "at the first valid state after a gap", which
//! would favour valid graphs that are easy to walk into.

use std::collections::HashMap;
use std::fmt;

use crate::ball::{Edge, Node};
use crate::canonical::{automorphism_count, canonical_graph};
use crate::check::Checker;
use crate::graph::{Graph, GraphError};
use crate::rng::Rng;
use crate::ruleset::CompiledRuleset;

/// What the chain minimizes. Any energy that is 0 exactly on the graphs to be
/// sampled gives the same uniform result, so it can be tuned for speed.
pub trait Energy {
    fn energy(&mut self, graph: &Graph) -> f64;
}

/// The design doc's energy in closed mode: the number of nodes whose ball is
/// not a kept type, plus (components - 1).
pub struct RulesetEnergy<'a> {
    checker: Checker<'a>,
    is_open: Vec<bool>,
    distance: Vec<u32>,
    passing: Vec<bool>,
}

impl<'a> RulesetEnergy<'a> {
    pub fn new(ruleset: &'a CompiledRuleset) -> Self {
        RulesetEnergy {
            checker: Checker::new(ruleset),
            is_open: Vec::new(),
            distance: Vec::new(),
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
    fn energy(&mut self, graph: &Graph) -> f64 {
        let n = graph.num_nodes();
        // Closed mode: nothing is open, so every node is core.
        self.is_open.resize(n, false);
        self.distance.resize(n, u32::MAX);
        self.passing.clear();
        for v in 0..n as Node {
            let report = self
                .checker
                .check_node(graph, &self.is_open, &self.distance, v);
            self.passing.push(report.outcome.passes());
        }
        let failing = self.passing.iter().filter(|&&p| !p).count();
        (failing + graph.component_count().saturating_sub(1)) as f64
    }
}

/// Energy 0 everywhere: the chain is then uniform over every numbered graph
/// under the degree cap. For testing the chain apart from the checker.
pub struct ZeroEnergy;

impl Energy for ZeroEnergy {
    fn energy(&mut self, _graph: &Graph) -> f64 {
        0.0
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
    /// A starting edge could not be added.
    Start(GraphError),
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerError::Beta(beta) => write!(f, "beta must be finite and >= 0, got {beta}"),
            SamplerError::Start(e) => write!(f, "bad starting graph: {e}"),
        }
    }
}

impl std::error::Error for SamplerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Remove,
    Add,
}

/// What one step did. Every variant except `Accepted` leaves the graph as
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
    /// Additions rejected by the degree cap (counted in `proposed_add`).
    pub capped: u64,
    /// Steps with nothing to propose (counted in the proposals).
    pub no_candidate: u64,
}

/// The Metropolis-Hastings acceptance probability for a move from a graph
/// with `edges` edges, out of `pairs` node pairs, changing the energy by
/// `delta`.
pub fn acceptance(mv: Move, edges: usize, pairs: usize, delta: f64, beta: f64) -> f64 {
    let proposal_ratio = match mv {
        Move::Remove => edges as f64 / (pairs - edges + 1) as f64,
        Move::Add => (pairs - edges) as f64 / (edges + 1) as f64,
    };
    ((-beta * delta).exp() * proposal_ratio).min(1.0)
}

/// One chain.
pub struct Sampler<E> {
    energy_fn: E,
    graph: Graph,
    /// Every edge once, in no order, so a uniformly random one is one draw.
    edges: Vec<Edge>,
    energy: f64,
    beta: f64,
    rng: Rng,
    pairs: usize,
    steps: u64,
    stats: MoveStats,
}

impl<E: Energy> Sampler<E> {
    /// A chain on `num_nodes` nodes whose degrees may not exceed `max_degree`.
    /// The random start, if any, draws from the same seeded generator.
    pub fn new(
        mut energy_fn: E,
        num_nodes: usize,
        max_degree: usize,
        beta: f64,
        seed: u64,
        start: Start,
    ) -> Result<Self, SamplerError> {
        if !(beta.is_finite() && beta >= 0.0) {
            return Err(SamplerError::Beta(beta));
        }
        let mut rng = Rng::new(seed);
        let pairs = num_nodes * num_nodes.saturating_sub(1) / 2;
        let graph = match start {
            Start::Empty => Graph::new(num_nodes, max_degree),
            Start::Edges(edges) => {
                Graph::from_edges(num_nodes, max_degree, &edges).map_err(SamplerError::Start)?
            }
            Start::Random { edges } => {
                let mut graph = Graph::new(num_nodes, max_degree);
                let target = edges.min(pairs);
                let mut attempts = 0;
                while graph.num_edges() < target && attempts < 100 * target {
                    attempts += 1;
                    let (u, v) = random_pair(&mut rng, num_nodes);
                    let _ = graph.add_edge(u, v); // skips repeats and capped pairs
                }
                graph
            }
        };
        let edges = graph.edges();
        let energy = energy_fn.energy(&graph);
        Ok(Sampler {
            energy_fn,
            graph,
            edges,
            energy,
            beta,
            rng,
            pairs,
            steps: 0,
            stats: MoveStats::default(),
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn stats(&self) -> MoveStats {
        self.stats
    }

    /// The number of accepted moves so far: changes exactly when the graph does.
    pub fn version(&self) -> u64 {
        self.stats.accepted_add + self.stats.accepted_remove
    }

    pub fn energy_fn(&self) -> &E {
        &self.energy_fn
    }

    /// One proposal, accepted or not.
    pub fn step(&mut self) -> StepResult {
        self.steps += 1;
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
            let new = self.energy_fn.energy(&self.graph);
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
            let new = self.energy_fn.energy(&self.graph);
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
    /// Snapshot the graph at every multiple of this many recorded steps, if
    /// it is valid then.
    pub snapshot_every: Option<u64>,
    /// Count the steps spent on each valid unlabelled graph, keyed by its
    /// canonical graph6. Only useful at small n, where graphs recur.
    pub count_graphs: bool,
    /// Record the energy at every multiple of this many recorded steps.
    pub trace_every: Option<u64>,
}

/// A valid graph seen at a snapshot step.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// The chain's step number, burn-in included.
    pub step: u64,
    pub edges: Vec<Edge>,
    pub automorphisms: f64,
}

/// Time spent on one unlabelled valid graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Visits {
    pub steps: u64,
    pub automorphisms: f64,
}

/// Runs a sampler for a burn-in and then a number of recorded steps, in as
/// many calls to [`Recorder::advance`] as the caller likes (so Python can
/// check for Ctrl-C between them).
#[derive(Debug, Clone)]
pub struct Recorder {
    options: RecordOptions,
    done: u64,
    /// Recorded steps at which the graph was valid.
    pub valid_steps: u64,
    /// By canonical graph6, when `count_graphs` is on.
    pub visits: HashMap<String, Visits>,
    pub snapshots: Vec<Snapshot>,
    /// `(step, energy)` pairs, when `trace_every` is set.
    pub trace: Vec<(u64, f64)>,
    /// The sampler version and canonical graph6 of the last valid graph keyed,
    /// so a graph that has not changed is not canonicalized again.
    cached: Option<(u64, String, f64)>,
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
            sampler.step();
            self.done += 1;
            if self.done <= self.options.burn_in {
                continue;
            }
            let t = self.done - self.options.burn_in;
            if self.options.trace_every.is_some_and(|k| t % k == 0) {
                self.trace.push((sampler.steps(), sampler.energy()));
            }
            if sampler.energy() != 0.0 {
                continue;
            }
            self.valid_steps += 1;
            if self.options.count_graphs {
                let (key, automorphisms) = self.key(sampler);
                self.visits
                    .entry(key)
                    .or_insert(Visits {
                        steps: 0,
                        automorphisms,
                    })
                    .steps += 1;
            }
            if self.options.snapshot_every.is_some_and(|k| t % k == 0) {
                let automorphisms = if self.options.count_graphs {
                    self.key(sampler).1
                } else {
                    automorphism_count(sampler.graph())
                };
                self.snapshots.push(Snapshot {
                    step: sampler.steps(),
                    edges: sampler.graph().edges(),
                    automorphisms,
                });
            }
        }
        count
    }

    fn key<E: Energy>(&mut self, sampler: &Sampler<E>) -> (String, f64) {
        match &self.cached {
            Some((version, key, aut)) if *version == sampler.version() => (key.clone(), *aut),
            _ => {
                let form = canonical_graph(sampler.graph());
                let key = form.graph6();
                self.cached = Some((sampler.version(), key.clone(), form.automorphisms));
                (key, form.automorphisms)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn mask_of(graph: &Graph, index: &HashMap<Edge, usize>) -> usize {
        graph.edges().iter().map(|e| 1 << index[e]).sum()
    }

    /// An arbitrary fixed energy for every graph on n nodes, by mask.
    struct TableEnergy {
        index: HashMap<Edge, usize>,
        table: Vec<f64>,
    }

    impl Energy for TableEnergy {
        fn energy(&mut self, graph: &Graph) -> f64 {
            self.table[mask_of(graph, &self.index)]
        }
    }

    fn random_table(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..1usize << (n * (n - 1) / 2))
            .map(|_| (rng.below(4)) as f64)
            .collect()
    }

    fn degrees_ok(mask: usize, pairs: &[Edge], n: usize, cap: usize) -> bool {
        let mut degree = vec![0; n];
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
        //! Every pair of graphs on 4 nodes one edge apart, under two caps,
        //! with arbitrary energies: π(x) P(x→y) = π(y) P(y→x), where P is
        //! built from the proposal (½ remove a uniform edge, ½ add a uniform
        //! non-edge, capped additions rejected) and `acceptance`.
        let n = 4;
        let index = pair_index(n);
        let mut pairs: Vec<Edge> = vec![(0, 0); index.len()];
        for (&e, &i) in &index {
            pairs[i] = e;
        }
        let total = pairs.len();
        for cap in [2, 3] {
            for (seed, beta) in [(1, 0.7), (2, 2.0), (3, 0.0)] {
                let energy = random_table(n, seed);
                let weight = |x: usize| (-beta * energy[x]).exp();
                let transition = |x: usize, i: usize| -> f64 {
                    let e = x.count_ones() as usize;
                    let y = x ^ (1 << i);
                    let delta = energy[y] - energy[x];
                    if x >> i & 1 == 1 {
                        0.5 / e as f64 * acceptance(Move::Remove, e, total, delta, beta)
                    } else if degrees_ok(y, &pairs, n, cap) {
                        0.5 / (total - e) as f64 * acceptance(Move::Add, e, total, delta, beta)
                    } else {
                        0.0
                    }
                };
                for x in (0..1 << total).filter(|&x| degrees_ok(x, &pairs, n, cap)) {
                    for i in 0..total {
                        let y = x ^ (1 << i);
                        if !degrees_ok(y, &pairs, n, cap) {
                            assert_eq!(transition(x, i), 0.0);
                            continue;
                        }
                        let forward = weight(x) * transition(x, i);
                        let backward = weight(y) * transition(y, i);
                        assert!(
                            (forward - backward).abs() <= 1e-12 * forward.max(backward),
                            "cap {cap}, beta {beta}: {x:06b} <-> {y:06b}: {forward} vs {backward}"
                        );
                    }
                }
            }
        }
    }

    /// Run the real chain on 4 nodes and compare its time at each graph with
    /// the exact stationary distribution exp(-βU) under the cap.
    fn check_stationary(energy: Vec<f64>, cap: usize, beta: f64, seed: u64) {
        let n = 4;
        let index = pair_index(n);
        let mut pairs: Vec<Edge> = vec![(0, 0); index.len()];
        for (&e, &i) in &index {
            pairs[i] = e;
        }
        let allowed: Vec<usize> = (0..1 << pairs.len())
            .filter(|&x| degrees_ok(x, &pairs, n, cap))
            .collect();
        let z: f64 = allowed.iter().map(|&x| (-beta * energy[x]).exp()).sum();
        let mut sampler = Sampler::new(
            TableEnergy {
                index: index.clone(),
                table: energy.clone(),
            },
            n,
            cap,
            beta,
            seed,
            Start::Empty,
        )
        .expect("valid");
        let steps = 2_000_000;
        let mut time = vec![0u64; 1 << pairs.len()];
        for _ in 0..steps {
            sampler.step();
            time[mask_of(sampler.graph(), &index)] += 1;
        }
        let distance: f64 = allowed
            .iter()
            .map(|&x| (time[x] as f64 / steps as f64 - (-beta * energy[x]).exp() / z).abs())
            .sum::<f64>()
            / 2.0;
        assert!(distance < 0.01, "total variation {distance}");
        let outside: u64 = (0..time.len())
            .filter(|x| !allowed.contains(x))
            .map(|x| time[x])
            .sum();
        assert_eq!(outside, 0, "the chain left the cap");
    }

    #[test]
    fn zero_energy_chain_is_uniform_under_the_cap() {
        check_stationary(vec![0.0; 64], 3, 1.0, 11);
        check_stationary(vec![0.0; 64], 2, 1.0, 12);
    }

    #[test]
    fn chain_matches_exp_minus_beta_u() {
        check_stationary(random_table(4, 5), 3, 0.8, 13);
        check_stationary(random_table(4, 6), 2, 1.5, 14);
    }

    #[test]
    fn ruleset_energy_matches_the_checker() {
        let rules = ruleset(2, &[PATH_BALL]);
        let mut energy = RulesetEnergy::new(&rules);
        let mut rng = crate::testing::Rng(4);
        for _ in 0..100 {
            let n = 3 + rng.below(12);
            let m = rng.below(2 * n).min(n * (n - 1) / 2);
            let g = random_graph(n, m, &mut rng);
            let report = Checker::new(&rules).check(&g, &[], None);
            assert_eq!(energy.energy(&g), report.energy() as f64);
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
            8,
            2,
            1.0,
            3,
            Start::Random { edges: 8 },
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
        let key = canonical_graph(&Graph::from_edges(8, 2, &cycle).expect("valid")).graph6();
        let keys: Vec<&String> = recorder.visits.keys().collect();
        assert_eq!(keys, [&key]);
        assert_eq!(recorder.visits[&key].steps, recorder.valid_steps);
        assert_eq!(recorder.visits[&key].automorphisms, 16.0);
        assert_eq!(recorder.trace.len(), 20);
        assert_eq!(recorder.trace[0].0, 3_000);
        for s in &recorder.snapshots {
            assert_eq!(s.step % 100, 0);
            assert_eq!(s.automorphisms, 16.0);
            assert_eq!(s.edges.len(), 8);
        }
        assert_eq!(sampler.steps(), 22_000);
    }

    #[test]
    fn recorder_counts_every_step_including_rejections() {
        //! Replays the same seeded chain by hand: the recorder must count
        //! every recorded step at U = 0, not only those right after an
        //! accepted move, and snapshot exactly the valid multiples.
        let rules = ruleset(2, &[PATH_BALL]);
        let new_sampler = || {
            Sampler::new(
                RulesetEnergy::new(&rules),
                7,
                2,
                0.7,
                21,
                Start::Random { edges: 7 },
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
        let run = |seed| {
            let mut s = Sampler::new(
                RulesetEnergy::new(&rules),
                9,
                2,
                0.5,
                seed,
                Start::Random { edges: 9 },
            )
            .expect("valid");
            for _ in 0..3_000 {
                s.step();
            }
            (s.graph().edges(), s.stats(), s.energy())
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1).0, run(2).0);
    }

    #[test]
    fn stats_add_up() {
        let mut s = Sampler::new(ZeroEnergy, 6, 2, 1.0, 9, Start::Empty).expect("valid");
        let mut accepted = 0;
        for _ in 0..10_000 {
            if let StepResult::Accepted(_) = s.step() {
                accepted += 1;
            }
            assert!((0..6).all(|v| s.graph().degree(v) <= 2));
        }
        let st = s.stats();
        assert_eq!(st.proposed_add + st.proposed_remove, 10_000);
        assert_eq!(st.accepted_add + st.accepted_remove, accepted);
        assert_eq!(s.version(), accepted);
        assert!(st.capped > 0);
        assert_eq!(s.graph().edges().len(), s.edges.len());
    }

    #[test]
    fn rejects_bad_settings() {
        assert_eq!(
            Sampler::new(ZeroEnergy, 4, 2, -1.0, 0, Start::Empty).err(),
            Some(SamplerError::Beta(-1.0))
        );
        assert!(Sampler::new(ZeroEnergy, 4, 2, f64::NAN, 0, Start::Empty).is_err());
        assert!(matches!(
            Sampler::new(ZeroEnergy, 4, 1, 1.0, 0, Start::Edges(vec![(0, 1), (0, 2)])),
            Err(SamplerError::Start(GraphError::DegreeCap { .. }))
        ));
    }

    #[test]
    fn tiny_graphs_do_not_panic() {
        for n in 0..3 {
            let mut s =
                Sampler::new(ZeroEnergy, n, 2, 1.0, 0, Start::Random { edges: 5 }).expect("valid");
            for _ in 0..100 {
                s.step();
            }
        }
    }
}
