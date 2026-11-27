//! Parallel tempering: several copies of the chain at different β, which
//! trade states so the hot copies' exploring reaches the cold copies.
//!
//! # The method
//!
//! K replicas, each an ordinary [`Sampler`], sit on a ladder of inverse
//! temperatures β_0 < β_1 < ... < β_{K-1}, rung 0 the hottest. Between swap
//! rounds every replica takes `swap_interval` steps of its own. Then
//! neighbouring rungs k and k + 1 propose to trade states, accepted with
//! probability
//!
//! ```text
//! min(1, exp((β_k − β_{k+1}) (U_k − U_{k+1})))
//! ```
//!
//! where U_k is the energy of the state at rung k. That is the Metropolis rule
//! for exchanging two states under the product distribution in which rung k
//! has weight exp(-β_k U), so every rung keeps sampling exp(-β_k U) exactly.
//! Rounds alternate between the even pairs (0-1, 2-3, ...) and the odd pairs
//! (1-2, 3-4, ...), so neighbouring proposals in a round never share a rung.
//!
//! A trade swaps the replicas' β values, not their graphs, so it costs
//! nothing. Rung k's statistics are recorded by its own [`Recorder`], fed by
//! whichever replica holds rung k at the time.
//!
//! Every valid state has energy 0, so within any rung valid numbered states
//! all have equal weight: valid samples from every rung can be pooled.
//!
//! # Threads
//!
//! Between swap rounds the replicas are independent, so they run on scoped
//! threads. Each replica owns its random stream and the swaps run on one
//! thread afterwards, so the result is the same for any number of threads.

use std::fmt;

use crate::ball::Node;
use crate::rng::Rng;
use crate::sample::{Energy, Recorder, Sampler, SamplerError, Settings, Start};

/// A tempered run's fixed settings.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderSettings {
    pub num_nodes: usize,
    pub max_degree: usize,
    /// The inverse temperatures, strictly increasing: rung 0 is the hottest.
    pub betas: Vec<f64>,
    pub seed: u64,
    /// m, as in [`Settings::open_nodes`].
    pub open_nodes: usize,
    /// As in [`Settings::swap_rate`]: open-flag swaps within each replica.
    pub swap_rate: f64,
    /// Steps each replica takes between swap rounds.
    pub swap_interval: u64,
    /// Threads to run replicas on between swap rounds; 0 means one per
    /// available core. The result does not depend on it.
    pub threads: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TemperError {
    /// The ladder is empty, not strictly increasing, or has a β that is not
    /// finite and at least 0.
    Betas(String),
    /// `swap_interval` must be at least 1.
    SwapInterval,
    /// The number of energy functions is not the number of rungs.
    Energies {
        given: usize,
        rungs: usize,
    },
    /// Recorders were not one per rung.
    Recorders {
        given: usize,
        rungs: usize,
    },
    Sampler(SamplerError),
}

impl fmt::Display for TemperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemperError::Betas(why) => write!(f, "bad beta ladder: {why}"),
            TemperError::SwapInterval => write!(f, "swap_interval must be at least 1"),
            TemperError::Energies { given, rungs } => {
                write!(f, "{given} energy functions for {rungs} rungs")
            }
            TemperError::Recorders { given, rungs } => {
                write!(f, "{given} recorders for {rungs} rungs")
            }
            TemperError::Sampler(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TemperError {}

/// Swap proposals between one pair of neighbouring rungs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairStats {
    pub attempts: u64,
    pub accepts: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    Hot,
    Cold,
}

/// Replicas on a β ladder.
pub struct Tempering<E> {
    replicas: Vec<Sampler<E>>,
    /// `replica_at[k]`: the replica holding rung k.
    replica_at: Vec<usize>,
    betas: Vec<f64>,
    swap_interval: u64,
    threads: usize,
    rng: Rng,
    /// Steps taken since the last swap round.
    into_round: u64,
    rounds: u64,
    pairs: Vec<PairStats>,
    /// The end of the ladder each replica touched last.
    last_end: Vec<Option<End>>,
    passages: u64,
}

impl<E: Energy + Send> Tempering<E> {
    /// One replica per rung, replica k starting at rung k, each with its own
    /// energy function and a random stream derived from the seed. A random
    /// start gives each replica its own random graph.
    pub fn new(
        energies: Vec<E>,
        settings: LadderSettings,
        start: Start,
        start_open: Option<Vec<Node>>,
    ) -> Result<Self, TemperError> {
        let k = settings.betas.len();
        if k == 0 {
            return Err(TemperError::Betas("no rungs".into()));
        }
        if let Some(b) = settings
            .betas
            .iter()
            .find(|b| !(b.is_finite() && **b >= 0.0))
        {
            return Err(TemperError::Betas(format!(
                "beta {b} is not finite and >= 0"
            )));
        }
        if settings.betas.windows(2).any(|w| w[0] >= w[1]) {
            return Err(TemperError::Betas(format!(
                "betas must increase strictly from the hottest rung, got {:?}",
                settings.betas
            )));
        }
        if settings.swap_interval == 0 {
            return Err(TemperError::SwapInterval);
        }
        if energies.len() != k {
            return Err(TemperError::Energies {
                given: energies.len(),
                rungs: k,
            });
        }
        let mut master = Rng::new(settings.seed);
        let seeds: Vec<u64> = (0..k).map(|_| master.next_u64()).collect();
        let rng = Rng::new(master.next_u64());
        let replicas = energies
            .into_iter()
            .zip(&settings.betas)
            .zip(seeds)
            .map(|((energy, &beta), seed)| {
                let replica_settings = Settings {
                    num_nodes: settings.num_nodes,
                    max_degree: settings.max_degree,
                    beta,
                    seed,
                    open_nodes: settings.open_nodes,
                    swap_rate: settings.swap_rate,
                };
                Sampler::new(energy, replica_settings, start.clone(), start_open.clone())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(TemperError::Sampler)?;
        let threads = if settings.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            settings.threads
        };
        let mut last_end = vec![None; k];
        if k > 1 {
            last_end[0] = Some(End::Hot);
            last_end[k - 1] = Some(End::Cold);
        }
        Ok(Tempering {
            replicas,
            replica_at: (0..k).collect(),
            betas: settings.betas,
            swap_interval: settings.swap_interval,
            threads,
            rng,
            into_round: 0,
            rounds: 0,
            pairs: vec![PairStats::default(); k.saturating_sub(1)],
            last_end,
            passages: 0,
        })
    }

    /// Run up to `max_steps` more steps per replica, recording rung k's steps
    /// in `recorders[k]`, with a swap round after every `swap_interval`
    /// steps. Stops early when the recorders have no steps left. Returns the
    /// steps each replica took. May be called repeatedly; a swap round can
    /// fall between calls.
    pub fn run(&mut self, recorders: &mut [Recorder], max_steps: u64) -> Result<u64, TemperError> {
        let k = self.replicas.len();
        if recorders.len() != k {
            return Err(TemperError::Recorders {
                given: recorders.len(),
                rungs: k,
            });
        }
        let mut done = 0;
        loop {
            let left = (max_steps - done).min(recorders[0].remaining());
            if left == 0 {
                return Ok(done);
            }
            let chunk = left.min(self.swap_interval - self.into_round);
            self.step_all(recorders, chunk);
            done += chunk;
            self.into_round += chunk;
            if self.into_round == self.swap_interval {
                self.into_round = 0;
                self.swap_round();
            }
        }
    }

    /// Every replica takes `steps` steps, recorded by its rung's recorder.
    fn step_all(&mut self, recorders: &mut [Recorder], steps: u64) {
        let mut slots: Vec<Option<&mut Recorder>> = (0..recorders.len()).map(|_| None).collect();
        for (rung, recorder) in recorders.iter_mut().enumerate() {
            slots[self.replica_at[rung]] = Some(recorder);
        }
        let mut work: Vec<(&mut Sampler<E>, &mut Recorder)> = self
            .replicas
            .iter_mut()
            .zip(
                slots
                    .into_iter()
                    .map(|r| r.expect("one recorder per replica")),
            )
            .collect();
        let run = |part: &mut [(&mut Sampler<E>, &mut Recorder)]| {
            for (sampler, recorder) in part.iter_mut() {
                for _ in 0..steps {
                    let result = sampler.step();
                    recorder.record(sampler, result);
                }
            }
        };
        let threads = self.threads.clamp(1, work.len());
        if threads == 1 {
            run(&mut work);
        } else {
            let per_thread = work.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for part in work.chunks_mut(per_thread) {
                    scope.spawn(move || run(part));
                }
            });
        }
    }

    /// Propose trades between neighbouring rungs: even pairs on even rounds,
    /// odd pairs on odd rounds.
    fn swap_round(&mut self) {
        let k = self.replicas.len();
        let first = (self.rounds % 2) as usize;
        self.rounds += 1;
        for low in (first..k.saturating_sub(1)).step_by(2) {
            let high = low + 1;
            let (a, b) = (self.replica_at[low], self.replica_at[high]);
            let delta = (self.betas[low] - self.betas[high])
                * (self.replicas[a].energy() - self.replicas[b].energy());
            self.pairs[low].attempts += 1;
            if self.rng.next_f64() < delta.exp().min(1.0) {
                self.pairs[low].accepts += 1;
                self.replica_at.swap(low, high);
                self.replicas[a].set_beta(self.betas[high]);
                self.replicas[b].set_beta(self.betas[low]);
            }
        }
        if k > 1 {
            self.note_end(self.replica_at[0], End::Hot);
            self.note_end(self.replica_at[k - 1], End::Cold);
        }
    }

    fn note_end(&mut self, replica: usize, end: End) {
        if self.last_end[replica].is_some_and(|last| last != end) {
            self.passages += 1;
        }
        self.last_end[replica] = Some(end);
    }

    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// The replica holding rung k.
    pub fn sampler_at(&self, rung: usize) -> &Sampler<E> {
        &self.replicas[self.replica_at[rung]]
    }

    /// Which replica holds each rung, hottest first.
    pub fn replica_at(&self) -> &[usize] {
        &self.replica_at
    }

    /// Swap proposals between rungs k and k + 1, for each k.
    pub fn pairs(&self) -> &[PairStats] {
        &self.pairs
    }

    /// Times a replica crossed the whole ladder, from one end to the other.
    /// Two passages make a round trip; without them, fresh states found at
    /// the hot end never reach the cold end.
    pub fn passages(&self) -> u64 {
        self.passages
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Edge;
    use crate::canonical::canonical_graph;
    use crate::graph::Graph;
    use crate::sample::{RecordOptions, RulesetEnergy, ZeroEnergy};
    use crate::testing::{ruleset, PATH_BALL};

    const N: usize = 4;

    fn pairs_of_four() -> Vec<Edge> {
        let mut pairs = Vec::new();
        for u in 0..N as Node {
            for v in u + 1..N as Node {
                pairs.push((u, v));
            }
        }
        pairs
    }

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

    /// An arbitrary fixed energy per (graph, open set) on 4 nodes.
    #[derive(Clone)]
    struct TableEnergy {
        pairs: Vec<Edge>,
        table: Vec<f64>,
    }

    impl Energy for TableEnergy {
        fn energy(&mut self, graph: &Graph, is_open: &[bool]) -> f64 {
            self.table[mask_of(graph, &self.pairs) * 16 + open_mask(is_open)]
        }
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

    fn quiet_recorders(k: usize, steps: u64) -> Vec<Recorder> {
        (0..k)
            .map(|_| {
                Recorder::new(RecordOptions {
                    burn_in: 0,
                    steps,
                    snapshot_every: None,
                    count_graphs: false,
                    trace_every: None,
                })
            })
            .collect()
    }

    fn settings(betas: &[f64], cap: usize, m: usize, seed: u64, interval: u64) -> LadderSettings {
        LadderSettings {
            num_nodes: N,
            max_degree: cap,
            betas: betas.to_vec(),
            seed,
            open_nodes: m,
            swap_rate: 0.3,
            swap_interval: interval,
            threads: 1,
        }
    }

    /// Each rung's state at the end of every round, against the exact
    /// exp(-β_k U) over states under the cap with m open nodes.
    fn check_rungs(m: usize, seed: u64) {
        let pairs = pairs_of_four();
        let mut rng = Rng::new(seed);
        let table: Vec<f64> = (0..64 * 16).map(|_| rng.below(4) as f64).collect();
        let betas = [0.3, 1.0, 2.5];
        let energy = TableEnergy {
            pairs: pairs.clone(),
            table: table.clone(),
        };
        let interval = 5;
        let rounds = 300_000u64;
        let mut tempering = Tempering::new(
            vec![energy; betas.len()],
            settings(&betas, 3, m, seed, interval),
            Start::Empty,
            None,
        )
        .expect("valid");
        let mut recorders = quiet_recorders(betas.len(), interval * rounds);
        let mut time = vec![vec![0u64; table.len()]; betas.len()];
        for _ in 0..rounds {
            tempering.run(&mut recorders, interval).expect("valid");
            for (rung, counts) in time.iter_mut().enumerate() {
                let s = tempering.sampler_at(rung);
                counts[mask_of(s.graph(), &pairs) * 16 + open_mask(s.is_open())] += 1;
                assert_eq!(s.beta(), betas[rung]);
            }
        }
        let allowed: Vec<usize> = (0..64)
            .filter(|&x| degrees_ok(x, &pairs, 3))
            .flat_map(|x| {
                (0..16)
                    .filter(move |o: &usize| o.count_ones() as usize == m)
                    .map(move |o| x * 16 + o)
            })
            .collect();
        for (rung, beta) in betas.iter().enumerate() {
            let z: f64 = allowed.iter().map(|&s| (-beta * table[s]).exp()).sum();
            let distance: f64 = allowed
                .iter()
                .map(|&s| {
                    (time[rung][s] as f64 / rounds as f64 - (-beta * table[s]).exp() / z).abs()
                })
                .sum::<f64>()
                / 2.0;
            assert!(
                distance < 0.02,
                "m {m}, rung {rung}: total variation {distance}"
            );
        }
        let accepted: u64 = tempering.pairs().iter().map(|p| p.accepts).sum();
        assert!(accepted > 0, "no swaps were accepted");
    }

    #[test]
    fn every_rung_samples_exp_minus_beta_u() {
        check_rungs(0, 31);
        check_rungs(1, 32);
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let rules = ruleset(2, &[PATH_BALL]);
        let run = |threads: usize| {
            let betas = vec![0.2, 0.5, 1.0, 2.0, 3.5];
            let energies = (0..betas.len())
                .map(|_| RulesetEnergy::new(&rules))
                .collect();
            let mut s = settings(&betas, 2, 0, 7, 50);
            s.num_nodes = 9;
            s.threads = threads;
            let mut tempering =
                Tempering::new(energies, s, Start::Random { edges: 9 }, None).expect("valid");
            let mut recorders: Vec<Recorder> = (0..betas.len())
                .map(|_| {
                    Recorder::new(RecordOptions {
                        burn_in: 100,
                        steps: 4_000,
                        snapshot_every: Some(100),
                        count_graphs: true,
                        trace_every: Some(500),
                    })
                })
                .collect();
            while tempering.run(&mut recorders, 777).expect("valid") > 0 {}
            let rungs: Vec<_> = (0..betas.len())
                .map(|k| {
                    let r = &recorders[k];
                    let s = tempering.sampler_at(k);
                    (
                        r.valid_steps,
                        r.snapshots.clone(),
                        r.trace.clone(),
                        r.moves,
                        s.graph().edges(),
                    )
                })
                .collect();
            (rungs, tempering.pairs().to_vec(), tempering.passages())
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one, run(0));
    }

    #[test]
    fn equal_energies_always_swap_and_cross_the_ladder() {
        //! With energy 0 every trade is accepted, so replicas shuffle along
        //! the ladder and cross it end to end.
        let betas = [0.1, 0.4, 1.0, 2.0];
        let energies = (0..4).map(|_| ZeroEnergy).collect();
        let mut tempering =
            Tempering::new(energies, settings(&betas, 3, 0, 3, 10), Start::Empty, None)
                .expect("valid");
        let mut recorders = quiet_recorders(4, 20_000);
        while tempering.run(&mut recorders, 5_000).expect("valid") > 0 {}
        assert_eq!(tempering.rounds(), 2_000);
        for (k, pair) in tempering.pairs().iter().enumerate() {
            assert_eq!(pair.accepts, pair.attempts);
            assert_eq!(pair.attempts, 1_000, "pair {k}: half the rounds each");
        }
        assert!(tempering.passages() > 10);
        let mut held: Vec<usize> = tempering.replica_at().to_vec();
        held.sort_unstable();
        assert_eq!(held, [0, 1, 2, 3]);
    }

    #[test]
    fn recorders_follow_rungs_not_replicas() {
        //! Every recorded step goes to some rung, and each rung's recorder sees
        //! exactly the steps taken there: totals match, and every valid state
        //! any rung counts is the 8-cycle, the path ball's only valid graph.
        let rules = ruleset(2, &[PATH_BALL]);
        let betas = vec![0.3, 0.8, 1.5, 3.0];
        let energies = (0..betas.len())
            .map(|_| RulesetEnergy::new(&rules))
            .collect();
        let mut s = settings(&betas, 2, 0, 11, 25);
        s.num_nodes = 8;
        let mut tempering =
            Tempering::new(energies, s, Start::Random { edges: 8 }, None).expect("valid");
        let mut recorders: Vec<Recorder> = (0..betas.len())
            .map(|_| {
                Recorder::new(RecordOptions {
                    burn_in: 500,
                    steps: 20_000,
                    snapshot_every: None,
                    count_graphs: true,
                    trace_every: None,
                })
            })
            .collect();
        while tempering.run(&mut recorders, 3_000).expect("valid") > 0 {}
        let cycle: Vec<Edge> = (0..8).map(|v| (v, (v + 1) % 8)).collect();
        let key = canonical_graph(&Graph::from_edges(8, 2, &cycle).expect("valid")).key();
        let total_moves: u64 = recorders
            .iter()
            .map(|r| r.moves.proposed_add + r.moves.proposed_remove)
            .sum();
        assert_eq!(total_moves, 4 * 20_500);
        assert!(
            recorders[3].valid_steps > 0,
            "the coldest rung never found the cycle"
        );
        for r in &recorders {
            assert!(r.visits.keys().all(|k| *k == key));
        }
        assert!(recorders[3].valid_steps > recorders[0].valid_steps);
    }

    #[test]
    fn rejects_bad_settings() {
        let new = |betas: &[f64], interval: u64, energies: usize| {
            Tempering::new(
                (0..energies).map(|_| ZeroEnergy).collect(),
                settings(betas, 3, 0, 1, interval),
                Start::Empty,
                None,
            )
            .err()
        };
        assert!(matches!(new(&[], 10, 0), Some(TemperError::Betas(_))));
        assert!(matches!(
            new(&[1.0, 0.5], 10, 2),
            Some(TemperError::Betas(_))
        ));
        assert!(matches!(
            new(&[1.0, 1.0], 10, 2),
            Some(TemperError::Betas(_))
        ));
        assert!(matches!(
            new(&[-1.0, 1.0], 10, 2),
            Some(TemperError::Betas(_))
        ));
        assert_eq!(new(&[0.5, 1.0], 0, 2), Some(TemperError::SwapInterval));
        assert_eq!(
            new(&[0.5, 1.0], 10, 3),
            Some(TemperError::Energies { given: 3, rungs: 2 })
        );
        let mut t = Tempering::new(
            vec![ZeroEnergy, ZeroEnergy],
            settings(&[0.5, 1.0], 3, 0, 1, 10),
            Start::Empty,
            None,
        )
        .expect("valid");
        assert_eq!(
            t.run(&mut quiet_recorders(3, 10), 10),
            Err(TemperError::Recorders { given: 3, rungs: 2 })
        );
    }

    #[test]
    fn a_single_rung_is_a_plain_chain() {
        let mut t = Tempering::new(
            vec![ZeroEnergy],
            settings(&[1.0], 3, 0, 1, 10),
            Start::Empty,
            None,
        )
        .expect("valid");
        let mut recorders = quiet_recorders(1, 1_000);
        assert_eq!(t.run(&mut recorders, 5_000), Ok(1_000));
        assert!(t.pairs().is_empty());
        assert_eq!(t.passages(), 0);
    }
}
