//! The validity checker: is a graph, with its open set, valid for a ruleset?
//!
//! Steps (the design doc's "Validity checker"):
//!
//! 1. **Global checks.** The graph is connected, and the open set has at most
//!    m nodes when a cap is given. (Simple is guaranteed by [`Graph`].)
//! 2. **Classify nodes** by distance to the nearest open node, from one
//!    multi-source BFS: 0 is open, 1 to r - 1 shadow, r or more core. With no
//!    open nodes every node is core.
//! 3. **Core nodes** see their whole ball: canonical form, then lookup.
//! 4. **Open and shadow nodes** get the embedding check.
//! 5. **Report**: the verdict, every node's class and outcome, the failing
//!    nodes, core fraction and depth, and the sampler's energy.
//!
//! Only types that survived pruning count. A core node whose ball is a pruned
//! type fails: pruning proved no graph can contain that ball, so some other
//! node fails too, but naming this one makes the report sharper. Valid graphs
//! are unaffected, since their balls are never pruned.
//!
//! [`Checker::check_node`] is the per-node function. The full check is that
//! function over every node; the incremental check the sampler gets later
//! will call it on the nodes near a changed edge.

use crate::ball::Node;
use crate::embed::{EmbedQuery, EmbeddingChecker};
use crate::extract::BallExtractor;
use crate::graph::Graph;
use crate::ruleset::CompiledRuleset;

/// Where a node stands relative to the open set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeClass {
    /// In the open set: may have neighbors the graph does not show.
    Open,
    /// Within r - 1 of an open node: its own edges are all present, but its
    /// ball may be incomplete.
    Shadow,
    /// At least r from every open node: its ball is its true ball.
    Core,
}

impl NodeClass {
    /// `"open"`, `"shadow"` or `"core"`, as Python receives it.
    pub fn as_str(self) -> &'static str {
        match self {
            NodeClass::Open => "open",
            NodeClass::Shadow => "shadow",
            NodeClass::Core => "core",
        }
    }
}

/// What checking one node found. Ball indices count the ruleset's balls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Core node whose ball is this kept type. Passes.
    Matched(usize),
    /// Core node whose ball is this type, which pruning removed. Fails.
    Pruned(usize),
    /// Core node whose ball is no type in the ruleset. Fails.
    Unmatched,
    /// Open or shadow node: the kept types its visible ball fits, ascending.
    /// Passes unless empty.
    Compatible(Vec<usize>),
}

impl Outcome {
    pub fn passes(&self) -> bool {
        match self {
            Outcome::Matched(_) => true,
            Outcome::Pruned(_) | Outcome::Unmatched => false,
            Outcome::Compatible(types) => !types.is_empty(),
        }
    }
}

/// One node's class, distance to the open set, and outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeReport {
    pub class: NodeClass,
    /// None when no open node is reachable, as in closed mode.
    pub distance_to_open: Option<u32>,
    pub outcome: Outcome,
}

/// Everything the checker found, rather than a bare verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckReport {
    pub r: u32,
    /// Indexed by node.
    pub nodes: Vec<NodeReport>,
    pub components: usize,
    /// Number of distinct open nodes.
    pub open_count: usize,
    /// The cap m on open nodes, if one was given.
    pub max_open: Option<usize>,
}

impl CheckReport {
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// One component, or an empty graph.
    pub fn is_connected(&self) -> bool {
        self.components <= 1
    }

    pub fn open_within_limit(&self) -> bool {
        self.max_open.is_none_or(|m| self.open_count <= m)
    }

    /// Nodes whose outcome fails, ascending.
    pub fn failing(&self) -> Vec<Node> {
        (0..self.nodes.len())
            .filter(|&v| !self.nodes[v].outcome.passes())
            .map(|v| v as Node)
            .collect()
    }

    /// Valid: connected, open set within the cap, and every node passes.
    pub fn is_valid(&self) -> bool {
        self.is_connected() && self.open_within_limit() && self.failing().is_empty()
    }

    /// The sampler's energy: failing nodes plus (components - 1). Zero
    /// exactly when the graph is valid, apart from the open-set cap, which
    /// the sampler enforces by its moves.
    pub fn energy(&self) -> usize {
        self.failing().len() + self.components.saturating_sub(1)
    }

    /// Share of nodes that are core. 0 for an empty graph.
    pub fn core_fraction(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let core = self
            .nodes
            .iter()
            .filter(|n| n.class == NodeClass::Core)
            .count();
        core as f64 / self.nodes.len() as f64
    }

    /// The largest distance to the open set, minus r: how far the deepest
    /// core node is from seeing the boundary. Negative when there is no core
    /// node. None when some node reaches no open node (always so in closed
    /// mode), since the core is then unbounded.
    pub fn core_depth(&self) -> Option<i64> {
        let mut deepest: i64 = i64::MIN;
        for node in &self.nodes {
            deepest = deepest.max(i64::from(node.distance_to_open?));
        }
        (!self.nodes.is_empty()).then(|| deepest - i64::from(self.r))
    }
}

/// Checks graphs against one compiled ruleset. Holds scratch buffers, so keep
/// one and reuse it.
pub struct Checker<'a> {
    ruleset: &'a CompiledRuleset,
    embedding: EmbeddingChecker,
    extractor: BallExtractor,
}

impl<'a> Checker<'a> {
    pub fn new(ruleset: &'a CompiledRuleset) -> Self {
        Checker {
            ruleset,
            embedding: EmbeddingChecker::new(ruleset),
            extractor: BallExtractor::new(),
        }
    }

    /// Check `graph` with open set `open` (repeats allowed) and an optional
    /// cap `max_open` on its size.
    ///
    /// # Panics
    ///
    /// If an open node is out of range for the graph.
    pub fn check(&mut self, graph: &Graph, open: &[Node], max_open: Option<usize>) -> CheckReport {
        let n = graph.num_nodes();
        let mut is_open = vec![false; n];
        for &v in open {
            assert!(
                (v as usize) < n,
                "open node {v} is out of range for {n} nodes"
            );
            is_open[v as usize] = true;
        }
        let distance = graph.distances_from(open);
        let nodes = (0..n as Node)
            .map(|v| self.check_node(graph, &is_open, &distance, v))
            .collect();
        CheckReport {
            r: self.ruleset.r(),
            nodes,
            components: graph.component_count(),
            open_count: is_open.iter().filter(|&&o| o).count(),
            max_open,
        }
    }

    /// Check one node, given every node's open flag and distance to the open
    /// set (`u32::MAX` where none is reachable).
    pub fn check_node(
        &mut self,
        graph: &Graph,
        is_open: &[bool],
        distance_to_open: &[u32],
        v: Node,
    ) -> NodeReport {
        let r = self.ruleset.r();
        let d = distance_to_open[v as usize];
        let class = if d == 0 {
            NodeClass::Open
        } else if d < r {
            NodeClass::Shadow
        } else {
            NodeClass::Core
        };
        let ball = self.extractor.extract(graph, v, r);
        let outcome = if class == NodeClass::Core {
            match self.ruleset.index_of(ball.to_compiled().canonical_key()) {
                Some(i) if self.ruleset.pruning().is_kept(i) => Outcome::Matched(i),
                Some(i) => Outcome::Pruned(i),
                None => Outcome::Unmatched,
            }
        } else {
            let query = EmbedQuery::new(&ball, is_open, distance_to_open, r);
            Outcome::Compatible(self.embedding.compatible(&query))
        };
        NodeReport {
            class,
            distance_to_open: (d != u32::MAX).then_some(d),
            outcome,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Edge;
    use crate::testing::{
        census_ruleset, cut_patch, graph, path, random_graph, ruleset, Rng, PATH_BALL,
    };

    fn cycle(n: usize) -> Vec<Edge> {
        let mut edges = path(n);
        edges.push((0, n as Node - 1));
        edges
    }

    #[test]
    fn star_example_fails_everywhere_once_pruned() {
        //! The design doc's star, closed, r = 2. Node a's ball is no type; the
        //! leaves' balls are the star type, which pruning removed.
        let star: &[Edge] = &[(0, 1), (1, 2), (1, 3)];
        let rules = ruleset(2, &[star]);
        let report = Checker::new(&rules).check(&graph(4, star), &[], None);
        let outcomes: Vec<&Outcome> = report.nodes.iter().map(|n| &n.outcome).collect();
        assert_eq!(
            outcomes,
            [
                &Outcome::Pruned(0),
                &Outcome::Unmatched,
                &Outcome::Pruned(0),
                &Outcome::Pruned(0)
            ]
        );
        assert!(!report.is_valid());
        assert_eq!(report.energy(), 4);
    }

    #[test]
    fn cycles_are_valid_from_six_nodes() {
        //! L = {path ball}, closed: a cycle is valid once n >= 2r + 2 = 6.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut checker = Checker::new(&rules);
        for n in 3..12 {
            let report = checker.check(&graph(n, &cycle(n)), &[], None);
            assert_eq!(report.is_valid(), n >= 6, "cycle of {n}");
        }
    }

    #[test]
    fn open_path_needs_room_for_its_open_ends() {
        //! Same L, a path with both ends open: valid at m = 2, not at m = 0.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut checker = Checker::new(&rules);
        let g = graph(9, &path(9));
        let report = checker.check(&g, &[0, 8], Some(2));
        assert!(report.is_valid());
        assert_eq!(report.failing(), [] as [Node; 0]);
        let report = checker.check(&g, &[0, 8], Some(0));
        assert!(!report.is_valid() && !report.open_within_limit());
        assert_eq!(report.failing(), [] as [Node; 0]); // every node passes; the cap fails
        let report = checker.check(&g, &[], Some(0));
        assert_eq!(report.failing(), [0, 1, 7, 8]);
    }

    #[test]
    fn classes_core_fraction_and_depth() {
        let rules = ruleset(2, &[PATH_BALL]);
        let report = Checker::new(&rules).check(&graph(9, &path(9)), &[0, 8], None);
        let classes: Vec<&str> = report.nodes.iter().map(|n| n.class.as_str()).collect();
        assert_eq!(
            classes,
            ["open", "shadow", "core", "core", "core", "core", "core", "shadow", "open"]
        );
        assert_eq!(report.nodes[4].distance_to_open, Some(4));
        assert_eq!(report.core_fraction(), 5.0 / 9.0);
        assert_eq!(report.core_depth(), Some(2));
        // Closed: unbounded core.
        let report = Checker::new(&rules).check(&graph(9, &cycle(9)), &[], None);
        assert_eq!((report.core_fraction(), report.core_depth()), (1.0, None));
    }

    #[test]
    fn disconnected_graph_is_invalid_even_if_every_node_passes() {
        let rules = ruleset(2, &[PATH_BALL]);
        let mut edges = cycle(6);
        edges.extend(cycle(7).iter().map(|&(a, b)| (a + 6, b + 6)));
        let report = Checker::new(&rules).check(&graph(13, &edges), &[], None);
        assert_eq!(report.failing(), [] as [Node; 0]);
        assert_eq!(report.components, 2);
        assert!(!report.is_valid());
        assert_eq!(report.energy(), 1);
    }

    #[test]
    fn reference_graph_is_valid_for_its_own_census() {
        let mut rng = Rng(31);
        let mut found = 0;
        while found < 20 {
            let g = random_graph(40, 60 + rng.below(30), &mut rng);
            if g.component_count() != 1 {
                continue;
            }
            found += 1;
            for r in 1..=3 {
                let rules = census_ruleset(&g, r);
                let report = Checker::new(&rules).check(&g, &[], None);
                assert!(report.is_valid(), "r = {r}, failing {:?}", report.failing());
            }
        }
    }

    #[test]
    fn cut_patches_are_valid() {
        //! The design doc's cut test, through the whole checker.
        let mut rng = Rng(32);
        for round in 0..40 {
            let r = 1 + (round % 3) as u32;
            let g = random_graph(60, 80 + rng.below(40), &mut rng);
            let rules = census_ruleset(&g, r);
            let (patch, open, _) = cut_patch(&g, 30, 0, &mut rng);
            let report = Checker::new(&rules).check(&patch, &open, Some(open.len()));
            assert!(
                report.is_valid(),
                "round {round}: failing {:?}",
                report.failing()
            );
        }
    }

    #[test]
    fn renumbering_never_changes_a_verdict() {
        let mut rng = Rng(33);
        for round in 0..40 {
            let r = 1 + (round % 3) as u32;
            let reference = random_graph(40, 55 + rng.below(25), &mut rng);
            let rules = census_ruleset(&reference, r);
            let (patch, open, _) = cut_patch(&reference, 20, round % 2 * 3, &mut rng);
            let n = patch.num_nodes();
            let mut order: Vec<Node> = (0..n as Node).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i + 1));
            }
            let renamed: Vec<Edge> = patch
                .edges()
                .iter()
                .map(|&(a, b)| (order[a as usize], order[b as usize]))
                .collect();
            let renamed_open: Vec<Node> = open.iter().map(|&v| order[v as usize]).collect();
            let mut checker = Checker::new(&rules);
            let before = checker.check(&patch, &open, None);
            let after = checker.check(&graph(n, &renamed), &renamed_open, None);
            assert_eq!(before.is_valid(), after.is_valid(), "round {round}");
            for (v, &renamed_v) in order.iter().enumerate() {
                assert_eq!(
                    before.nodes[v], after.nodes[renamed_v as usize],
                    "round {round}, node {v}"
                );
            }
        }
    }
}
