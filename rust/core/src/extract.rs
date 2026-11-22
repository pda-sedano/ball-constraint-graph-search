//! Ball extraction: the radius-r ball around a node, as seen in a graph.
//!
//! "Visible" because distances are measured inside the graph at hand. For a
//! core node of a patch that is its true ball; for nodes near the patch's
//! boundary it is the part of the ball the patch shows, which is what the
//! embedding check works from.

use std::collections::HashMap;

use crate::ball::{CompiledBall, Edge, Node};
use crate::canonical::CanonicalKey;
use crate::graph::Graph;

/// The radius-r ball around a center node: the subgraph induced on every
/// node within distance r, numbered in BFS order from the center.
///
/// Induced means edges between two nodes both at distance r are included,
/// while edges leaving the ball are not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleBall {
    /// `nodes[i]` is the graph node numbered i. `nodes[0]` is the center.
    nodes: Vec<Node>,
    /// `depths[i]` is node i's distance from the center; non-decreasing.
    depths: Vec<u32>,
    /// Edges in ball numbering, normalized `(min, max)` and sorted.
    edges: Vec<Edge>,
}

impl VisibleBall {
    /// Graph node of each ball node, by ball number.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Distance from the center of each ball node, by ball number.
    pub fn depths(&self) -> &[u32] {
        &self.depths
    }

    /// Edges in ball numbering, normalized and sorted.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Largest distance from the center: r, or less if the ball holds the
    /// center's whole component.
    pub fn depth(&self) -> u32 {
        *self.depths.last().expect("a ball always holds its center")
    }

    /// As a `CompiledBall`, for its canonical key. Always valid: the ball is
    /// connected, numbered 0..k with the center at 0, with no self-loops or
    /// repeated edges.
    pub fn to_compiled(&self) -> CompiledBall {
        CompiledBall::from_checked_edges(&self.edges)
    }
}

/// Extracts balls, reusing its buffers from one call to the next.
///
/// A "stamp" array marks the nodes of the current ball with the current
/// call's tick, so nothing needs clearing between calls. Keep one extractor
/// per thread and reuse it: that is the point.
#[derive(Debug, Clone, Default)]
pub struct BallExtractor {
    /// `stamp[v] == tick` exactly when v is in the ball being extracted.
    stamp: Vec<u32>,
    /// Ball number of v, valid while `stamp[v] == tick`.
    local: Vec<Node>,
    tick: u32,
}

impl BallExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// The radius-r ball around `center` in `graph`. Panics if `center` is
    /// out of range.
    pub fn extract(&mut self, graph: &Graph, center: Node, r: u32) -> VisibleBall {
        assert!(
            (center as usize) < graph.num_nodes(),
            "center {center} is out of range for {} nodes",
            graph.num_nodes()
        );
        let tick = self.next_tick(graph.num_nodes());

        let mut nodes = vec![center];
        let mut depths = vec![0];
        self.stamp[center as usize] = tick;
        self.local[center as usize] = 0;
        let mut head = 0;
        while head < nodes.len() {
            let (u, depth) = (nodes[head], depths[head]);
            head += 1;
            if depth == r {
                continue;
            }
            for &w in graph.neighbors(u) {
                if self.stamp[w as usize] != tick {
                    self.stamp[w as usize] = tick;
                    self.local[w as usize] = nodes.len() as Node;
                    nodes.push(w);
                    depths.push(depth + 1);
                }
            }
        }

        let mut edges = Vec::new();
        for (i, &u) in nodes.iter().enumerate() {
            for &w in graph.neighbors(u) {
                if self.stamp[w as usize] == tick {
                    let j = self.local[w as usize];
                    if (i as Node) < j {
                        edges.push((i as Node, j));
                    }
                }
            }
        }
        edges.sort_unstable();

        VisibleBall {
            nodes,
            depths,
            edges,
        }
    }

    /// Size the buffers for `num_nodes` and start a new tick. On the rare
    /// wraparound, clear the stamps so old marks cannot match.
    fn next_tick(&mut self, num_nodes: usize) -> u32 {
        if self.stamp.len() < num_nodes {
            self.stamp.resize(num_nodes, 0);
            self.local.resize(num_nodes, 0);
        }
        self.tick = self.tick.wrapping_add(1);
        if self.tick == 0 {
            self.stamp.fill(0);
            self.tick = 1;
        }
        self.tick
    }
}

/// One ball shape found in a reference graph.
#[derive(Debug, Clone)]
pub struct CensusEntry {
    pub ball: CompiledBall,
    /// How many nodes have this ball.
    pub count: usize,
    /// The lowest-numbered such node.
    pub example_center: Node,
}

/// Every node's radius-r ball, grouped by canonical key: the distinct shapes
/// a ruleset extracted from this graph would contain, with their counts.
///
/// Most common first, ties broken by canonical key, so the order (and so a
/// ruleset file written from it) does not depend on how the graph is
/// numbered. Counts sum to the number of nodes.
pub fn ball_census(graph: &Graph, r: u32) -> Vec<CensusEntry> {
    let mut extractor = BallExtractor::new();
    let mut by_key: HashMap<CanonicalKey, CensusEntry> = HashMap::new();
    for center in 0..graph.num_nodes() as Node {
        let ball = extractor.extract(graph, center, r).to_compiled();
        if let Some(entry) = by_key.get_mut(ball.canonical_key()) {
            entry.count += 1;
        } else {
            by_key.insert(
                ball.canonical_key().clone(),
                CensusEntry {
                    ball,
                    count: 1,
                    example_center: center,
                },
            );
        }
    }
    let mut entries: Vec<CensusEntry> = by_key.into_values().collect();
    entries.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.ball.canonical_key().cmp(b.ball.canonical_key()))
    });
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile;
    use std::collections::{BTreeSet, VecDeque};

    fn graph(n: usize, edges: &[Edge]) -> Graph {
        Graph::from_edges_with_own_cap(n, edges).expect("valid test graph")
    }

    fn cycle(n: u32) -> Graph {
        let edges: Vec<Edge> = (0..n).map(|v| (v, (v + 1) % n)).collect();
        graph(n as usize, &edges)
    }

    /// A w x h grid, wrapping around into a torus if `torus`.
    fn grid(w: u32, h: u32, torus: bool) -> Graph {
        let id = |x: u32, y: u32| x * h + y;
        let mut edges = Vec::new();
        for x in 0..w {
            for y in 0..h {
                if x + 1 < w || torus {
                    edges.push((id(x, y), id((x + 1) % w, y)));
                }
                if y + 1 < h || torus {
                    edges.push((id(x, y), id(x, (y + 1) % h)));
                }
            }
        }
        graph((w * h) as usize, &edges)
    }

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    fn random_graph(rng: &mut Rng, n: usize, m: usize) -> Graph {
        let mut set = BTreeSet::new();
        for _ in 0..m {
            let (u, v) = (rng.below(n) as Node, rng.below(n) as Node);
            if u != v {
                set.insert((u.min(v), u.max(v)));
            }
        }
        graph(n, &set.into_iter().collect::<Vec<_>>())
    }

    #[test]
    fn cycle_balls_are_induced() {
        //! In a 5-cycle, the two nodes at distance 2 from the center are
        //! adjacent, so the radius-2 ball keeps that edge. At radius 1 the
        //! center's neighbors are not adjacent and nothing extra appears.
        let g = cycle(5);
        let mut x = BallExtractor::new();
        let ball = x.extract(&g, 0, 2);
        assert_eq!(ball.num_nodes(), 5);
        assert_eq!(ball.depths(), [0, 1, 1, 2, 2]);
        assert_eq!(
            ball.edges().len(),
            5,
            "the whole cycle, including the far edge"
        );

        let ball = x.extract(&g, 0, 1);
        assert_eq!(ball.depths(), [0, 1, 1]);
        assert_eq!(ball.edges(), [(0, 1), (0, 2)]);
    }

    #[test]
    fn small_component_gives_a_shallow_ball() {
        let g = graph(6, &[(0, 1), (1, 2), (3, 4)]);
        let mut x = BallExtractor::new();
        let ball = x.extract(&g, 0, 5);
        assert_eq!(
            (ball.num_nodes(), ball.depth()),
            (3, 2),
            "only its own component"
        );
        let isolated = x.extract(&g, 5, 2);
        assert_eq!((isolated.nodes(), isolated.depth()), (&[5][..], 0));
        assert!(isolated.edges().is_empty());
    }

    /// Reference ball by plain BFS over a map: node -> depth, plus induced edges.
    fn reference_ball(g: &Graph, center: Node, r: u32) -> (Vec<(Node, u32)>, BTreeSet<Edge>) {
        let mut depth = std::collections::HashMap::from([(center, 0u32)]);
        let mut queue = VecDeque::from([center]);
        while let Some(u) = queue.pop_front() {
            for &w in g.neighbors(u) {
                if depth[&u] < r && !depth.contains_key(&w) {
                    depth.insert(w, depth[&u] + 1);
                    queue.push_back(w);
                }
            }
        }
        let mut nodes: Vec<(Node, u32)> = depth.iter().map(|(&v, &d)| (v, d)).collect();
        nodes.sort_unstable();
        let edges = g
            .edges()
            .into_iter()
            .filter(|(u, v)| depth.contains_key(u) && depth.contains_key(v))
            .collect();
        (nodes, edges)
    }

    #[test]
    fn matches_a_plain_bfs_on_random_graphs() {
        let mut rng = Rng(99);
        let mut x = BallExtractor::new(); // reused across graphs of different sizes
        for _ in 0..300 {
            let n = 1 + rng.below(60);
            let m = rng.below(2 * n + 1);
            let g = random_graph(&mut rng, n, m);
            for _ in 0..5 {
                let (center, r) = (rng.below(n) as Node, rng.below(5) as u32);
                let ball = x.extract(&g, center, r);
                let (want_nodes, want_edges) = reference_ball(&g, center, r);

                let mut got_nodes: Vec<(Node, u32)> = ball
                    .nodes()
                    .iter()
                    .copied()
                    .zip(ball.depths().iter().copied())
                    .collect();
                got_nodes.sort_unstable();
                assert_eq!(got_nodes, want_nodes);

                let global = |i: Node| ball.nodes()[i as usize];
                let got_edges: BTreeSet<Edge> = ball
                    .edges()
                    .iter()
                    .map(|&(a, b)| (global(a).min(global(b)), global(a).max(global(b))))
                    .collect();
                assert_eq!(got_edges, want_edges);
                assert!(ball.depths().windows(2).all(|w| w[0] <= w[1]), "BFS order");
                assert_eq!(ball.nodes()[0], center);
            }
        }
    }

    #[test]
    fn survives_tick_wraparound() {
        let g = cycle(6);
        let mut x = BallExtractor::new();
        let before = x.extract(&g, 0, 1);
        x.tick = u32::MAX - 1;
        for _ in 0..4 {
            assert_eq!(x.extract(&g, 0, 1), before);
        }
    }

    #[test]
    fn torus_has_one_ball_shape() {
        let census = ball_census(&grid(6, 6, true), 2);
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].count, 36);
        assert_eq!(census[0].ball.layer_sizes(), [1, 4, 8]);
    }

    #[test]
    fn open_grid_census_counts_boundary_shapes() {
        //! A 5x5 grid at radius 1: 12 edge nodes (3 neighbors), 9 interior
        //! (4), 4 corners (2), most common first.
        let census = ball_census(&grid(5, 5, false), 1);
        let summary: Vec<(usize, usize)> = census
            .iter()
            .map(|e| (e.count, e.ball.root_degree()))
            .collect();
        assert_eq!(summary, [(12, 3), (9, 4), (4, 2)]);
        assert_eq!(census.iter().map(|e| e.count).sum::<usize>(), 25);
    }

    #[test]
    fn census_order_ignores_graph_numbering() {
        let mut rng = Rng(5);
        for _ in 0..20 {
            let n = 5 + rng.below(30);
            let g = random_graph(&mut rng, n, 2 * n);
            let mut perm: Vec<Node> = (0..n as Node).collect();
            for i in (1..n).rev() {
                perm.swap(i, rng.below(i + 1));
            }
            let renumbered: Vec<Edge> = g
                .edges()
                .iter()
                .map(|&(u, v)| (perm[u as usize], perm[v as usize]))
                .collect();
            let summary = |g: &Graph| -> Vec<(CanonicalKey, usize)> {
                ball_census(g, 2)
                    .into_iter()
                    .map(|e| (e.ball.canonical_key().clone(), e.count))
                    .collect()
            };
            assert_eq!(summary(&g), summary(&graph(n, &renumbered)));
        }
    }

    #[test]
    fn census_of_a_graph_compiles_without_issues() {
        //! Distinct shapes and full-depth balls: nothing for compile to flag.
        let census = ball_census(&grid(8, 8, false), 2);
        let types: Vec<Vec<Edge>> = census
            .iter()
            .map(|e| e.ball.canonical_key().edges().to_vec())
            .collect();
        let outcome = compile(2, &types);
        assert_eq!(outcome.issues, []);
        assert_eq!(outcome.ruleset.expect("clean").len(), census.len());
    }
}
