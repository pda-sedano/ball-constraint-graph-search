//! Helpers shared by the unit tests: a small random generator, random
//! graphs and patches, and rulesets built from edge lists or a census.

use crate::ball::{Edge, Node};
use crate::compile::compile;
use crate::extract::ball_census;
use crate::graph::Graph;
use crate::ruleset::CompiledRuleset;

const NONE: u32 = u32::MAX;

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub(crate) fn ruleset(r: u32, types: &[&[Edge]]) -> CompiledRuleset {
    let types: Vec<Vec<Edge>> = types.iter().map(|t| t.to_vec()).collect();
    compile(r, &types).ruleset.expect("no errors")
}

pub(crate) fn graph(n: usize, edges: &[Edge]) -> Graph {
    Graph::from_edges_with_own_cap(n, edges).expect("valid graph")
}

pub(crate) fn path(n: usize) -> Vec<Edge> {
    (1..n as Node).map(|v| (v - 1, v)).collect()
}

pub(crate) const PATH_BALL: &[Edge] = &[(0, 1), (0, 2), (1, 3), (2, 4)];

/// Random patches: cut a connected region out of `g`, mark nodes that
/// lost a neighbor open, optionally flip a few edges.
pub(crate) fn cut_patch(
    g: &Graph,
    size: usize,
    flips: usize,
    rng: &mut Rng,
) -> (Graph, Vec<Node>, Vec<Node>) {
    let n = g.num_nodes();
    let mut index = vec![NONE; n];
    let mut region: Vec<Node> = Vec::new();
    let mut frontier = vec![rng.below(n) as Node];
    while region.len() < size && !frontier.is_empty() {
        let v = frontier.swap_remove(rng.below(frontier.len()));
        if index[v as usize] == NONE {
            index[v as usize] = region.len() as u32;
            region.push(v);
            frontier.extend(
                g.neighbors(v)
                    .iter()
                    .filter(|&&w| index[w as usize] == NONE),
            );
        }
    }
    let mut edges = Vec::new();
    let mut open = Vec::new();
    for (i, &v) in region.iter().enumerate() {
        let mut lost = false;
        for &w in g.neighbors(v) {
            match index[w as usize] {
                NONE => lost = true,
                j if (i as u32) < j => edges.push((i as Node, j)),
                _ => {}
            }
        }
        if lost {
            open.push(i as Node);
        }
    }
    let k = region.len();
    let mut patch = Graph::new(k, g.max_degree() + flips);
    for &(a, b) in &edges {
        patch.add_edge(a, b).expect("fits");
    }
    for _ in 0..flips {
        let (a, b) = (rng.below(k) as Node, rng.below(k) as Node);
        if a != b && !patch.remove_edge(a, b) {
            patch.add_edge(a, b).expect("cap leaves room");
        }
    }
    (patch, open, region)
}

pub(crate) fn random_graph(n: usize, m: usize, rng: &mut Rng) -> Graph {
    let mut g = Graph::new(n, n);
    let mut added = 0;
    while added < m {
        let (a, b) = (rng.below(n) as Node, rng.below(n) as Node);
        if a != b && g.add_edge(a, b).expect("in range") {
            added += 1;
        }
    }
    graph(n, &g.edges())
}

pub(crate) fn census_ruleset(g: &Graph, r: u32) -> CompiledRuleset {
    let types: Vec<Vec<Edge>> = ball_census(g, r)
        .iter()
        .map(|e| e.ball.edges().to_vec())
        .collect();
    compile(r, &types).ruleset.expect("a census compiles")
}
