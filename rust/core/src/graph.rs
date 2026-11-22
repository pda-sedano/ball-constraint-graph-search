//! The graph the search works on: fixed-size adjacency rows, edited in place.
//!
//! Every node gets a row of `max_degree` slots in one flat array, plus a
//! degree count. Adding, removing or testing an edge touches one or two short
//! rows and never allocates, which matters because the sampler edits the
//! graph millions of times. For a graph checked against a ruleset,
//! `max_degree` is the ruleset's largest root degree: no node of a valid
//! graph can exceed it. A reference graph uses its own largest degree.

use std::fmt;

use crate::ball::{Edge, Node};

/// Why an edge could not be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphError {
    NodeOutOfRange {
        node: Node,
        num_nodes: usize,
    },
    SelfLoop {
        node: Node,
    },
    /// The node already has `max_degree` neighbors.
    DegreeCap {
        node: Node,
        max_degree: usize,
    },
    /// `Graph::from_edges` was given the same edge twice.
    RepeatedEdge {
        edge: Edge,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GraphError::NodeOutOfRange { node, num_nodes } => {
                write!(f, "node {node} is out of range for {num_nodes} nodes")
            }
            GraphError::SelfLoop { node } => write!(f, "self-loop on node {node}"),
            GraphError::DegreeCap { node, max_degree } => {
                write!(f, "node {node} already has the maximum degree {max_degree}")
            }
            GraphError::RepeatedEdge { edge: (u, v) } => write!(f, "edge [{u}, {v}] given twice"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A simple undirected graph on nodes `0..num_nodes` with a degree cap.
///
/// Neighbor lists are unordered: removing an edge moves the row's last entry
/// into the gap. Use [`Graph::edges`] for a canonical listing.
#[derive(Debug, Clone)]
pub struct Graph {
    max_degree: usize,
    /// `degrees[v]` is how many slots of node v's row are in use.
    degrees: Vec<u32>,
    /// Node v's neighbors are `slots[v * max_degree..v * max_degree + degrees[v]]`.
    slots: Vec<Node>,
    num_edges: usize,
}

impl Graph {
    /// A graph with no edges.
    pub fn new(num_nodes: usize, max_degree: usize) -> Self {
        Self {
            max_degree,
            degrees: vec![0; num_nodes],
            slots: vec![0; num_nodes * max_degree],
            num_edges: 0,
        }
    }

    /// A graph with these edges. Every edge must be addable, and none may
    /// repeat (in either orientation).
    pub fn from_edges(
        num_nodes: usize,
        max_degree: usize,
        edges: &[Edge],
    ) -> Result<Self, GraphError> {
        let mut graph = Self::new(num_nodes, max_degree);
        for &(u, v) in edges {
            if !graph.add_edge(u, v)? {
                return Err(GraphError::RepeatedEdge { edge: (u, v) });
            }
        }
        Ok(graph)
    }

    /// Like [`Graph::from_edges`], with the degree cap set to the largest
    /// degree the edges produce: for reference graphs, which have no ruleset.
    pub fn from_edges_with_own_cap(num_nodes: usize, edges: &[Edge]) -> Result<Self, GraphError> {
        let mut degrees = vec![0usize; num_nodes];
        for &(u, v) in edges {
            for node in [u, v] {
                if let Some(d) = degrees.get_mut(node as usize) {
                    *d += 1;
                }
            }
        }
        // Out-of-range nodes and self-loops are reported by from_edges.
        let max_degree = degrees.into_iter().max().unwrap_or(0);
        Self::from_edges(num_nodes, max_degree, edges)
    }

    pub fn num_nodes(&self) -> usize {
        self.degrees.len()
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Panics if `v` is out of range.
    pub fn degree(&self, v: Node) -> usize {
        self.degrees[v as usize] as usize
    }

    /// Node v's neighbors, in no particular order. Panics if `v` is out of range.
    pub fn neighbors(&self, v: Node) -> &[Node] {
        let start = v as usize * self.max_degree;
        &self.slots[start..start + self.degree(v)]
    }

    /// Whether u and v are adjacent. False for out-of-range nodes.
    pub fn has_edge(&self, u: Node, v: Node) -> bool {
        if u as usize >= self.num_nodes() || v as usize >= self.num_nodes() {
            return false;
        }
        // Scan the shorter row.
        let (a, b) = if self.degree(u) <= self.degree(v) {
            (u, v)
        } else {
            (v, u)
        };
        self.neighbors(a).contains(&b)
    }

    /// Add the edge u–v. Returns `Ok(false)` if it was already present, and
    /// an error, leaving the graph unchanged, if it cannot be added.
    pub fn add_edge(&mut self, u: Node, v: Node) -> Result<bool, GraphError> {
        for node in [u, v] {
            if node as usize >= self.num_nodes() {
                return Err(GraphError::NodeOutOfRange {
                    node,
                    num_nodes: self.num_nodes(),
                });
            }
        }
        if u == v {
            return Err(GraphError::SelfLoop { node: u });
        }
        if self.has_edge(u, v) {
            return Ok(false);
        }
        for node in [u, v] {
            if self.degree(node) == self.max_degree {
                return Err(GraphError::DegreeCap {
                    node,
                    max_degree: self.max_degree,
                });
            }
        }
        self.push_neighbor(u, v);
        self.push_neighbor(v, u);
        self.num_edges += 1;
        Ok(true)
    }

    /// Remove the edge u–v. Returns whether it was present.
    pub fn remove_edge(&mut self, u: Node, v: Node) -> bool {
        if !self.has_edge(u, v) {
            return false;
        }
        self.drop_neighbor(u, v);
        self.drop_neighbor(v, u);
        self.num_edges -= 1;
        true
    }

    /// Every edge once, as `(min, max)`, sorted.
    pub fn edges(&self) -> Vec<Edge> {
        let mut edges: Vec<Edge> = (0..self.num_nodes() as Node)
            .flat_map(|u| {
                self.neighbors(u)
                    .iter()
                    .filter(move |&&w| u < w)
                    .map(move |&w| (u, w))
            })
            .collect();
        edges.sort_unstable();
        edges
    }

    /// Each node's distance to the nearest of `sources`, by one breadth-first
    /// search from all of them at once. `u32::MAX` where no source reaches,
    /// so with no sources every entry is `u32::MAX`.
    ///
    /// The validity checker uses it with the open set as sources: 0 is open,
    /// 1 to r - 1 shadow, r or more core.
    pub fn distances_from(&self, sources: &[Node]) -> Vec<u32> {
        let mut distance = vec![u32::MAX; self.num_nodes()];
        let mut queue = std::collections::VecDeque::with_capacity(sources.len());
        for &s in sources {
            if distance[s as usize] != 0 {
                distance[s as usize] = 0;
                queue.push_back(s);
            }
        }
        while let Some(v) = queue.pop_front() {
            let next = distance[v as usize] + 1;
            for &w in self.neighbors(v) {
                if distance[w as usize] == u32::MAX {
                    distance[w as usize] = next;
                    queue.push_back(w);
                }
            }
        }
        distance
    }

    fn push_neighbor(&mut self, v: Node, w: Node) {
        let degree = self.degree(v);
        self.slots[v as usize * self.max_degree + degree] = w;
        self.degrees[v as usize] += 1;
    }

    fn drop_neighbor(&mut self, v: Node, w: Node) {
        let start = v as usize * self.max_degree;
        let last = start + self.degree(v) - 1;
        let position = start
            + self
                .neighbors(v)
                .iter()
                .position(|&x| x == w)
                .expect("caller checked the edge exists");
        self.slots[position] = self.slots[last];
        self.degrees[v as usize] -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_from_several_sources() {
        //! A path 0-1-2-3-4-5 with sources at both ends, and a separate node 6.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let g = Graph::from_edges(7, 2, &edges).expect("valid");
        assert_eq!(g.distances_from(&[0, 5]), [0, 1, 2, 2, 1, 0, u32::MAX]);
        assert_eq!(g.distances_from(&[]), [u32::MAX; 7]);
        assert_eq!(g.distances_from(&[3, 3]), [3, 2, 1, 0, 1, 2, u32::MAX]);
    }

    #[test]
    fn add_remove_and_query() {
        let mut g = Graph::new(4, 2);
        assert_eq!(g.add_edge(0, 1), Ok(true));
        assert_eq!(g.add_edge(1, 2), Ok(true));
        assert_eq!(
            g.add_edge(2, 1),
            Ok(false),
            "already present, either orientation"
        );
        assert!(g.has_edge(1, 0) && g.has_edge(2, 1) && !g.has_edge(0, 2));
        assert_eq!((g.num_edges(), g.degree(1)), (2, 2));

        assert!(g.remove_edge(1, 0));
        assert!(!g.remove_edge(1, 0), "already gone");
        assert_eq!(g.edges(), [(1, 2)]);
        assert_eq!((g.num_edges(), g.degree(0), g.degree(1)), (1, 0, 1));
    }

    #[test]
    fn errors_leave_the_graph_unchanged() {
        let mut g = Graph::from_edges(3, 1, &[(0, 1)]).expect("valid");
        assert_eq!(
            g.add_edge(0, 3),
            Err(GraphError::NodeOutOfRange {
                node: 3,
                num_nodes: 3
            })
        );
        assert_eq!(g.add_edge(2, 2), Err(GraphError::SelfLoop { node: 2 }));
        assert_eq!(
            g.add_edge(1, 2),
            Err(GraphError::DegreeCap {
                node: 1,
                max_degree: 1
            })
        );
        assert_eq!(
            g.add_edge(0, 1),
            Ok(false),
            "present edges never hit the cap"
        );
        assert_eq!((g.edges(), g.num_edges()), (vec![(0, 1)], 1));
    }

    #[test]
    fn from_edges_rejects_repeats() {
        assert_eq!(
            Graph::from_edges(3, 2, &[(0, 1), (1, 0)]).unwrap_err(),
            GraphError::RepeatedEdge { edge: (1, 0) }
        );
    }

    #[test]
    fn own_cap_is_the_largest_degree() {
        let g =
            Graph::from_edges_with_own_cap(5, &[(0, 1), (0, 2), (0, 3), (3, 4)]).expect("valid");
        assert_eq!(g.max_degree(), 3);
        assert_eq!(
            Graph::from_edges_with_own_cap(2, &[])
                .expect("valid")
                .max_degree(),
            0
        );
        assert!(matches!(
            Graph::from_edges_with_own_cap(2, &[(0, 5)]),
            Err(GraphError::NodeOutOfRange { node: 5, .. })
        ));
    }

    #[test]
    fn random_edits_match_a_simple_model() {
        //! Thousands of random adds and removes against a set of edges.
        let (n, cap) = (12, 3);
        let mut g = Graph::new(n, cap);
        let mut model = std::collections::BTreeSet::new();
        let mut state = 0x9e3779b97f4a7c15u64;
        for _ in 0..5000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let (u, v) = (
                (state % n as u64) as Node,
                ((state >> 20) % n as u64) as Node,
            );
            if u == v {
                continue;
            }
            let e = (u.min(v), u.max(v));
            if state & (1 << 40) == 0 {
                let degree = |x: Node| model.iter().filter(|&&(a, b)| a == x || b == x).count();
                match g.add_edge(u, v) {
                    Ok(added) => assert_eq!(added, model.insert(e)),
                    Err(GraphError::DegreeCap { .. }) => {
                        assert!(degree(u) == cap || degree(v) == cap)
                    }
                    Err(other) => panic!("unexpected {other}"),
                }
            } else {
                assert_eq!(g.remove_edge(u, v), model.remove(&e));
            }
            assert_eq!(g.num_edges(), model.len());
        }
        assert_eq!(g.edges(), model.into_iter().collect::<Vec<_>>());
    }
}
