//! One ball type: structural checks on its edge list, and the compiled form.
//!
//! Edges arrive exactly as written in the file (Python checked only that they
//! are pairs of integers from 0 to 2^32 - 1). `check_structure` reports every
//! problem with the graph; edges that pass become a [`CompiledBall`], whose
//! invariants every later step can rely on.

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::canonical::{canonical_form, CanonicalForm, CanonicalKey};
use crate::issue::{Issue, IssueCode};

/// A node id. Balls are small, so 32 bits is plenty.
pub type Node = u32;

/// An edge as written: order and orientation are the file's.
pub type Edge = (Node, Node);

/// How many node ids an issue message lists before summarizing the rest.
const MAX_LISTED: usize = 8;

/// Structural checks on one type's edge list. Reports every problem rather
/// than stopping at the first, in this order:
///
/// - No self-loops (`SelfLoop`), one issue per offending edge.
/// - No edge appears twice after normalizing to `(min, max)`
///   (`RepeatedEdge`), reported at the later position.
/// - Nodes are exactly `0..k` (`NodeGap`). A missing node 0 is a gap like
///   any other. An empty list means the single-node ball {0}, which passes.
/// - Every node is reachable from the root (`Disconnected`). Always runs,
///   gap or not. Without node 0, every node is unreachable, so a ball type
///   written with no root gets both `NodeGap` and `Disconnected`.
///
/// Messages name edges by their position in `edges`, so they match the file.
/// Every issue carries `type_index`.
pub fn check_structure(edges: &[Edge], type_index: usize) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut report = |code: IssueCode, message: String| {
        issues.push(Issue::new(code, message, Some(type_index)));
    };

    for (i, &(u, v)) in edges.iter().enumerate() {
        if u == v {
            report(
                IssueCode::SelfLoop,
                format!("edge {i} [{u}, {v}] is a self-loop"),
            );
        }
    }

    let mut first_position: HashMap<Edge, usize> = HashMap::new();
    for (i, &(u, v)) in edges.iter().enumerate() {
        if let Some(&first) = first_position.get(&normalize(u, v)) {
            report(
                IssueCode::RepeatedEdge,
                format!("edge {i} [{u}, {v}] repeats edge {first}"),
            );
        } else {
            first_position.insert(normalize(u, v), i);
        }
    }

    if edges.is_empty() {
        return issues; // the single-node ball {0}
    }

    let nodes: BTreeSet<Node> = edges.iter().flat_map(|&(u, v)| [u, v]).collect();
    let missing = missing_ids(&nodes);
    if !missing.is_empty() {
        report(
            IssueCode::NodeGap,
            format!(
                "nodes must be 0 to {} with none missing; missing {}",
                nodes.len() - 1,
                preview(&missing, missing_count(&nodes)),
            ),
        );
    }

    let unreached = unreachable_from_root(edges, &nodes);
    if !unreached.is_empty() {
        let message = if nodes.contains(&0) {
            format!(
                "not reachable from the root: {}",
                preview(&unreached, unreached.len())
            )
        } else {
            format!(
                "no root (node 0), so none of the {} nodes is reachable",
                nodes.len()
            )
        };
        report(IssueCode::Disconnected, message);
    }

    issues
}

/// The edge as `(min, max)`.
fn normalize(u: Node, v: Node) -> Edge {
    (u.min(v), u.max(v))
}

/// Up to `MAX_LISTED` of the ids in `0..=max` that `nodes` lacks, smallest
/// first. Walks the sorted ids, so a huge id costs nothing extra.
fn missing_ids(nodes: &BTreeSet<Node>) -> Vec<Node> {
    let mut missing = Vec::new();
    let mut expected: u64 = 0;
    for &node in nodes {
        while expected < u64::from(node) && missing.len() < MAX_LISTED {
            missing.push(expected as Node);
            expected += 1;
        }
        if missing.len() == MAX_LISTED {
            break;
        }
        expected = u64::from(node) + 1;
    }
    missing
}

/// How many ids in `0..=max` are missing: `max + 1 - len`.
fn missing_count(nodes: &BTreeSet<Node>) -> usize {
    let max = nodes.last().map_or(0, |&m| u64::from(m));
    (max + 1 - nodes.len() as u64) as usize
}

/// Sorted nodes the root cannot reach. Every node, if there is no node 0.
fn unreachable_from_root(edges: &[Edge], nodes: &BTreeSet<Node>) -> Vec<Node> {
    let mut adjacency: HashMap<Node, Vec<Node>> = HashMap::new();
    for &(u, v) in edges {
        adjacency.entry(u).or_default().push(v);
        adjacency.entry(v).or_default().push(u);
    }

    let mut reached: BTreeSet<Node> = BTreeSet::new();
    if nodes.contains(&0) {
        let mut queue = VecDeque::from([0]);
        reached.insert(0);
        while let Some(u) = queue.pop_front() {
            for &w in &adjacency[&u] {
                if reached.insert(w) {
                    queue.push_back(w);
                }
            }
        }
    }
    nodes.difference(&reached).copied().collect()
}

/// `shown` as a list, plus how many more there are out of `total`.
fn preview(shown: &[Node], total: usize) -> String {
    let listed: Vec<String> = shown.iter().take(MAX_LISTED).map(Node::to_string).collect();
    let more = total.saturating_sub(listed.len());
    if more > 0 {
        format!("[{}, ...] ({more} more)", listed.join(", "))
    } else {
        format!("[{}]", listed.join(", "))
    }
}

/// A ball type that passed [`check_structure`].
///
/// Invariants: nodes are `0..num_nodes`, the root is 0, the graph is
/// connected, and there are no self-loops or repeated edges. Depth against r
/// is not an invariant; `compile` checks that separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledBall {
    /// `adjacency[v]` is the sorted list of v's neighbors.
    adjacency: Vec<Vec<Node>>,
    /// `layers[d]` is the sorted list of nodes at distance d from the root.
    layers: Vec<Vec<Node>>,
    /// Normalized `(min, max)` edges, sorted. Equal lists mean exact duplicates.
    edges: Vec<Edge>,
    /// Canonical key and numbering. Equal keys mean isomorphic balls.
    canonical: CanonicalForm,
}

impl CompiledBall {
    /// Build from an edge list that `check_structure` accepted: normalize and
    /// sort the edges, build adjacency, BFS from the root for the layers, and
    /// compute the canonical form.
    ///
    /// Callers must run `check_structure` first, or use [`Self::from_edges`].
    /// Debug builds assert it, so a missed check fails loudly in tests.
    pub fn from_checked_edges(edges: &[Edge]) -> Self {
        debug_assert!(
            check_structure(edges, 0).is_empty(),
            "from_checked_edges called on edges that fail check_structure: {edges:?}"
        );

        let mut normalized: Vec<Edge> = edges.iter().map(|&(u, v)| normalize(u, v)).collect();
        normalized.sort_unstable();

        let num_nodes = normalized
            .iter()
            .map(|&(_, v)| v as usize + 1)
            .max()
            .unwrap_or(1);
        let mut adjacency = vec![Vec::new(); num_nodes];
        for &(u, v) in &normalized {
            adjacency[u as usize].push(v);
            adjacency[v as usize].push(u);
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }

        let layers = bfs_layers(&adjacency);
        let canonical = canonical_form(&adjacency, &layers);
        Self {
            adjacency,
            layers,
            edges: normalized,
            canonical,
        }
    }

    /// Run `check_structure`, then build: the ball, or every issue found.
    /// Issues carry type index 0.
    pub fn from_edges(edges: &[Edge]) -> Result<Self, Vec<Issue>> {
        let issues = check_structure(edges, 0);
        if issues.is_empty() {
            Ok(Self::from_checked_edges(edges))
        } else {
            Err(issues)
        }
    }

    /// Number of nodes, at least 1.
    pub fn num_nodes(&self) -> usize {
        self.adjacency.len()
    }

    /// Sorted neighbors of `v`. Panics if `v` is out of range.
    pub fn neighbors(&self, v: Node) -> &[Node] {
        &self.adjacency[v as usize]
    }

    pub fn degree(&self, v: Node) -> usize {
        self.neighbors(v).len()
    }

    /// `degrees()[v]` is the degree of node v.
    pub fn degrees(&self) -> Vec<usize> {
        self.adjacency.iter().map(Vec::len).collect()
    }

    /// Nodes grouped by distance from the root.
    pub fn layers(&self) -> &[Vec<Node>] {
        &self.layers
    }

    /// Largest distance from the root. 0 for the single-node ball.
    pub fn depth(&self) -> usize {
        self.layers.len() - 1
    }

    pub fn root_degree(&self) -> usize {
        self.degree(0)
    }

    /// Number of nodes at each distance. Part of the canonical key.
    pub fn layer_sizes(&self) -> Vec<usize> {
        self.layers.iter().map(Vec::len).collect()
    }

    /// Normalized, sorted edges.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Equal for two balls exactly when they are isomorphic with root
    /// mapped to root.
    pub fn canonical_key(&self) -> &CanonicalKey {
        &self.canonical.key
    }

    /// `canonical_labeling()[v]` is node v's number in the canonical form.
    pub fn canonical_labeling(&self) -> &[Node] {
        &self.canonical.labeling
    }
}

/// BFS layers from node 0 of a connected graph given as adjacency lists.
fn bfs_layers(adjacency: &[Vec<Node>]) -> Vec<Vec<Node>> {
    let mut seen = vec![false; adjacency.len()];
    seen[0] = true;
    let mut layers = vec![vec![0]];
    loop {
        let mut next: Vec<Node> = Vec::new();
        for &u in layers.last().expect("layers starts non-empty") {
            for &w in &adjacency[u as usize] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_unstable();
        layers.push(next);
    }
    debug_assert!(seen.iter().all(|&s| s), "graph is not connected");
    layers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(edges: &[Edge]) -> Vec<IssueCode> {
        check_structure(edges, 0)
            .into_iter()
            .map(|issue| issue.code)
            .collect()
    }

    const STAR: &[Edge] = &[(0, 1), (1, 2), (1, 3)];

    // check_structure: one test per code, mirroring the old Python cases.

    #[test]
    fn self_loop() {
        assert_eq!(codes(&[(0, 1), (1, 1)]), [IssueCode::SelfLoop]);
    }

    #[test]
    fn repeated_edge_including_reversed() {
        for edges in [[(0, 1), (0, 1)], [(0, 1), (1, 0)]] {
            let issues = check_structure(&edges, 0);
            assert_eq!(issues.len(), 1, "{issues:?}");
            assert_eq!(issues[0].code, IssueCode::RepeatedEdge);
            assert!(
                issues[0].message.starts_with("edge 1 "),
                "{}",
                issues[0].message
            );
        }
    }

    #[test]
    fn node_gap() {
        assert_eq!(codes(&[(0, 1), (1, 3)]), [IssueCode::NodeGap]);
    }

    #[test]
    fn disconnected() {
        assert_eq!(codes(&[(0, 1), (2, 3)]), [IssueCode::Disconnected]);
    }

    #[test]
    fn missing_root_is_a_gap_and_disconnected() {
        for edges in [&[(1, 2)][..], &[(2, 3), (3, 4)][..]] {
            assert_eq!(codes(edges), [IssueCode::NodeGap, IssueCode::Disconnected]);
        }
    }

    #[test]
    fn reports_all_problems_with_type_index() {
        let issues = check_structure(&[(0, 1), (1, 1), (1, 0)], 7);
        let found: Vec<_> = issues.iter().map(|i| (i.code, i.type_index)).collect();
        assert_eq!(
            found,
            [
                (IssueCode::SelfLoop, Some(7)),
                (IssueCode::RepeatedEdge, Some(7))
            ]
        );
    }

    #[test]
    fn accepts_valid() {
        for edges in [&[][..], &[(0, 1)][..], STAR] {
            assert_eq!(check_structure(edges, 0), [], "{edges:?}");
        }
    }

    #[test]
    fn huge_node_id_is_a_gap_without_enumerating() {
        //! The largest u32 is a gap of about 4 billion ids; the message lists a few.
        let issues = check_structure(&[(0, Node::MAX)], 0);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, IssueCode::NodeGap);
        assert!(
            issues[0].message.contains("[1, 2, 3, 4, 5, 6, 7, 8, ...]"),
            "{}",
            issues[0].message
        );
    }

    // CompiledBall

    #[test]
    fn star_layers_and_degrees() {
        let ball = CompiledBall::from_checked_edges(STAR);
        assert_eq!(ball.layers(), [vec![0], vec![1], vec![2, 3]]);
        assert_eq!(ball.degrees(), [1, 3, 1, 1]);
        assert_eq!(ball.depth(), 2);
        assert_eq!(ball.layer_sizes(), [1, 1, 2]);
        assert_eq!(ball.root_degree(), 1);
        assert_eq!(ball.neighbors(1), [0, 2, 3]);
    }

    #[test]
    fn single_node_ball() {
        let ball = CompiledBall::from_checked_edges(&[]);
        assert_eq!(ball.num_nodes(), 1);
        assert_eq!(ball.layers(), [vec![0]]);
        assert_eq!(ball.depth(), 0);
        assert_eq!(ball.root_degree(), 0);
    }

    #[test]
    fn edges_are_normalized_and_sorted() {
        let ball = CompiledBall::from_checked_edges(&[(2, 1), (1, 0)]);
        assert_eq!(ball.edges(), [(0, 1), (1, 2)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "fail check_structure")]
    fn unchecked_edges_panic_in_debug() {
        CompiledBall::from_checked_edges(&[(0, 1), (2, 3)]);
    }
}
