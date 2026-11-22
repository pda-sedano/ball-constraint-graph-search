//! Canonical form for rooted balls: ball in, key out.
//!
//! Two balls get the same [`CanonicalKey`] exactly when they are isomorphic by
//! a map that sends root to root. That turns matching a ball against a ruleset
//! into a hash lookup.
//!
//! # Method
//!
//! nauty's `densenauty` (through the `nauty-Traces-sys` bindings) computes a
//! canonical labeling of the ball, given an initial coloring with one color
//! per distance from the root. Any root-preserving isomorphism preserves
//! distances, so the coloring is invariant, and since the root is alone in
//! the first color, nauty can only map it to itself. The colors are ordered
//! by depth, so the canonical numbering puts the root first and numbers the
//! nodes layer by layer.
//!
//! The key is the layer sizes plus the edge list under that numbering. The
//! layer sizes are needed because a labeled graph alone does not say which
//! colors its nodes had.
//!
//! # Whole graphs
//!
//! [`canonical_graph`] and [`automorphism_count`] work on a whole [`Graph`],
//! for the sampler and the enumeration baseline. They use nauty's sparse mode
//! (`sparsenauty`), whose memory grows with the number of edges. The dense
//! mode used for balls stores an n-by-n adjacency matrix, which is the faster
//! choice for balls of a few dozen nodes but would need about 125 GB for a
//! million-node sample.

use std::os::raw::c_int;

use nauty_Traces_sys::{
    densenauty, empty_graph, optionblk, sparsegraph, sparsenauty, statsblk, SparseGraph,
    ADDONEEDGE, FALSE, SETWORDSNEEDED, TRUE,
};

use crate::ball::{Edge, Node};
use crate::graph::Graph;
use crate::graph6::write_graph6;

/// A canonical key: equal keys mean isomorphic rooted balls.
///
/// Ordered and hashable, so keys can go in maps and sets. The order has no
/// meaning beyond being consistent. Not guaranteed stable across nauty
/// versions: recompute on load, never save (as the design doc requires).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CanonicalKey {
    layer_sizes: Vec<usize>,
    edges: Vec<Edge>,
}

impl CanonicalKey {
    /// Number of nodes at each distance from the root.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// The edges under the canonical numbering, normalized `(min, max)` and
    /// sorted. The root is 0 and nodes are numbered layer by layer.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// A canonical key plus the numbering that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalForm {
    pub key: CanonicalKey,
    /// `labeling[v]` is node v's canonical number. Relabeling the ball's
    /// edges by it gives `key.edges()`.
    pub labeling: Vec<Node>,
}

/// A whole graph under nauty's canonical numbering, with its automorphism
/// count. Two graphs have equal `edges` (and `num_nodes`) exactly when they are
/// isomorphic.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalGraph {
    pub num_nodes: usize,
    /// Edges under the canonical numbering, normalized `(min, max)` and sorted.
    pub edges: Vec<Edge>,
    /// |Aut|, as in [`automorphism_count`].
    pub automorphisms: f64,
}

impl CanonicalGraph {
    /// The canonical edges as one graph6 line: a short, hashable key for an
    /// unlabelled graph. graph6 stores every node pair, so it takes about
    /// n^2/12 bytes; meant for the small graphs whose visits the sampler
    /// counts, not for large samples.
    pub fn graph6(&self) -> String {
        write_graph6(self.num_nodes, &self.edges)
    }
}

/// `graph` under nauty's canonical numbering, with |Aut|.
pub fn canonical_graph(graph: &Graph) -> CanonicalGraph {
    let (lab, automorphisms) = sparse_nauty(graph, true);
    let mut labeling: Vec<Node> = vec![0; lab.len()];
    for (number, &v) in lab.iter().enumerate() {
        labeling[v as usize] = number as Node;
    }
    let mut edges: Vec<Edge> = graph
        .edges()
        .into_iter()
        .map(|(u, v)| {
            let (a, b) = (labeling[u as usize], labeling[v as usize]);
            (a.min(b), a.max(b))
        })
        .collect();
    edges.sort_unstable();
    CanonicalGraph {
        num_nodes: graph.num_nodes(),
        edges,
        automorphisms,
    }
}

/// |Aut| of a whole graph: how many permutations of its nodes preserve every
/// edge, from nauty's sparse mode. nauty reports it as a double, so it is
/// exact below 2^53; past that it is the nearest double. 1 for the empty graph.
pub fn automorphism_count(graph: &Graph) -> f64 {
    sparse_nauty(graph, false).1
}

/// Run `sparsenauty` on the whole graph with every node in one cell. Returns
/// `lab` (with `canonical`, `lab[i]` is the node numbered i) and |Aut|.
fn sparse_nauty(graph: &Graph, canonical: bool) -> (Vec<c_int>, f64) {
    let n = graph.num_nodes();
    if n == 0 {
        return (Vec::new(), 1.0);
    }
    let arcs = 2 * graph.num_edges();
    let mut input = SparseGraph::new(n, arcs);
    let mut position = 0;
    for v in 0..n {
        let neighbors = graph.neighbors(v as Node);
        input.v[v] = position;
        input.d[v] = neighbors.len() as c_int;
        for &w in neighbors {
            input.e[position] = w as c_int;
            position += 1;
        }
    }
    // nauty writes the canonical graph here. It reallocates only buffers
    // that are too small, and these are exactly big enough, so it never
    // frees or replaces memory Rust owns.
    let mut output = SparseGraph::new(n, arcs);
    let mut output_raw: sparsegraph = (&mut output).into();

    let mut lab: Vec<c_int> = vec![0; n];
    let mut ptn: Vec<c_int> = vec![0; n];
    let mut orbits: Vec<c_int> = vec![0; n];
    let mut options = optionblk {
        getcanon: if canonical { TRUE } else { FALSE },
        defaultptn: TRUE, // one cell: every node may go anywhere
        ..optionblk::default_sparse()
    };
    let mut stats = statsblk::default();
    // SAFETY: `input` is a valid sparse graph: n entries in v and d, and the
    // neighbors of v at e[v[v]..v[v] + d[v]], within e's 2m entries. `lab`,
    // `ptn` and `orbits` hold n ints, and `output_raw` points at buffers of
    // the same sizes as `input`'s, which is all nauty needs for the canonical
    // graph. nauty uses thread-local workspace, so concurrent calls are safe.
    unsafe {
        sparsenauty(
            &mut (&mut input).into(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            if canonical {
                &mut output_raw
            } else {
                std::ptr::null_mut()
            },
        );
    }
    assert_eq!(
        stats.errstatus, 0,
        "nauty reported error {}",
        stats.errstatus
    );
    debug_assert!(
        output_raw.v == output.v.as_mut_ptr() && output_raw.e == output.e.as_mut_ptr(),
        "nauty reallocated the canonical graph's buffers"
    );
    (lab, stats.grpsize1 * 10f64.powi(stats.grpsize2))
}

/// The canonical form of a connected rooted graph, root 0.
///
/// `adjacency[v]` lists v's neighbors and `layers[d]` the nodes at distance d
/// from the root, as stored by `CompiledBall`.
pub(crate) fn canonical_form(adjacency: &[Vec<Node>], layers: &[Vec<Node>]) -> CanonicalForm {
    let n = adjacency.len();
    let m = SETWORDSNEEDED(n);

    let mut graph = empty_graph(m, n);
    for (u, neighbors) in adjacency.iter().enumerate() {
        for &w in neighbors {
            if u < w as usize {
                ADDONEEDGE(&mut graph, u, w as usize, m);
            }
        }
    }

    // The initial coloring, as nauty takes it: `lab` lists the nodes cell by
    // cell, and `ptn` is 0 at the last node of each cell, 1 elsewhere. One
    // cell per layer, in depth order. On return, `lab` is the canonical
    // order: `lab[i]` is the node numbered i.
    let mut lab: Vec<c_int> = Vec::with_capacity(n);
    let mut ptn: Vec<c_int> = Vec::with_capacity(n);
    for layer in layers {
        for (i, &v) in layer.iter().enumerate() {
            lab.push(v as c_int);
            ptn.push(if i + 1 == layer.len() { 0 } else { 1 });
        }
    }
    debug_assert_eq!(lab.len(), n, "layers must cover every node once");

    let mut orbits: Vec<c_int> = vec![0; n];
    let mut canonical_graph = empty_graph(m, n);
    let mut options = optionblk {
        getcanon: TRUE,
        defaultptn: 0, // use lab/ptn as given
        ..optionblk::default()
    };
    let mut stats = statsblk::default();

    // SAFETY: every buffer has the length nauty expects for n nodes and m
    // setwords per row: `graph` and `canonical_graph` hold n * m setwords,
    // `lab`, `ptn` and `orbits` hold n ints. nauty only reads and writes
    // within them. The bindings build nauty with thread-local storage, so
    // concurrent calls are safe.
    unsafe {
        densenauty(
            graph.as_mut_ptr(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut options,
            &mut stats,
            m as c_int,
            n as c_int,
            canonical_graph.as_mut_ptr(),
        );
    }

    let mut labeling: Vec<Node> = vec![0; n];
    for (number, &v) in lab.iter().enumerate() {
        labeling[v as usize] = number as Node;
    }

    let mut edges: Vec<Edge> = Vec::new();
    for (u, neighbors) in adjacency.iter().enumerate() {
        for &w in neighbors {
            if u < w as usize {
                let (a, b) = (labeling[u], labeling[w as usize]);
                edges.push((a.min(b), a.max(b)));
            }
        }
    }
    edges.sort_unstable();

    CanonicalForm {
        key: CanonicalKey {
            layer_sizes: layers.iter().map(Vec::len).collect(),
            edges,
        },
        labeling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::{check_structure, CompiledBall};
    use std::collections::HashMap;

    fn ball(edges: &[Edge]) -> CompiledBall {
        CompiledBall::from_checked_edges(edges)
    }

    fn key(edges: &[Edge]) -> CanonicalKey {
        ball(edges).canonical_key().clone()
    }

    /// Small deterministic generator, so tests need no extra crates.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Renumber nodes by a random permutation that keeps the root at 0.
    fn relabel(edges: &[Edge], n: usize, rng: &mut Rng) -> Vec<Edge> {
        let mut perm: Vec<Node> = (0..n as Node).collect();
        for i in (2..n).rev() {
            let j = 1 + rng.below(i);
            perm.swap(i, j);
        }
        let mut out: Vec<Edge> = edges
            .iter()
            .map(|&(u, v)| (perm[u as usize], perm[v as usize]))
            .collect();
        // Shuffle edge order and orientation too: neither should matter.
        for i in (1..out.len()).rev() {
            out.swap(i, rng.below(i + 1));
        }
        for e in &mut out {
            if rng.below(2) == 1 {
                *e = (e.1, e.0);
            }
        }
        out
    }

    /// A random connected rooted graph: a random tree on `0..n` plus `extra`
    /// random extra edges.
    fn random_ball(n: usize, extra: usize, rng: &mut Rng) -> Vec<Edge> {
        let mut edges: Vec<Edge> = (1..n).map(|v| (rng.below(v) as Node, v as Node)).collect();
        for _ in 0..extra {
            let (u, v) = (rng.below(n) as Node, rng.below(n) as Node);
            let e = (u.min(v), u.max(v));
            if u != v && !edges.iter().any(|&(a, b)| (a.min(b), a.max(b)) == e) {
                edges.push(e);
            }
        }
        edges
    }

    /// Reference canonical form by brute force: the smallest relabeled edge
    /// list over every permutation fixing the root. Only for tiny graphs.
    fn brute_key(edges: &[Edge], n: usize) -> Vec<Edge> {
        let mut rest: Vec<Node> = (1..n as Node).collect();
        let mut best: Option<Vec<Edge>> = None;
        permutations(&mut rest, 0, &mut |perm| {
            let map = |v: Node| if v == 0 { 0 } else { perm[v as usize - 1] };
            let mut relabeled: Vec<Edge> = edges
                .iter()
                .map(|&(u, v)| {
                    let (a, b) = (map(u), map(v));
                    (a.min(b), a.max(b))
                })
                .collect();
            relabeled.sort_unstable();
            if best.as_ref().is_none_or(|b| relabeled < *b) {
                best = Some(relabeled);
            }
        });
        best.unwrap_or_default()
    }

    fn permutations(items: &mut [Node], k: usize, visit: &mut impl FnMut(&[Node])) {
        if k == items.len() {
            visit(items);
            return;
        }
        for i in k..items.len() {
            items.swap(k, i);
            permutations(items, k + 1, visit);
            items.swap(k, i);
        }
    }

    #[test]
    fn labeling_reproduces_key() {
        let mut rng = Rng(7);
        for _ in 0..200 {
            let n = 2 + rng.below(20);
            let b = ball(&random_ball(n, rng.below(n), &mut rng));
            let labeling = b.canonical_labeling();
            let mut relabeled: Vec<Edge> = b
                .edges()
                .iter()
                .map(|&(u, v)| {
                    let (a, c) = (labeling[u as usize], labeling[v as usize]);
                    (a.min(c), a.max(c))
                })
                .collect();
            relabeled.sort_unstable();
            assert_eq!(relabeled, b.canonical_key().edges());
            assert_eq!(labeling[0], 0, "root keeps number 0");
        }
    }

    #[test]
    fn agrees_with_brute_force_on_every_small_rooted_graph() {
        //! Every connected graph on up to 6 nodes, rooted at node 0. Keys must
        //! partition them exactly as brute force does. The class counts are
        //! the numbers of rooted connected unlabeled graphs (OEIS A126100).
        let expected_classes = [0, 1, 1, 3, 11, 58, 407];
        for (n, &classes) in expected_classes.iter().enumerate().skip(1) {
            let pairs: Vec<Edge> = (0..n as Node)
                .flat_map(|u| (u + 1..n as Node).map(move |v| (u, v)))
                .collect();
            let mut ours_to_brute: HashMap<CanonicalKey, Vec<Edge>> = HashMap::new();
            let mut brute_to_ours: HashMap<Vec<Edge>, CanonicalKey> = HashMap::new();
            for mask in 0u32..(1 << pairs.len()) {
                let edges: Vec<Edge> = pairs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, &e)| e)
                    .collect();
                let uses_all_nodes = n == 1 || edges.iter().any(|&(_, v)| v as usize == n - 1);
                if !uses_all_nodes || !check_structure(&edges, 0).is_empty() {
                    continue; // counted at a smaller n, or not a valid ball
                }
                let ours = key(&edges);
                let brute = brute_key(&edges, n);
                assert_eq!(
                    ours_to_brute
                        .entry(ours.clone())
                        .or_insert_with(|| brute.clone()),
                    &brute,
                    "two non-isomorphic graphs share a key: {edges:?}"
                );
                assert_eq!(
                    brute_to_ours.entry(brute).or_insert_with(|| ours.clone()),
                    &ours,
                    "two isomorphic graphs got different keys: {edges:?}"
                );
            }
            assert_eq!(ours_to_brute.len(), classes, "n = {n}");
        }
    }

    #[test]
    fn invariant_under_relabeling() {
        let mut rng = Rng(42);
        for _ in 0..300 {
            let n = 2 + rng.below(40);
            let edges = random_ball(n, rng.below(2 * n), &mut rng);
            let original = key(&edges);
            for _ in 0..3 {
                assert_eq!(key(&relabel(&edges, n, &mut rng)), original, "{edges:?}");
            }
        }
    }

    #[test]
    fn root_matters() {
        //! A path rooted at an end and the same path rooted in the middle differ.
        assert_ne!(key(&[(0, 1), (1, 2)]), key(&[(0, 1), (0, 2)]));
    }

    #[test]
    fn separates_balls_refinement_alone_cannot() {
        //! Root joined to a 6-cycle versus root joined to two triangles: every
        //! non-root node has degree 3 in both, so color refinement never
        //! splits them. Only nauty's search can tell them apart.
        let hexagon: Vec<Edge> = (1..=6)
            .map(|v| (0, v))
            .chain((1..=6).map(|v| (v, v % 6 + 1)))
            .collect();
        let triangles: Vec<Edge> = (1..=6)
            .map(|v| (0, v))
            .chain([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)])
            .collect();
        assert_ne!(key(&hexagon), key(&triangles));
        let mut rng = Rng(3);
        assert_eq!(key(&relabel(&hexagon, 7, &mut rng)), key(&hexagon));
        assert_eq!(key(&relabel(&triangles, 7, &mut rng)), key(&triangles));
    }

    /// A complete tree: the root has `degree` children, every other internal
    /// node `degree - 1`, down to `depth`. The largest ball a degree bound
    /// allows, and the most symmetric.
    fn complete_tree(degree: usize, depth: usize) -> (Vec<Edge>, usize) {
        let mut edges = Vec::new();
        let mut frontier = vec![0 as Node];
        let mut next_id: Node = 1;
        for level in 0..depth {
            let children = if level == 0 { degree } else { degree - 1 };
            let mut next = Vec::new();
            for &parent in &frontier {
                for _ in 0..children {
                    edges.push((parent, next_id));
                    next.push(next_id);
                    next_id += 1;
                }
            }
            frontier = next;
        }
        (edges, next_id as usize)
    }

    #[test]
    fn highly_symmetric_balls_finish_and_stay_invariant() {
        //! Each of these has an automorphism group far too large to search
        //! leaf by leaf, the hard case for any canonical labeling.
        let mut cases: Vec<(&str, Vec<Edge>, usize)> = Vec::new();

        let (tree, n) = complete_tree(4, 3); // 53 nodes, |Aut| = 4! * 3!^4 * 3!^12
        cases.push(("degree-4 tree, depth 3", tree, n));
        let (tree, n) = complete_tree(3, 5); // 94 nodes
        cases.push(("degree-3 tree, depth 5", tree, n));

        // Root joined to every node of 8 disjoint triangles: refinement does
        // nothing, |Aut| = 8! * 6^8.
        let triangles: Vec<Edge> = (0..8u32)
            .flat_map(|t| {
                let (a, b, c) = (3 * t + 1, 3 * t + 2, 3 * t + 3);
                [(0, a), (0, b), (0, c), (a, b), (b, c), (c, a)]
            })
            .collect();
        cases.push(("root + 8 triangles", triangles, 25));

        // Radius-2 ball of the 6-dimensional hypercube around vertex 0.
        let mut cube_ids: HashMap<u32, Node> = HashMap::from([(0, 0)]);
        let mut cube_edges = Vec::new();
        let mut words: Vec<u32> = (0..64u32).filter(|w| w.count_ones() <= 2).collect();
        words.sort_by_key(|w| w.count_ones());
        for &w in &words {
            let id = cube_ids.len() as Node;
            cube_ids.entry(w).or_insert(id);
        }
        for &w in &words {
            for bit in 0..6 {
                let x = w ^ (1 << bit);
                if x.count_ones() <= 2 && w < x {
                    cube_edges.push((cube_ids[&w], cube_ids[&x]));
                }
            }
        }
        cases.push(("hypercube Q6, radius 2", cube_edges, words.len()));

        let mut rng = Rng(11);
        for (name, edges, n) in cases {
            let original = key(&edges);
            for _ in 0..3 {
                assert_eq!(key(&relabel(&edges, n, &mut rng)), original, "{name}");
            }
        }
    }

    #[test]
    fn automorphism_counts_of_known_graphs() {
        use crate::graph::Graph;
        let graph =
            |n: usize, edges: &[Edge]| Graph::from_edges_with_own_cap(n, edges).expect("valid");
        let cycle =
            |n: usize| -> Vec<Edge> { (0..n as Node).map(|v| (v, (v + 1) % n as Node)).collect() };
        for n in 3..12 {
            assert_eq!(
                automorphism_count(&graph(n, &cycle(n))),
                2.0 * n as f64,
                "cycle {n}"
            );
        }
        let path: Vec<Edge> = (1..6).map(|v| (v - 1, v)).collect();
        assert_eq!(automorphism_count(&graph(6, &path)), 2.0);
        assert_eq!(
            automorphism_count(&graph(4, &[(0, 1), (0, 2), (0, 3)])),
            6.0
        );
        let petersen: Vec<Edge> = (0..5)
            .flat_map(|i| [(i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)])
            .collect();
        assert_eq!(automorphism_count(&graph(10, &petersen)), 120.0);
        assert_eq!(automorphism_count(&graph(5, &[])), 120.0);
        assert_eq!(automorphism_count(&Graph::new(0, 0)), 1.0);
        // Two disjoint triangles plus an isolated node: (3!)^2 * 2.
        let triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        assert_eq!(automorphism_count(&graph(7, &triangles)), 72.0);
    }

    #[test]
    fn automorphism_count_scales_to_large_sparse_graphs() {
        //! A 100,000-node cycle. The dense mode would need a 1.25 GB matrix.
        use crate::graph::Graph;
        let n = 100_000usize;
        let cycle: Vec<Edge> = (0..n as Node).map(|v| (v, (v + 1) % n as Node)).collect();
        let g = Graph::from_edges(n, 2, &cycle).expect("valid");
        assert_eq!(automorphism_count(&g), 2.0 * n as f64);
    }

    /// Every graph on n nodes (connected or not), by bitmask over node pairs.
    fn all_graphs(n: usize) -> impl Iterator<Item = Vec<Edge>> {
        let pairs: Vec<Edge> = (0..n as Node)
            .flat_map(|u| (u + 1..n as Node).map(move |v| (u, v)))
            .collect();
        (0u32..1 << pairs.len()).map(move |mask| {
            pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &e)| e)
                .collect()
        })
    }

    /// The smallest relabeled edge list over every permutation: a brute-force
    /// canonical form for tiny graphs.
    fn brute_whole_key(edges: &[Edge], n: usize) -> Vec<Edge> {
        let mut nodes: Vec<Node> = (0..n as Node).collect();
        let mut best: Option<Vec<Edge>> = None;
        permutations(&mut nodes, 0, &mut |perm| {
            let mut relabeled: Vec<Edge> = edges
                .iter()
                .map(|&(u, v)| {
                    let (a, b) = (perm[u as usize], perm[v as usize]);
                    (a.min(b), a.max(b))
                })
                .collect();
            relabeled.sort_unstable();
            if best.as_ref().is_none_or(|b| relabeled < *b) {
                best = Some(relabeled);
            }
        });
        best.unwrap_or_default()
    }

    #[test]
    fn whole_graph_keys_agree_with_brute_force() {
        //! Every graph on up to 5 nodes: equal canonical edges exactly when
        //! brute force says isomorphic. The class counts are the numbers of
        //! unlabelled graphs (OEIS A000088); n = 6 checks the count only.
        use crate::graph::Graph;
        let expected_classes = [1, 1, 2, 4, 11, 34, 156];
        for (n, &classes) in expected_classes.iter().enumerate().skip(1) {
            let mut ours_to_brute: HashMap<Vec<Edge>, Vec<Edge>> = HashMap::new();
            let mut brute_to_ours: HashMap<Vec<Edge>, Vec<Edge>> = HashMap::new();
            for edges in all_graphs(n) {
                let g = Graph::from_edges(n, n, &edges).expect("valid");
                let form = canonical_graph(&g);
                assert_eq!(form.automorphisms, automorphism_count(&g));
                if n <= 5 {
                    let brute = brute_whole_key(&edges, n);
                    let mapped = ours_to_brute
                        .entry(form.edges.clone())
                        .or_insert_with(|| brute.clone());
                    assert_eq!(mapped, &brute, "two non-isomorphic graphs share a key");
                    let mapped = brute_to_ours
                        .entry(brute)
                        .or_insert_with(|| form.edges.clone());
                    assert_eq!(mapped, &form.edges, "isomorphic graphs got different keys");
                } else {
                    ours_to_brute.insert(form.edges, Vec::new());
                }
            }
            assert_eq!(ours_to_brute.len(), classes, "n = {n}");
        }
    }

    #[test]
    fn whole_graph_key_invariant_under_relabeling() {
        use crate::graph::Graph;
        let mut rng = Rng(5);
        for _ in 0..200 {
            let n = 1 + rng.below(60);
            let mut edges = random_ball(n, rng.below(2 * n), &mut rng);
            if rng.below(3) == 0 {
                edges.truncate(edges.len() / 2); // sometimes disconnected
            }
            let original = canonical_graph(&Graph::from_edges(n, n, &edges).expect("valid"));
            let mut perm: Vec<Node> = (0..n as Node).collect();
            for i in (1..n).rev() {
                perm.swap(i, rng.below(i + 1));
            }
            let relabeled: Vec<Edge> = edges
                .iter()
                .map(|&(u, v)| (perm[v as usize], perm[u as usize]))
                .collect();
            let other = canonical_graph(&Graph::from_edges(n, n, &relabeled).expect("valid"));
            assert_eq!(other, original);
        }
    }
}
