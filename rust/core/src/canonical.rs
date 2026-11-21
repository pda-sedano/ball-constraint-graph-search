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

use std::os::raw::c_int;

use nauty_Traces_sys::{
    densenauty, empty_graph, optionblk, statsblk, ADDONEEDGE, SETWORDSNEEDED, TRUE,
};

use crate::ball::{Edge, Node};

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
}
