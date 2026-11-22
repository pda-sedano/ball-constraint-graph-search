//! The embedding check: could a node's visible ball be part of an allowed type?
//!
//! Open and shadow nodes see only part of their ball, so they cannot be looked
//! up by canonical key. A node v passes if its visible ball embeds into some
//! kept type T as a rooted induced subgraph: v goes to T's root, visible nodes
//! go to distinct nodes of T, and edges and non-edges are both preserved.
//! Unseen paths can only shorten distances, so every visible node really is in
//! v's ball, and the patch is an induced piece of the larger graph, so what it
//! shows between visible nodes is final.
//!
//! Two rules keep the embedding honest (the design doc's "Embedding check"):
//!
//! 1. **Distance window.** A visible node u at distance d(v, u) in the patch
//!    must sit in T at a depth between `min(d(v,u), d(v,O) + 2 + d(u,O))` and
//!    `d(v,u)`, where O is the open set: a shortcut through unseen territory
//!    leaves the patch at an open node, takes at least 2 steps outside, and
//!    comes back in at another.
//! 2. **Closed degree.** If d(v, u) < r, all of u's neighbors are in the ball,
//!    so its image has exactly u's degree if u is closed, and at least that
//!    many if u is open.
//!
//! Every image also has at least u's degree in the ball, since u's ball
//! neighbors go to distinct neighbors of the image. That follows from the
//! definition, so it filters without changing any answer.
//!
//! For a core node every window is one depth and every degree is exact, which
//! forces an isomorphism: canonical lookup is the fast path of this check.
//!
//! # Method
//!
//! Chosen by benchmark (see the design doc): plain backtracking over visible
//! nodes in BFS order, with twin symmetry breaking, and constraint propagation
//! as the fallback once a step budget runs out.
//!
//! - **Twins** are visible nodes with the same neighbors and the same values
//!   for everything the rules read. They are interchangeable, so only
//!   embeddings that give twins increasing images are searched. This removes
//!   the factorial blowup over interchangeable leaves.
//! - **Propagation** keeps a candidate set per visible node, as a bitset over
//!   T's nodes, and shrinks the sets with LAD's neighborhood matching filter
//!   and a global all-different matching check, deciding shallow nodes first.
//!   It costs more per call but has far shorter worst cases.

use crate::ball::Node;
use crate::extract::VisibleBall;
use crate::ruleset::CompiledRuleset;

/// Backtracking steps before falling back to propagation. Easy checks take
/// tens of steps; the benchmark's hard cases took millions.
pub const STEP_BUDGET: u64 = 10_000;

const NONE: u32 = u32::MAX;

// ------------------------------------------------------------------ bitsets

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

fn has_bit(row: &[u64], b: usize) -> bool {
    row[b / 64] >> (b % 64) & 1 == 1
}

fn set_bit(row: &mut [u64], b: usize) {
    row[b / 64] |= 1 << (b % 64);
}

fn clear_bit(row: &mut [u64], b: usize) {
    row[b / 64] &= !(1 << (b % 64));
}

fn is_empty(row: &[u64]) -> bool {
    row.iter().all(|&w| w == 0)
}

fn count(row: &[u64]) -> u32 {
    row.iter().map(|w| w.count_ones()).sum()
}

fn bits(row: &[u64]) -> impl Iterator<Item = usize> + '_ {
    row.iter().enumerate().flat_map(|(i, &word)| {
        let mut w = word;
        std::iter::from_fn(move || {
            (w != 0).then(|| {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                i * 64 + b
            })
        })
    })
}

/// Adjacency rows as bitsets: row a, bit b set when a and b are adjacent.
fn adjacency_bits(adjacency: &[Vec<Node>]) -> (Vec<u64>, usize) {
    let words = words_for(adjacency.len());
    let mut rows = vec![0u64; adjacency.len() * words];
    for (a, neighbors) in adjacency.iter().enumerate() {
        for &b in neighbors {
            set_bit(&mut rows[a * words..][..words], b as usize);
        }
    }
    (rows, words)
}

// ------------------------------------------------------------------ inputs

/// One kept ball type, laid out for matching.
struct Pattern {
    depth: Vec<u32>,
    degree: Vec<u32>,
    adjacency: Vec<Vec<Node>>,
    bits: Vec<u64>,
    words: usize,
    /// `within[k]`: how many nodes have depth at most k, for k in 0..=r.
    within: Vec<usize>,
}

impl Pattern {
    fn len(&self) -> usize {
        self.depth.len()
    }

    fn adjacent(&self, a: usize, b: usize) -> bool {
        has_bit(&self.bits[a * self.words..][..self.words], b)
    }

    fn row(&self, a: usize) -> &[u64] {
        &self.bits[a * self.words..][..self.words]
    }
}

/// A visible ball, with everything the rules need to know about each node.
///
/// Nodes are in the ball's BFS numbering: the center is 0, and every other
/// node comes after a neighbor one step closer to the center.
#[derive(Debug, Clone)]
pub struct EmbedQuery {
    r: u32,
    /// Distance from the center in the patch, d(v, u).
    depth: Vec<u32>,
    /// Lower end of the distance window.
    lower: Vec<u32>,
    open: Vec<bool>,
    /// Degree within the ball.
    degree: Vec<u32>,
    adjacency: Vec<Vec<Node>>,
    bits: Vec<u64>,
    words: usize,
    /// A neighbor one step closer to the center (0 for the center itself).
    parent: Vec<u32>,
    /// `within[k]`: how many nodes are at distance at most k, for k in 0..=r.
    within: Vec<usize>,
    /// The previous and next member of each node's twin class, or NONE.
    prev_twin: Vec<u32>,
    next_twin: Vec<u32>,
}

impl EmbedQuery {
    /// The query for the center of `ball`, taken with radius `r` in a patch
    /// where `open[w]` says whether graph node w is open and
    /// `distance_to_open[w]` is its distance to the nearest open node
    /// (`u32::MAX` if there is none, as in closed mode).
    ///
    /// # Panics
    ///
    /// If `open` or `distance_to_open` is too short for the ball's nodes.
    pub fn new(ball: &VisibleBall, open: &[bool], distance_to_open: &[u32], r: u32) -> Self {
        let n = ball.num_nodes();
        let depth = ball.depths().to_vec();
        let mut adjacency = vec![Vec::new(); n];
        for &(a, b) in ball.edges() {
            adjacency[a as usize].push(b);
            adjacency[b as usize].push(a);
        }
        let (bits, words) = adjacency_bits(&adjacency);

        let graph_node = |i: usize| ball.nodes()[i] as usize;
        let open: Vec<bool> = (0..n).map(|i| open[graph_node(i)]).collect();
        let to_open: Vec<u32> = (0..n).map(|i| distance_to_open[graph_node(i)]).collect();
        let lower = (0..n)
            .map(|i| {
                let shortcut = to_open[0].saturating_add(2).saturating_add(to_open[i]);
                depth[i].min(shortcut)
            })
            .collect();
        let parent = (0..n)
            .map(|i| {
                adjacency[i]
                    .iter()
                    .copied()
                    .filter(|&p| depth[p as usize] + 1 == depth[i])
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        let within = (0..=r)
            .map(|k| depth.iter().filter(|&&d| d <= k).count())
            .collect();

        let mut query = EmbedQuery {
            r,
            degree: adjacency.iter().map(|a| a.len() as u32).collect(),
            depth,
            lower,
            open,
            adjacency,
            bits,
            words,
            parent,
            within,
            prev_twin: vec![NONE; n],
            next_twin: vec![NONE; n],
        };
        query.find_twins();
        query
    }

    /// Number of visible nodes.
    pub fn len(&self) -> usize {
        self.depth.len()
    }

    /// Always false: a ball holds at least its center.
    pub fn is_empty(&self) -> bool {
        self.depth.is_empty()
    }

    fn adjacent(&self, a: usize, b: usize) -> bool {
        has_bit(&self.bits[a * self.words..][..self.words], b)
    }

    /// Chain each node to the previous node with the same neighbors (apart
    /// from each other) and the same rule inputs. Openness is a rule input
    /// only where the degree rule applies, below depth r.
    fn find_twins(&mut self) {
        let words = self.words;
        for i in 1..self.len() {
            for j in (1..i).rev() {
                let same_inputs = self.depth[i] == self.depth[j]
                    && self.lower[i] == self.lower[j]
                    && (self.depth[i] == self.r || self.open[i] == self.open[j])
                    && self.next_twin[j] == NONE;
                if !same_inputs {
                    continue;
                }
                let same_neighbors = (0..words).all(|w| {
                    let mut a = self.bits[i * words + w];
                    let mut b = self.bits[j * words + w];
                    if w == j / 64 {
                        a &= !(1 << (j % 64));
                    }
                    if w == i / 64 {
                        b &= !(1 << (i % 64));
                    }
                    a == b
                });
                if same_neighbors {
                    self.prev_twin[i] = j as u32;
                    self.next_twin[j] = i as u32;
                    break;
                }
            }
        }
    }
}

/// Whether visible node `u` may go to pattern node `t`: both rules, plus the
/// degree bound every image satisfies.
fn admits(q: &EmbedQuery, u: usize, p: &Pattern, t: usize) -> bool {
    let depth = p.depth[t];
    if depth < q.lower[u] || depth > q.depth[u] {
        return false;
    }
    let degree = p.degree[t];
    if degree < q.degree[u] {
        return false;
    }
    q.depth[u] >= q.r || q.open[u] || degree == q.degree[u]
}

/// Cheap reasons a pattern cannot fit: too few nodes, too few within some
/// depth (nodes can only move shallower), or a root that fails the rules.
fn could_fit(q: &EmbedQuery, p: &Pattern) -> bool {
    p.len() >= q.len()
        && q.within
            .iter()
            .zip(&p.within)
            .all(|(need, have)| need <= have)
        && admits(q, 0, p, 0)
}

// ------------------------------------------------------------------ backtracking

/// Plain backtracking in BFS order with twin symmetry breaking.
#[derive(Default)]
struct Backtrack {
    image: Vec<u32>,
    used: Vec<bool>,
    steps: u64,
    budget: u64,
}

impl Backtrack {
    /// Some(answer), or None if the budget ran out first.
    fn embeds(&mut self, q: &EmbedQuery, p: &Pattern, budget: u64) -> Option<bool> {
        self.image.clear();
        self.image.resize(q.len(), 0);
        self.used.clear();
        self.used.resize(p.len(), false);
        self.used[0] = true;
        self.steps = 0;
        self.budget = budget;
        let found = self.extend(q, p, 1);
        if found || self.steps <= self.budget {
            Some(found)
        } else {
            None
        }
    }

    fn extend(&mut self, q: &EmbedQuery, p: &Pattern, u: usize) -> bool {
        if u == q.len() {
            return true;
        }
        self.steps += 1;
        if self.steps > self.budget {
            return false;
        }
        let anchor = self.image[q.parent[u] as usize] as usize;
        let after = match q.prev_twin[u] {
            NONE => None,
            twin => Some(self.image[twin as usize]),
        };
        for &t in &p.adjacency[anchor] {
            let t = t as usize;
            if self.used[t] || after.is_some_and(|a| t as u32 <= a) || !admits(q, u, p, t) {
                continue;
            }
            let consistent =
                (0..u).all(|x| q.adjacent(u, x) == p.adjacent(t, self.image[x] as usize));
            if !consistent {
                continue;
            }
            self.image[u] = t as u32;
            self.used[t] = true;
            if self.extend(q, p, u + 1) {
                return true;
            }
            self.used[t] = false;
        }
        false
    }
}

// ------------------------------------------------------------------ propagation

/// Constraint propagation over bitset candidate sets, after LAD and the
/// Glasgow subgraph solver. Complete: it always answers.
#[derive(Default)]
struct Propagation {
    /// Matching scratch: right node to its left node + 1, 0 if free.
    owner: Vec<u32>,
    seen: Vec<u64>,
    rows: Vec<u64>,
}

impl Propagation {
    fn embeds(&mut self, q: &EmbedQuery, p: &Pattern) -> bool {
        let words = p.words;
        let mut candidates = vec![0u64; q.len() * words];
        for u in 0..q.len() {
            let row = &mut candidates[u * words..][..words];
            for t in 0..p.len() {
                if admits(q, u, p, t) {
                    set_bit(row, t);
                }
            }
            if is_empty(row) {
                return false;
            }
        }
        let mut applied = vec![false; q.len()];
        self.narrow(q, p, &mut candidates, &mut applied) && self.search(q, p, &candidates, &applied)
    }

    /// Decide the shallowest undecided node with the fewest candidates, try
    /// each, narrow, recurse.
    fn search(
        &mut self,
        q: &EmbedQuery,
        p: &Pattern,
        candidates: &[u64],
        applied: &[bool],
    ) -> bool {
        let words = p.words;
        let choice = (0..q.len())
            .map(|u| (u, count(&candidates[u * words..][..words])))
            .filter(|&(_, size)| size > 1)
            .min_by_key(|&(u, size)| (q.depth[u], size));
        let Some((u, _)) = choice else {
            return true; // every node decided, every constraint applied
        };
        let options: Vec<usize> = bits(&candidates[u * words..][..words]).collect();
        for t in options {
            let mut next = candidates.to_vec();
            let mut next_applied = applied.to_vec();
            let row = &mut next[u * words..][..words];
            row.fill(0);
            set_bit(row, t);
            order_twins(q, words, &mut next, u, t);
            if self.narrow(q, p, &mut next, &mut next_applied)
                && self.search(q, p, &next, &next_applied)
            {
                return true;
            }
        }
        false
    }

    /// Shrink candidate sets until nothing changes. False if a set empties or
    /// the nodes cannot all get distinct images.
    fn narrow(
        &mut self,
        q: &EmbedQuery,
        p: &Pattern,
        candidates: &mut [u64],
        applied: &mut [bool],
    ) -> bool {
        let words = p.words;
        loop {
            let mut changed = false;
            // A decided node's image is taken; its neighbors must sit next to
            // the image and its non-neighbors must not (induced).
            for u in 0..q.len() {
                if applied[u] || count(&candidates[u * words..][..words]) != 1 {
                    continue;
                }
                applied[u] = true;
                changed = true;
                let t = bits(&candidates[u * words..][..words])
                    .next()
                    .expect("one candidate");
                for x in (0..q.len()).filter(|&x| x != u) {
                    let adjacent = q.adjacent(u, x);
                    let row = &mut candidates[x * words..][..words];
                    for (cell, &a) in row.iter_mut().zip(p.row(t)) {
                        *cell &= if adjacent { a } else { !a };
                    }
                    clear_bit(row, t);
                    if is_empty(row) {
                        return false;
                    }
                }
            }
            // LAD: t stays a candidate for u only if u's neighbors can go
            // one-to-one to t's neighbors, each within its candidates.
            for u in (0..q.len()).filter(|&u| !applied[u]) {
                let options: Vec<usize> = bits(&candidates[u * words..][..words]).collect();
                for t in options {
                    if !self.neighbors_fit(q, p, candidates, u, t) {
                        clear_bit(&mut candidates[u * words..][..words], t);
                        changed = true;
                    }
                }
                if is_empty(&candidates[u * words..][..words]) {
                    return false;
                }
            }
            if !changed {
                break;
            }
        }
        self.rows.clear();
        self.rows.extend_from_slice(candidates);
        self.perfect_matching(q.len(), words, p.len())
    }

    fn neighbors_fit(
        &mut self,
        q: &EmbedQuery,
        p: &Pattern,
        candidates: &[u64],
        u: usize,
        t: usize,
    ) -> bool {
        let words = p.words;
        self.rows.clear();
        for &x in &q.adjacency[u] {
            let row = &candidates[x as usize * words..][..words];
            self.rows
                .extend(row.iter().zip(p.row(t)).map(|(c, a)| c & a));
        }
        self.perfect_matching(q.adjacency[u].len(), words, p.len())
    }

    /// Can every left node l get a distinct right node from its row
    /// `rows[l * words..][..words]`? Augmenting paths; the graphs are tiny.
    fn perfect_matching(&mut self, left: usize, words: usize, right: usize) -> bool {
        self.owner.clear();
        self.owner.resize(right, 0);
        for l in 0..left {
            self.seen.clear();
            self.seen.resize(words, 0);
            if !self.augment(l, words) {
                return false;
            }
        }
        true
    }

    fn augment(&mut self, l: usize, words: usize) -> bool {
        for w in 0..words {
            let mut free = self.rows[l * words + w] & !self.seen[w];
            while free != 0 {
                let b = free.trailing_zeros() as usize;
                free &= free - 1;
                self.seen[w] |= 1 << b;
                let r = w * 64 + b;
                let holder = self.owner[r];
                if holder == 0 || self.augment(holder as usize - 1, words) {
                    self.owner[r] = l as u32 + 1;
                    return true;
                }
            }
        }
        false
    }
}

/// Twins take increasing images: after deciding u -> t, earlier members of
/// u's class must go below t and later members above.
fn order_twins(q: &EmbedQuery, words: usize, candidates: &mut [u64], u: usize, t: usize) {
    let mut x = q.prev_twin[u];
    while x != NONE {
        let row = &mut candidates[x as usize * words..][..words];
        for b in t..words * 64 {
            clear_bit(row, b);
        }
        x = q.prev_twin[x as usize];
    }
    let mut x = q.next_twin[u];
    while x != NONE {
        let row = &mut candidates[x as usize * words..][..words];
        for b in 0..=t {
            clear_bit(row, b);
        }
        x = q.next_twin[x as usize];
    }
}

// ------------------------------------------------------------------ checker

/// Checks visible balls against the kept types of a compiled ruleset.
///
/// Holds scratch buffers, so keep one and reuse it for every node.
pub struct EmbeddingChecker {
    r: u32,
    patterns: Vec<Pattern>,
    /// Ball index in the ruleset of each pattern.
    ball_indices: Vec<usize>,
    backtrack: Backtrack,
    propagation: Propagation,
}

impl EmbeddingChecker {
    /// A checker for the types that survived pruning: pruned types occur in
    /// no graph, so matching against them could only accept wrongly.
    pub fn new(ruleset: &CompiledRuleset) -> Self {
        let r = ruleset.r();
        let ball_indices = ruleset.pruning().kept().to_vec();
        let patterns = ball_indices
            .iter()
            .map(|&i| {
                let ball = &ruleset.balls()[i];
                let n = ball.num_nodes();
                let mut depth = vec![0; n];
                for (d, layer) in ball.layers().iter().enumerate() {
                    for &v in layer {
                        depth[v as usize] = d as u32;
                    }
                }
                let adjacency: Vec<Vec<Node>> =
                    (0..n).map(|v| ball.neighbors(v as Node).to_vec()).collect();
                let (bits, words) = adjacency_bits(&adjacency);
                Pattern {
                    degree: adjacency.iter().map(|a| a.len() as u32).collect(),
                    within: (0..=r)
                        .map(|k| depth.iter().filter(|&&d| d <= k).count())
                        .collect(),
                    depth,
                    adjacency,
                    bits,
                    words,
                }
            })
            .collect();
        EmbeddingChecker {
            r,
            patterns,
            ball_indices,
            backtrack: Backtrack::default(),
            propagation: Propagation::default(),
        }
    }

    pub fn r(&self) -> u32 {
        self.r
    }

    /// Ball indices (into the ruleset) of the kept types the visible ball
    /// embeds into, ascending. Empty means the node fails.
    ///
    /// # Panics
    ///
    /// If the query was built with a different r.
    pub fn compatible(&mut self, query: &EmbedQuery) -> Vec<usize> {
        assert_eq!(
            query.r, self.r,
            "query built for r = {}, ruleset has r = {}",
            query.r, self.r
        );
        let mut found = Vec::new();
        for k in 0..self.patterns.len() {
            if self.embeds(query, k, STEP_BUDGET) {
                found.push(self.ball_indices[k]);
            }
        }
        found
    }

    fn embeds(&mut self, q: &EmbedQuery, k: usize, budget: u64) -> bool {
        let p = &self.patterns[k];
        if !could_fit(q, p) {
            return false;
        }
        match self.backtrack.embeds(q, p, budget) {
            Some(answer) => answer,
            None => self.propagation.embeds(q, p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Edge;
    use crate::extract::BallExtractor;
    use crate::graph::Graph;
    use crate::testing::{
        census_ruleset, cut_patch, graph, path, random_graph, ruleset, Rng, PATH_BALL,
    };

    /// The query for `center` in `g` with these open nodes.
    fn query(g: &Graph, open_nodes: &[Node], center: Node, r: u32) -> EmbedQuery {
        let mut open = vec![false; g.num_nodes()];
        for &v in open_nodes {
            open[v as usize] = true;
        }
        let distance = g.distances_from(open_nodes);
        let ball = BallExtractor::new().extract(g, center, r);
        EmbedQuery::new(&ball, &open, &distance, r)
    }

    #[test]
    fn open_path_end_fits_the_path_ball_but_closed_does_not() {
        //! The design doc's hand case: a long path with L = {path ball}.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut checker = EmbeddingChecker::new(&rules);
        let g = graph(7, &path(7));
        assert_eq!(checker.compatible(&query(&g, &[0, 6], 0, 2)), [0]);
        assert_eq!(checker.compatible(&query(&g, &[0, 6], 1, 2)), [0]);
        assert_eq!(checker.compatible(&query(&g, &[], 0, 2)), [] as [usize; 0]);
        assert_eq!(checker.compatible(&query(&g, &[], 3, 2)), [0]);
    }

    #[test]
    fn distance_window_allows_a_shortcut_only_between_open_nodes() {
        //! r = 3, L = {5-cycle}. In the patch a-b-c-d, d is 3 from a, but a
        //! 5-cycle has nothing at depth 3. With a and d both open they might be
        //! 2 apart through an unseen node, so d may sit at depth 2 and the
        //! path fits around the cycle. With d closed it cannot.
        let cycle5: &[Edge] = &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)];
        let rules = ruleset(3, &[cycle5]);
        let mut checker = EmbeddingChecker::new(&rules);
        let g = graph(4, &path(4));
        assert_eq!(checker.compatible(&query(&g, &[0, 3], 0, 3)), [0]);
        assert_eq!(checker.compatible(&query(&g, &[0], 0, 3)), [] as [usize; 0]);
    }

    #[test]
    fn closed_degree_must_match_exactly_and_open_may_fall_short() {
        //! L = {path ball}: every node within depth 1 has degree 2.
        let rules = ruleset(2, &[PATH_BALL]);
        let mut checker = EmbeddingChecker::new(&rules);
        // A path with a pendant on node 3: node 3 has degree 3.
        let mut edges = path(7);
        edges.push((3, 7));
        let g = graph(8, &edges);
        // Node 2 sees node 3 at distance 1; closed, its degree 3 must match: no.
        assert_eq!(
            checker.compatible(&query(&g, &[0, 6, 7], 2, 2)),
            [] as [usize; 0]
        );
        // Node 1 sees node 3 at distance 2 = r: no degree rule there, but the
        // pendant 7 is at distance 3, out of sight. Passes.
        assert_eq!(checker.compatible(&query(&g, &[0, 6, 7], 1, 2)), [0]);
        // A node of degree 1 that is open may have an unseen neighbor.
        let g = graph(3, &path(3));
        assert_eq!(checker.compatible(&query(&g, &[0, 2], 1, 2)), [0]);
        assert_eq!(checker.compatible(&query(&g, &[2], 1, 2)), [] as [usize; 0]);
    }

    #[test]
    fn only_kept_types_are_matched() {
        //! The star is pruned, so even a ball that fits it gets nothing from it.
        let star: &[Edge] = &[(0, 1), (1, 2), (1, 3)];
        let rules = ruleset(2, &[PATH_BALL, star]);
        assert_eq!(rules.pruning().kept(), [0]);
        let mut checker = EmbeddingChecker::new(&rules);
        let g = graph(4, star);
        assert_eq!(checker.compatible(&query(&g, &[], 0, 2)), [] as [usize; 0]);
    }

    #[test]
    fn cut_test_every_node_keeps_its_own_type() {
        //! A patch cut from a graph whose census is L: each open or shadow node
        //! must still fit the type of its full ball in the original graph.
        let mut rng = Rng(12);
        let mut extractor = BallExtractor::new();
        for round in 0..30 {
            let r = 2 + (round % 2) as u32;
            let g = random_graph(60, 70 + rng.below(40), &mut rng);
            let rules = census_ruleset(&g, r);
            let mut checker = EmbeddingChecker::new(&rules);
            let (patch, open, region) = cut_patch(&g, 25, 0, &mut rng);
            let mut open_flags = vec![false; patch.num_nodes()];
            for &v in &open {
                open_flags[v as usize] = true;
            }
            let distance = patch.distances_from(&open);
            for v in 0..patch.num_nodes() {
                if distance[v] >= r {
                    continue;
                }
                let ball = extractor.extract(&patch, v as Node, r);
                let found = checker.compatible(&EmbedQuery::new(&ball, &open_flags, &distance, r));
                let truth = extractor.extract(&g, region[v], r).to_compiled();
                let own = rules
                    .index_of(truth.canonical_key())
                    .expect("census has it");
                assert!(
                    found.contains(&own),
                    "round {round}, node {v}: {found:?} lacks {own}"
                );
            }
        }
    }

    #[test]
    fn core_nodes_agree_with_canonical_lookup() {
        //! With no open nodes every window is exact: the embedding check must
        //! find exactly the type canonical lookup finds, if it was kept.
        let mut rng = Rng(5);
        let mut extractor = BallExtractor::new();
        for round in 0..30 {
            let r = 1 + (round % 3) as u32;
            let reference = random_graph(40, 50 + rng.below(30), &mut rng);
            let rules = census_ruleset(&reference, r);
            let mut checker = EmbeddingChecker::new(&rules);
            let g = random_graph(30, 35 + rng.below(25), &mut rng);
            let open = vec![false; g.num_nodes()];
            let distance = vec![u32::MAX; g.num_nodes()];
            for v in 0..g.num_nodes() as Node {
                let ball = extractor.extract(&g, v, r);
                let expected: Vec<usize> = rules
                    .index_of(ball.to_compiled().canonical_key())
                    .filter(|&i| rules.pruning().is_kept(i))
                    .into_iter()
                    .collect();
                let found = checker.compatible(&EmbedQuery::new(&ball, &open, &distance, r));
                assert_eq!(found, expected, "round {round}, node {v}");
            }
        }
    }

    #[test]
    fn backtracking_and_propagation_agree() {
        //! Every (query, type) pair answered by both methods: plain
        //! backtracking with an unlimited budget, and propagation alone.
        let mut rng = Rng(77);
        let mut extractor = BallExtractor::new();
        let mut yes = 0;
        let mut no = 0;
        for round in 0..40 {
            let r = 2 + (round % 2) as u32;
            let g = random_graph(50, 60 + rng.below(40), &mut rng);
            let rules = census_ruleset(&g, r);
            let mut checker = EmbeddingChecker::new(&rules);
            let flips = if round % 2 == 0 { 0 } else { 3 };
            let (patch, open, _) = cut_patch(&g, 20, flips, &mut rng);
            let mut open_flags = vec![false; patch.num_nodes()];
            for &v in &open {
                open_flags[v as usize] = true;
            }
            let distance = patch.distances_from(&open);
            for v in 0..patch.num_nodes() as Node {
                let ball = extractor.extract(&patch, v, r);
                let q = EmbedQuery::new(&ball, &open_flags, &distance, r);
                for k in 0..checker.patterns.len() {
                    let p = &checker.patterns[k];
                    if !could_fit(&q, p) {
                        continue;
                    }
                    let plain = checker
                        .backtrack
                        .embeds(&q, p, u64::MAX)
                        .expect("no budget");
                    let propagated = checker.propagation.embeds(&q, p);
                    assert_eq!(plain, propagated, "round {round}, node {v}, pattern {k}");
                    if plain {
                        yes += 1;
                    } else {
                        no += 1;
                    }
                }
            }
        }
        assert!(
            yes > 500 && no > 500,
            "too few of one answer: {yes} yes, {no} no"
        );
    }

    #[test]
    fn twins_are_interchangeable_nodes_only() {
        //! A star seen from its center: the leaves are twins. Opening one
        //! leaf at depth 1 < r splits it off, since the degree rule reads it.
        let star: &[Edge] = &[(0, 1), (0, 2), (0, 3)];
        let g = graph(4, star);
        let q = query(&g, &[], 0, 2);
        assert_eq!(q.prev_twin, [NONE, NONE, 1, 2]);
        let q = query(&g, &[3], 0, 2);
        assert_eq!(q.prev_twin, [NONE, NONE, 1, NONE]);
        // At depth r, openness does not matter.
        let g = graph(5, &[(0, 1), (1, 2), (1, 3), (1, 4)]);
        let q = query(&g, &[4], 0, 2);
        assert_eq!(q.prev_twin, [NONE, NONE, NONE, 2, 3]);
    }
}
