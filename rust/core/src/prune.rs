//! Visible sub-ball pruning: removing ball types that no graph can contain.
//!
//! Inside a type B, a node u at distance d from the root sees its whole
//! radius-(r − d) ball: every node within r − d of u is within r of the root,
//! and so is every shortest path between them. In any graph where B occurs,
//! the node playing u has its own radius-r ball, which must also be a type,
//! and cutting that type down to radius r − d must give the same sub-ball. So
//! if no type cuts down to it, B can never occur and is removed. Removing one
//! type can leave another without a match, so this repeats until nothing
//! changes.
//!
//! The check is necessary, not sufficient: types that survive may still be
//! unrealizable. The argument holds in infinite graphs too, so it also holds
//! for patches whose open nodes stand for a larger graph.
//!
//! Rounds are synchronous: round k removes every type that fails against the
//! types left after round k − 1. Which types survive does not depend on the
//! order (they form the largest set in which every type passes), and neither
//! do the rounds, so the report does not depend on how the file orders types.
//!
//! With r = 1 there is nothing to check: the only sub-ball inside a type is
//! a single node, which every type has.

use std::collections::HashMap;

use crate::ball::{CompiledBall, Node};
use crate::canonical::CanonicalKey;
use crate::extract::BallExtractor;
use crate::graph::Graph;

/// One ball type removed by pruning, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removal {
    /// Index of the removed type among the ruleset's balls.
    pub index: usize,
    /// The round that removed it, from 1.
    pub round: usize,
    /// A node of the removed type whose sub-ball no remaining type has: the
    /// first such node in BFS layer order, so the shallowest.
    pub node: Node,
    /// That node's distance from the root, between 1 and r − 1.
    pub depth: u32,
    /// Radius of the sub-ball, r − depth.
    pub radius: u32,
    /// The sub-ball itself, numbered from `node` as its root 0.
    pub sub_ball: CompiledBall,
    /// Indices of the types that did have this sub-ball at their root, all
    /// removed in earlier rounds. Empty in round 1: no type ever had it.
    pub lost_suppliers: Vec<usize>,
}

/// The result of pruning a ruleset: which types survive, and why the others
/// were removed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pruning {
    /// Indices of surviving types, ascending.
    kept: Vec<usize>,
    /// In order of removal: by round, then by index.
    removed: Vec<Removal>,
}

impl Pruning {
    /// Indices of the types that survive, ascending.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    /// Every removed type with its reason, by round, then by index.
    pub fn removed(&self) -> &[Removal] {
        &self.removed
    }

    /// Whether the type at `index` survived.
    pub fn is_kept(&self, index: usize) -> bool {
        self.kept.binary_search(&index).is_ok()
    }

    /// Number of rounds that removed something. 0 if nothing was removed.
    pub fn rounds(&self) -> usize {
        self.removed.last().map_or(0, |removal| removal.round)
    }
}

/// Prune `balls`, the distinct types of a ruleset of radius `r`: tabulate
/// what each type needs and supplies, then remove types in rounds.
///
/// Each type costs one extraction and canonical form per node at depth 1 to
/// r − 1 and per radius: the same order of work as compiling the types, on
/// smaller balls. Measured: under a millisecond for a handful of types, 65 ms
/// for 1,154 types at r = 2.
pub(crate) fn prune(r: u32, balls: &[CompiledBall]) -> Pruning {
    remove_in_rounds(&NeedTable::build(r, balls))
}

/// A sub-ball that some type needs another type to have at its root.
struct Need {
    radius: u32,
    ball: CompiledBall,
    /// Indices of the types whose cut to `radius` is this ball.
    suppliers: Vec<usize>,
    /// Indices of the types that contain this ball as some node's sub-ball.
    dependents: Vec<usize>,
}

/// Where a type needs a sub-ball: the need, and the first node that shows it.
#[derive(Clone, Copy)]
struct Witness {
    need: usize,
    node: Node,
    depth: u32,
}

/// Every sub-ball that some type needs, stored once, with who needs it and
/// who supplies it. Built once; the rounds only read it.
struct NeedTable {
    needs: Vec<Need>,
    /// `ids[radius]` maps a canonical key to its need's index.
    ids: Vec<HashMap<CanonicalKey, usize>>,
    /// `witnesses[t]`: each need of type t, at the first node that shows it.
    witnesses: Vec<Vec<Witness>>,
    /// `supplies[t]`: the needs that type t's own cuts supply.
    supplies: Vec<Vec<usize>>,
}

impl NeedTable {
    /// Tabulate needs for every type first, then supplies: a cut supplies
    /// something only if some type needs it.
    fn build(r: u32, balls: &[CompiledBall]) -> Self {
        let mut table = Self {
            needs: Vec::new(),
            ids: vec![HashMap::new(); r as usize],
            witnesses: Vec::with_capacity(balls.len()),
            supplies: Vec::with_capacity(balls.len()),
        };
        let mut extractor = BallExtractor::new();
        let graphs: Vec<Graph> = balls.iter().map(ball_graph).collect();
        for (index, (ball, graph)) in balls.iter().zip(&graphs).enumerate() {
            table.add_needs(index, ball, graph, r, &mut extractor);
        }
        for (index, graph) in graphs.iter().enumerate() {
            table.add_supplies(index, graph, r, &mut extractor);
        }
        table
    }

    fn num_types(&self) -> usize {
        self.witnesses.len()
    }

    /// Type `index`'s needs: for every node at depth 1 to r − 1, its sub-ball
    /// of radius r − depth, taken inside the type.
    fn add_needs(
        &mut self,
        index: usize,
        ball: &CompiledBall,
        graph: &Graph,
        r: u32,
        extractor: &mut BallExtractor,
    ) {
        let mut found: Vec<Witness> = Vec::new();
        for (depth, layer) in ball.layers().iter().enumerate().take(r as usize).skip(1) {
            let depth = depth as u32;
            let radius = r - depth;
            for &node in layer {
                let need =
                    self.intern(radius, extractor.extract(graph, node, radius).to_compiled());
                if !found.iter().any(|w| w.need == need) {
                    found.push(Witness { need, node, depth });
                    self.needs[need].dependents.push(index);
                }
            }
        }
        self.witnesses.push(found);
    }

    /// The index of the need for this sub-ball, adding it if it is new.
    fn intern(&mut self, radius: u32, sub_ball: CompiledBall) -> usize {
        let ids = &mut self.ids[radius as usize];
        if let Some(&id) = ids.get(sub_ball.canonical_key()) {
            return id;
        }
        ids.insert(sub_ball.canonical_key().clone(), self.needs.len());
        self.needs.push(Need {
            radius,
            ball: sub_ball,
            suppliers: Vec::new(),
            dependents: Vec::new(),
        });
        self.needs.len() - 1
    }

    /// Type `index`'s supplies: for each radius from 1 to r − 1, its cut
    /// (the ball of that radius around its root), if some type needs it.
    fn add_supplies(&mut self, index: usize, graph: &Graph, r: u32, extractor: &mut BallExtractor) {
        let mut supplied = Vec::new();
        for radius in 1..r {
            let cut = extractor.extract(graph, 0, radius).to_compiled();
            if let Some(&need) = self.ids[radius as usize].get(cut.canonical_key()) {
                supplied.push(need);
                self.needs[need].suppliers.push(index);
            }
        }
        self.supplies.push(supplied);
    }

    /// The first need of type `index` that no type alive supplies, if any.
    fn first_unmet(&self, index: usize, supply: &[usize]) -> Option<Witness> {
        self.witnesses[index]
            .iter()
            .find(|w| supply[w.need] == 0)
            .copied()
    }

    /// Why type `index` is removed in `round`, given the supply at the start
    /// of that round.
    fn removal(&self, index: usize, round: usize, supply: &[usize]) -> Removal {
        let witness = self
            .first_unmet(index, supply)
            .expect("every type in a round has a need nothing supplies");
        let need = &self.needs[witness.need];
        Removal {
            index,
            round,
            node: witness.node,
            depth: witness.depth,
            radius: need.radius,
            sub_ball: need.ball.clone(),
            lost_suppliers: need.suppliers.clone(),
        }
    }

    /// Take away what the `removed` types supplied. Returns the types still
    /// alive that depend on a need left with no supplier: the next round.
    fn withdraw(&self, removed: &[usize], supply: &mut [usize], alive: &[bool]) -> Vec<usize> {
        let mut next = Vec::new();
        for &index in removed {
            for &need in &self.supplies[index] {
                supply[need] -= 1;
                if supply[need] == 0 {
                    next.extend(self.needs[need].dependents.iter().filter(|&&d| alive[d]));
                }
            }
        }
        next
    }
}

/// The graph of a compiled ball, to extract sub-balls from.
fn ball_graph(ball: &CompiledBall) -> Graph {
    Graph::from_edges_with_own_cap(ball.num_nodes(), ball.edges())
        .expect("a compiled ball is a valid graph")
}

/// Round 1 removes every type with a need that nothing supplies. Each
/// removal lowers the supply of what it supplied, and a need whose supply
/// reaches 0 sends its remaining dependents to the next round.
fn remove_in_rounds(table: &NeedTable) -> Pruning {
    let mut supply: Vec<usize> = table
        .needs
        .iter()
        .map(|need| need.suppliers.len())
        .collect();
    let mut alive = vec![true; table.num_types()];
    let mut removed = Vec::new();
    let mut members: Vec<usize> = (0..table.num_types())
        .filter(|&index| table.first_unmet(index, &supply).is_some())
        .collect();
    let mut round = 0;
    while !members.is_empty() {
        round += 1;
        members.sort_unstable();
        members.dedup();
        // Reasons first, while `supply` still counts the types alive at the
        // start of this round.
        for &index in &members {
            removed.push(table.removal(index, round, &supply));
            alive[index] = false;
        }
        members = table.withdraw(&members, &mut supply, &alive);
    }
    let kept = (0..table.num_types())
        .filter(|&index| alive[index])
        .collect();
    Pruning { kept, removed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Edge;

    fn balls(lists: &[&[Edge]]) -> Vec<CompiledBall> {
        lists
            .iter()
            .map(|edges| CompiledBall::from_checked_edges(edges))
            .collect()
    }

    /// (index, round) of each removal.
    fn rounds(pruning: &Pruning) -> Vec<(usize, usize)> {
        pruning
            .removed()
            .iter()
            .map(|removal| (removal.index, removal.round))
            .collect()
    }

    /// The design doc's example: root, then a, then b and c under a.
    const STAR: &[Edge] = &[(0, 1), (1, 2), (1, 3)];
    /// The path seen from its middle: root with two arms of length 2.
    const PATH_BALL: &[Edge] = &[(0, 1), (0, 2), (1, 3), (2, 4)];
    /// Root with a leaf and an arm of length 2: needs a root of degree 1 at
    /// radius 1, which only STAR has here.
    const LEAF_AND_ARM: &[Edge] = &[(0, 1), (0, 2), (2, 3)];

    #[test]
    fn design_doc_star_is_removed_and_nothing_is_left() {
        let pruning = prune(2, &balls(&[STAR]));
        assert_eq!(pruning.kept(), [] as [usize; 0]);
        let [removal] = pruning.removed() else {
            panic!("one removal expected, got {:?}", pruning.removed());
        };
        assert_eq!((removal.index, removal.round), (0, 1));
        assert_eq!((removal.node, removal.depth, removal.radius), (1, 1, 1));
        // Node a's radius-1 ball: a with three neighbors.
        assert_eq!(removal.sub_ball.edges(), [(0, 1), (0, 2), (0, 3)]);
        assert_eq!(removal.lost_suppliers, [] as [usize; 0]);
        assert_eq!(pruning.rounds(), 1);
    }

    #[test]
    fn a_type_can_supply_its_own_needs() {
        let pruning = prune(2, &balls(&[PATH_BALL]));
        assert_eq!(pruning.kept(), [0]);
        assert_eq!(pruning.removed(), []);
        assert_eq!(pruning.rounds(), 0);
    }

    #[test]
    fn removals_cascade_through_rounds() {
        let pruning = prune(2, &balls(&[PATH_BALL, STAR, LEAF_AND_ARM]));
        assert_eq!(rounds(&pruning), [(1, 1), (2, 2)]);
        assert_eq!(pruning.kept(), [0]);
        let second = &pruning.removed()[1];
        assert_eq!((second.node, second.depth), (1, 1));
        assert_eq!(second.sub_ball.edges(), [(0, 1)]);
        assert_eq!(second.lost_suppliers, [1]);
        assert!(pruning.is_kept(0) && !pruning.is_kept(1) && !pruning.is_kept(2));
    }

    #[test]
    fn rounds_do_not_depend_on_type_order() {
        let pruning = prune(2, &balls(&[LEAF_AND_ARM, STAR, PATH_BALL]));
        assert_eq!(rounds(&pruning), [(1, 1), (0, 2)]);
        assert_eq!(pruning.kept(), [2]);
    }

    #[test]
    fn radius_one_never_prunes() {
        let pruning = prune(1, &balls(&[&[(0, 1)], &[(0, 1), (0, 2), (0, 3)]]));
        assert_eq!(pruning.kept(), [0, 1]);
        assert_eq!(pruning.removed(), []);
    }

    #[test]
    fn shallow_and_deep_types_of_one_graph_support_each_other() {
        //! The 3-node path at r = 2: its middle sees the whole graph (depth 1,
        //! shallow) and each end sees depth 2. Neither type survives alone.
        let middle: &[Edge] = &[(0, 1), (0, 2)];
        let end: &[Edge] = &[(0, 1), (1, 2)];
        assert_eq!(prune(2, &balls(&[middle, end])).kept(), [0, 1]);
        assert_eq!(prune(2, &balls(&[middle])).kept(), [] as [usize; 0]);
        assert_eq!(prune(2, &balls(&[end])).kept(), [] as [usize; 0]);
    }

    #[test]
    fn deeper_nodes_are_checked_at_smaller_radius() {
        //! r = 3, a path seen from its middle: depth-1 nodes need its radius-2
        //! cut and depth-2 nodes its radius-1 cut, and it has both. With one
        //! arm cut short, node 1 still sees a radius-2 path, but node 2, on the
        //! short arm, sees the path end 2 steps away on one side: no match.
        let path3: &[Edge] = &[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)];
        assert_eq!(prune(3, &balls(&[path3])).kept(), [0]);
        let lopsided: &[Edge] = &[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)];
        let pruning = prune(3, &balls(&[lopsided]));
        let [removal] = pruning.removed() else {
            panic!("one removal expected");
        };
        assert_eq!((removal.node, removal.depth, removal.radius), (2, 1, 2));
        assert_eq!(removal.sub_ball.edges(), [(0, 1), (0, 2), (1, 3)]);
    }

    #[test]
    fn empty_ruleset_prunes_to_empty() {
        assert_eq!(prune(2, &[]), Pruning::default());
    }
}
