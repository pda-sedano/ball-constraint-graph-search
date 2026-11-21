//! A compiled ruleset: radius r plus the ball types that survived compiling.

use std::collections::HashMap;

use crate::ball::CompiledBall;
use crate::canonical::CanonicalKey;

/// The ball types of a ruleset with no compile errors, duplicates merged.
///
/// Each kept ball remembers its position in the ruleset as written, so
/// anything reported later (pruning, sampling statistics) can point back at
/// the file. Step 4 adds the pruned list beside this one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledRuleset {
    r: u32,
    /// Distinct ball types, in order of first appearance.
    balls: Vec<CompiledBall>,
    /// `source_indices[i]` is the position in the file of `balls[i]`.
    source_indices: Vec<usize>,
    /// Canonical key of each ball to its index in `balls`.
    index: HashMap<CanonicalKey, usize>,
}

impl CompiledRuleset {
    /// Only `compile` builds these. Panics if the two vectors differ in
    /// length, or if two balls are isomorphic (compile merges those first).
    pub(crate) fn new(r: u32, balls: Vec<CompiledBall>, source_indices: Vec<usize>) -> Self {
        assert_eq!(
            balls.len(),
            source_indices.len(),
            "every ball needs exactly one source index"
        );
        let mut index = HashMap::with_capacity(balls.len());
        for (i, ball) in balls.iter().enumerate() {
            let previous = index.insert(ball.canonical_key().clone(), i);
            assert!(
                previous.is_none(),
                "balls {previous:?} and {i} are isomorphic"
            );
        }
        Self {
            r,
            balls,
            source_indices,
            index,
        }
    }

    pub fn r(&self) -> u32 {
        self.r
    }

    /// Number of distinct ball types.
    pub fn len(&self) -> usize {
        self.balls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.balls.is_empty()
    }

    pub fn balls(&self) -> &[CompiledBall] {
        &self.balls
    }

    pub fn ball(&self, i: usize) -> Option<&CompiledBall> {
        self.balls.get(i)
    }

    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Indices (into `balls`) of types with depth less than r. They are kept:
    /// one applies only when n equals its size, known only at search time.
    pub fn shallow_indices(&self) -> Vec<usize> {
        self.balls
            .iter()
            .enumerate()
            .filter(|(_, ball)| (ball.depth() as u64) < u64::from(self.r))
            .map(|(i, _)| i)
            .collect()
    }

    /// The index of the ball type with this canonical key, if any: the
    /// lookup that checks a core node, whose ball must match a type exactly.
    pub fn index_of(&self, key: &CanonicalKey) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Largest root degree over all types: the sampler's hard degree cap.
    /// 0 for an empty ruleset.
    pub fn max_root_degree(&self) -> usize {
        self.balls
            .iter()
            .map(CompiledBall::root_degree)
            .max()
            .unwrap_or(0)
    }
}
