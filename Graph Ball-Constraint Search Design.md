# Graph Ball-Constraint Search: Design

Oct 8, 2026 · @Perrin Alaine-Sedano

## Purpose and scope

The search finds finite graphs in which every node's radius-r neighborhood is one of an allowed list of shapes, then samples them as uniformly as possible.

- Graphs are simple, undirected, unlabelled and connected.
- Rulesets that only an infinite graph can satisfy are kept. For those, the result is a finite patch with a marked boundary.
- No construction order is built in. The target is a declarative ensemble, sampled by a Markov chain.
- Version 1 is a Python prototype on NetworkX and pynauty, for small n. Speed comes later, with a stated reason for each addition.

## Definitions

A node's ball is the induced subgraph on every node within distance r of it, with that node as root.

- **Induced** means absence counts. Two ball nodes that are not adjacent in the type must not be adjacent in the graph.
- **Ball type:** a rooted ball up to root-preserving isomorphism.
- **Ruleset L:** a radius r plus a set of allowed ball types. Allowed means a type may go unused.
- **Open set:** the nodes permitted to have unseen neighbors outside the patch. The search chooses it.

Every node falls into one of three classes, fixed by its distance to the nearest open node.

| Class | Distance to nearest open node | What is known | Constraint |
| --- | --- | --- | --- |
| Open | 0 | May be missing neighbors | Visible ball embeds into some type in L |
| Shadow | 1 to r − 1 | Own edges all present, ball may be incomplete | Visible ball embeds into some type in L |
| Core | r or more | Ball in the patch is the true ball | Ball is isomorphic to a type in L |

A **valid result** is a connected simple graph on n nodes plus an open set of at most m nodes, with every node meeting its class constraint.

With m = 0 every node is core. This is closed mode, and a valid result proves the ruleset is realizable by a finite graph.

## Inputs and outputs

The search takes four inputs and returns a graph with its open set, or a report that none was found.

| Input | Meaning |
| --- | --- |
| r | Ball radius |
| L | Allowed rooted ball types, each of depth exactly r |
| n | Number of nodes |
| m | Cap on open nodes. 0 means closed mode |

Outputs:

- The graph and its open set, or "none found".
- Each node's class, and its type ID or set of compatible types.
- Core fraction: the share of nodes that are core.
- Core depth: the largest distance-to-nearest-open-node, minus r. A deeper core means more trust in the result.
- Across a sweep: the smallest workable m for each n.

**How to use m.** The boundary a ruleset needs depends on its geometry. Lattice-like structure needs a boundary that grows like n^((d−1)/d). Tree-like structure needs a constant fraction of n. So m is a cap to sweep, not a fixed fraction, and how the smallest workable m scales with n is a dimension estimate.

Opening a node only ever relaxes constraints. With spare budget the sampler scatters open flags through the interior as defects. Sampling runs use exactly m flags at the smallest workable m.

## Ruleset file format

A ruleset is a JSON file that stores the ball graphs themselves, with the root always at node 0.

```json
{
  "name": "star-only",
  "r": 2,
  "types": [
    {"label": "leaf-view", "edges": [[0,1],[1,2],[1,3]]}
  ]
}
```

Conventions:

- Nodes are integers. The root is node 0.
- Non-edges are implied. Any pair not listed must not be adjacent.
- Labels are optional and only for reading.
- A type's identity is its canonical form, not its label or position. This makes rulesets comparable and lets them be combined by union, intersection and difference.
- Canonical keys are recomputed on load and never saved. They may not be stable across library versions.

Checks on load:

- Each type is simple and connected, with every node within distance r of the root.
- Depth is exactly r. Shallower types get a warning, since they only apply when n equals their size.
- Isomorphic duplicates are merged, with a warning.
- The sub-ball pruning runs and reports which types it removed and why. Both the list as written and the pruned list are kept.

Three ways to produce a ruleset, all writing this same format:

1. **By hand.** Fine for small cases. A function that draws each type by layer helps confirm the file says what was meant.
2. **From a reference graph.** Collect every ball that occurs, with its count. The counts are saved as metadata for comparing samples against later.
3. **By enumeration, later.** Generate every ball up to a degree bound and remove forbidden ones. A search over rulesets needs this. It is realistic only for small r and small degree.

## Pre-checks before searching

These are cheap filters that can prove a ruleset impossible early. Passing all of them does not prove a graph exists.

1. **Visible sub-ball pruning.** Inside a type B, a node at distance d from the root has its radius-(r − d) ball fully contained in B. Some type in L must truncate to that sub-ball. If none does, delete B and repeat until nothing changes. An empty L means the ruleset is unrealizable.
2. **Shallow types.** In a connected graph, a ball of depth less than r is the whole graph. Drop such types unless n equals their size.
3. **Counting, closed mode only.** Let x\_T be the number of nodes of type T. The counts sum to n, the degree sum is even, and adjacencies between types balance in both directions. This makes realizability depend on n.
4. **Size threshold.** Tree-like balls force high girth, so the smallest valid n grows quickly with r.

Worked example for check 1, with r = 2 and the single type root → a → (b, c): node a shows a degree-3 radius-1 ball, but the only type truncates to degree 1. The type is deleted and L is empty.

Worked example for check 3, with r = 1: if the only type is a root with one neighbor, the degree sum equals n, so odd n is impossible.

**Why there is no complete test.** For unrestricted neighborhoods the realization problem is recursively unsolvable, following Bulitko (1973), as stated in [Esselstein and Winkler, Building Graphs from Colored Trees](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v17i1r161). That result is for vertex-colored graphs. For a fixed n the problem is finite, so the search itself is the decider.

## Canonical form for rooted balls

A canonical form turns matching a ball against L into a dictionary lookup. The interface is one function: ball in, type ID out.

The root gives every node a distance label from 0 to r that any isomorphism must preserve. The standard method builds on that:

1. Color each node by its distance from the root.
2. Refine: a node's new color is its old color plus the sorted multiset of its neighbors' colors. Repeat until no class splits.
3. If every node has a unique color, sorting by color gives the canonical order.
4. If ties remain, single out each node of the first tied class in turn, refine again and recurse. Keep the branch with the lexicographically smallest edge list.
5. Output the layer sizes plus the sorted edge list under that numbering, as a hashable tuple.

Step 4 is what makes the result exact. Ties come either from real symmetry, where any choice gives the same output, or from structure that refinement cannot distinguish, where the choice matters.

Implementations:

- **Main path: pynauty.** Pass the ball with distance layers as vertex colors and take the canonical certificate. The key must also include the layer sizes, since a certificate alone does not record which colors were used.
- **Cross-check: NetworkX only.** `weisfeiler_lehman_graph_hash` covers steps 1 to 3 without tie-breaking, so non-isomorphic balls can collide. Use it as a bucket key, then confirm with a VF2 isomorphism test that matches nodes on distance.
- **Trees.** For a ball with no cycles, AHU encoding is exact: each node's code is the sorted tuple of its children's codes.

The two implementations share no code. Agreement on thousands of random balls is the test.

## Embedding check

An open or shadow node passes if its visible ball fits inside some type in L as a rooted induced subgraph, subject to two extra rules. Without those rules the check accepts almost anything.

**Why embedding is the right relation.** Unseen paths can only shorten distances, so every visible node lies in the true ball. The patch is an induced piece of the larger graph, so edges and non-edges among visible nodes are final.

**Notation.** v is the node being checked. u is a visible node, mapped to a node of type T. O is the open set. All distances are measured in the patch, and d(x, O) is the distance from x to the nearest open node.

**Rule 1: distance window.** A shortcut through unseen territory must leave and re-enter through open nodes. The depth of u's image in T must satisfy:

```latex
\min\bigl(d(v,u),\; d(v,O) + 2 + d(u,O)\bigr) \;\le\; \operatorname{depth}_T(\text{image of } u) \;\le\; d(v,u)
```

The root maps to the root, since its window is 0 to 0.

**Rule 2: closed degree.** If u is not open and d(v, u) < r, all of u's neighbors are visible. Its image must have exactly the same degree. If u is open, the image's degree must be at least as large.

A consequence: a shadow node's own degree is final, so it only matches types with that exact root degree.

**Output.** The check returns the set of compatible type IDs. An empty set is a violation. The set's size tells the search which nodes are most constrained.

**How to run it.**

- Both rules are pairwise tests between one visible node and one type node. They fit NetworkX's VF2 induced-subgraph matcher as a custom node-match function.
- Prefilter types by root degree and layer sizes before matching.
- Memoize lazily. The key is the canonical form of the visible ball colored by distance, open or closed, and window lower bound. The value is the compatible set.

**One checker for all three classes.** For a core node every window collapses to an exact depth and every degree is exact, which forces a full isomorphism. Core checking is the special case, with canonical lookup as its fast path.

**Limits.** The check is necessary, not sufficient, and errs toward accepting.

- Patch nodes at visible distance greater than r might be in the true ball. They are ignored.
- Each node is judged alone, so two neighbors can each pass using choices that disagree.

**Optional later pass: pairwise agreement.** Treat each node's choices as (type, placement) pairs. A choice at u fixes what neighbor w looks like out to radius r − 1. Delete any choice with no agreeing partner at some neighbor, and repeat until stable. This adds no nodes and catches conflicts between pairs only.

The exact fix is not in scope. Requiring all boundary nodes to agree jointly means building the nodes outside the patch, which moves the same question one layer out. A larger patch read from its deep core does the same job.

## Validity checker

The checker takes a graph, an open set, r, the compiled L and m, and returns a report instead of a bare true or false. Every search strategy depends on it, so it is built and tested first.

Steps:

1. **Global checks.** The graph is simple and connected, and the open set has at most m nodes.
2. **Classify nodes.** One multi-source BFS from the open set gives each node's distance to the nearest open node. 0 is open, 1 to r − 1 is shadow, r or more is core. With an empty open set every node is core.
3. **Extract each visible ball.** BFS to depth r, take the induced subgraph, keep the distances.
4. **Core nodes.** Canonical form, then lookup in L. Absent means fail.
5. **Open and shadow nodes.** Embedding check. An empty compatible set means fail.
6. **Report.** Verdict, failing nodes with reasons, each node's class and type ID or compatible set, core fraction, core depth.

Design choices:

- **A per-node function at the center.** The full check is that function over all nodes. After an edge change, only nodes within distance r of either endpoint need rechecking, before and after the change. The incremental version reuses the same function on that subset.
- **Full check first.** The incremental check is added later and tested against the full one.
- **Compiled L as its own object,** built once: canonical keys, per-node depth and degree for each type, and truncations for pruning.
- **The failing-node count doubles as the sampler's energy.**

## Sampler

The sampler is a Markov chain with soft constraints. In equilibrium it visits every valid state equally often, and it has no construction order.

**Setup.**

- State: a graph on n nodes plus an open set.
- Energy U: the number of failing nodes, plus (number of components − 1). Valid states are exactly those with U = 0.
- Hard cap: reject any edge addition that pushes a node above the largest root degree in L. Valid graphs never exceed it, so uniformity is unaffected.
- Sampling: record the state whenever U = 0, spaced far enough apart to be independent.

**Moves.** With probability 1/2 remove a uniformly random existing edge. Otherwise add a uniformly random non-edge. If the chosen kind has no candidates, the step is a rejection.

This proposal is not symmetric, so acceptance includes a correction. Let E be the current edge count, N = n(n − 1)/2 the number of node pairs, and ΔU the energy change.

```latex
P_{\text{remove}} = \min\!\left(1,\; e^{-\beta\,\Delta U}\,\frac{E}{N - E + 1}\right)
\qquad
P_{\text{add}} = \min\!\left(1,\; e^{-\beta\,\Delta U}\,\frac{N - E}{E + 1}\right)
```

In open mode, a third move swaps the flags of one random open node and one random non-open node. That move is symmetric and needs no correction.

**Why it is uniform.** The chain settles into a distribution where each state has weight e^(−βU). Every valid state has U = 0, so all valid states have equal weight at any β.

**Why there is no temporal bias.** Moves land at random places. The equilibrium distribution does not depend on the starting graph or the order of moves.

**Free choice.** Any energy that is zero exactly on valid states gives the same uniform result. The energy can be tuned for speed without affecting correctness.

**Counting convention.** The chain is uniform over numbered graphs. An unlabelled graph with symmetry group size |Aut| has n!/|Aut| numberings, so the chain weights it by 1/|Aut|.

- Primary statistic: each unlabelled graph counted once. Weight every sample by its |Aut|.
- Comparison statistic: the chain's native weighting.
- Store |Aut| with every sample, from pynauty. With open sets, count only symmetries that preserve which nodes are open.
- Reweighting works unless a few rare, very symmetric graphs dominate the count-once answer.

**Caveats.**

- Uniformity holds only after the chain has equilibrated, which cannot be proven in general. The checks are agreement between independent runs and exhaustive enumeration at small n.
- Steadily raising β finds valid graphs but not uniformly. It answers "does one exist", not statistical questions.

**Version 1.** One chain at a fixed β, closed mode only, the add and remove moves above, and a full recheck after every move.

## Tests

Each piece has a test that does not depend on the pieces built after it.

Hand cases for the checker:

- The star example, r = 2, closed mode: node a fails and the other three pass.
- A cycle with L holding only the path ball: valid in closed mode once n ≥ 2r + 2.
- With the same L, a path with both endpoints open: valid at m = 2, invalid at m = 0.

Property tests:

- Renumbering nodes at random never changes a verdict.
- The pynauty canonical form and the NetworkX hash-plus-VF2 path agree on random balls.
- On core nodes, the embedding check and the canonical lookup agree.
- A ruleset extracted from a reference graph accepts that graph, and survives a save and reload unchanged.
- **Cut test.** Take a known valid closed graph, cut out a random connected piece, and mark every node that lost a neighbor as open. The checker must accept. A rejection means the embedding rules are too strict, the most likely bug in the design.
- Once it exists, the incremental check agrees with the full check after every move.

Sampler ground truth:

- At n up to 7, enumerate every unlabelled graph and keep the valid ones. NetworkX's graph atlas holds all of them.
- Compare against how often the chain produces each. Frequencies should be proportional to n!/|Aut|.
- Rerun this comparison after every later addition to the sampler.

## Build order

Each step is testable before the next one exists, and steps 1 to 3 are already useful on their own.

1. **Ruleset.** JSON loader, load-time checks, ball types with root 0.
2. **Canonical form.** pynauty main path, NetworkX hash-plus-VF2 cross-check.
3. **Ball extraction.** Visible r-ball of a node with distances, plus "all balls in a reference graph".
4. **Pruning.** Visible sub-ball check on L, with a report.
5. **Embedding check.** VF2 matcher with the distance-window and closed-degree rules, memoized.
6. **Validity checker.** Classification, per-node check, report.
7. **Enumeration baseline.** All valid graphs at very small n, for ground truth.
8. **Sampler, version 1.** One chain, closed mode, full recheck.

After step 3 a ruleset can be extracted from any graph and round-tripped through a file.

## When to add complexity

Each addition waits for an observable symptom.

| Symptom | Add |
| --- | --- |
| The full recheck dominates runtime, expected past n of about 30 | Incremental energy: recheck only nodes within distance r of a changed edge |
| Profiling shows canonical forms dominating the recheck | Cheap fingerprint first: reject on layer sizes and per-layer degrees, canonicalize only on a match |
| Rechecks are still the bottleneck after fingerprints, or the Rust port begins | Incremental neighborhood hashes: reject a node from its level-r hash with no ball extraction |
| Nearly every proposed move is rejected | Degree-preserving edge swaps alongside add and remove |
| Independent runs disagree, or the chain never reaches energy 0 on a ruleset known to be realizable | Parallel tempering: several copies at different β with swaps |
| A ruleset has no closed solutions at reachable sizes | Open nodes: m above 0, the flag-swap move, and the m sweep |
| Open-mode samples keep showing boundaries that cannot be extended | The pairwise agreement pass |
| n beyond roughly 1,000 is needed | Validity-preserving local moves mined from L, and a Rust inner loop |

**Fast rejection, in detail.** Most proposed moves produce invalid balls, so most checks should fail before any canonical form is computed. Neither method below changes correctness.

- **Fingerprint.** Compute the layer sizes plus the sorted degree sequence of each layer, and precompute the same for every type in L. A ball whose fingerprint is absent fails immediately.
- **Incremental neighborhood hashes.** Every node keeps hashes for levels 0 to r. Level 0 is a constant. Each higher level hashes the node's own level-below hash with the sorted level-below hashes of its neighbors.
- A node's level-r hash is determined by its ball, so the allowed level-r values are precomputed from L. A node whose hash is not among them is invalid.
- An edge change alters level-k hashes only within distance k of its endpoints, so the hashes are maintained by a small local update per move.
- The hash is a filter, not a full test. It misses edges between two nodes both at distance r, and different balls can share a hash. Nodes that pass still get the exact check.
- Both filters apply to core nodes only. Open and shadow nodes see incomplete balls, so their fingerprints and hashes need not match any type.

## Scaling expectations and known limits

The Python prototype should reach n of about 50 to 200 for sampling that can be trusted. These figures are estimates, not benchmarks, and the ruleset matters more than the language.

Cost model at max degree 3 and r = 2:

- A ball is about 10 nodes, so one edge change means about 20 rechecks.
- A sweep is one attempted move per edge, about 1.5n moves.
- Tempering multiplies the work by the number of copies, typically 10 to 30.
- The number of sweeps between independent samples is unknown and depends entirely on the ruleset.

| Regime | What limits n | Expected reach |
| --- | --- | --- |
| Loose ruleset: many solutions, easy to move between | Compute | Python to about 1,000 for a few samples. A Rust inner loop should add two to three orders of magnitude |
| Rigid ruleset: few solutions, far apart | Mixing, which can grow exponentially with n | Possibly a few dozen nodes in any language |
| Unrealizable at this n and m | Nothing to find | Indistinguishable from "too slow" without enumeration at small n |

**Finding the wall.** For each ruleset, measure time to first valid state and the spacing between independent samples at n = 20, 40, 80 and 160. Polynomial growth can be extrapolated. Exponential growth says where to stop.

**How large is useful.** Expanding, tree-like graphs have a diameter near 10 at n = 1,000. Lattice-like graphs have side length n^(1/d): 1,000 nodes is about 30 per side in 2D and under 6 in 4D.

**Reaching millions of nodes.** Raw speed is not the obstacle. Exact energy 0 and tempering both break at that size. Two changes make it feasible for some rulesets:

- Validity-preserving moves: local rewrites that swap one valid configuration for another with identical surroundings, mined from L. The chain never leaves the valid set.
- Tolerating sparse defects: run slightly warm and treat the rare violated node like an open node.

Such moves may not exist or may not connect all valid states, so the fast sampler is validated against this design's sampler at small n.

**Limits that do not go away.**

- Realizability is undecidable in general. The pre-checks and the search give evidence, not proof, unless a closed solution is found.
- A valid patch can be a dead end that extends no further. A deeper core means more trust, never certainty.
- Uniform sampling is guaranteed only in equilibrium, and equilibrium cannot be certified.
