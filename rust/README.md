# Rust core

Two crates in one workspace:

- `core/` (`graph-search-core`): plain Rust. Compiling rulesets, canonical form, the search graph and
  ball extraction now; the checker and sampler later.
- `py/` (`graph-search-py`): PyO3 bindings, built by maturin into the Python module `graph_search_rs`.

Python's `src/compiled.py` and `src/extract.py` are the only modules that import `graph_search_rs`,
and only when called. Loading, checking and drawing rulesets work without it.

## Build and test

The core depends on nauty (through `nauty-Traces-sys`), compiled from bundled C source on first
build. That needs a C compiler and libclang; on macOS, `xcode-select --install` provides both.

From the repository root, with the venv active:

```sh
pip install maturin                       # once
maturin develop -m rust/py/Cargo.toml     # build the extension into the venv; rerun after Rust changes
pytest                                    # Python tests, including the extension's (test_compiled, _canonical, _extract)
cargo test --manifest-path rust/Cargo.toml                 # Rust tests
```

Add `--release` to `maturin develop` once speed matters; debug builds are much slower.

## Issue codes

`core/src/issue.rs` mirrors the compile codes in `src/report.py`. When adding a code, add it in both
places; `tests/test_compiled.py` checks they match.

## Canonical form

`core/src/canonical.rs` gives each ball a key that is equal exactly for balls isomorphic with root
mapped to root, using nauty's `densenauty` with one color per distance from the root. It is checked
two ways: against brute force on every connected rooted graph with up to 6 nodes (Rust), and
against a NetworkX Weisfeiler–Lehman + VF2 oracle on random and WL-blind balls
(`tests/test_canonical.py`). Keys are not guaranteed stable across nauty versions; never save them.

## Graphs and ball extraction

`core/src/graph.rs` is the graph the search edits: one fixed-size row of neighbor slots per node,
capped at the ruleset's largest root degree (a reference graph uses its own largest degree).
`core/src/extract.rs` takes the radius-r ball around a node (`BallExtractor`, which reuses its
buffers between calls) and counts every ball shape in a graph (`ball_census`). From Python,
`extract.ruleset_from_graph(G, r)` turns any NetworkX graph into a ruleset with counts, and
`extract.extract_ball(G, node, r)` shows one ball. `tests/test_extract.py` checks extraction
against `networkx.ego_graph`.

## Pruning

`core/src/prune.rs` removes ball types that no graph can contain, every time a ruleset compiles.
A node at depth d inside a type sees its whole radius-(r − d) ball, so some type must cut down to
exactly that ball at its root; a type with a node no type matches is removed, in rounds, until
nothing changes. Pruned types stay in the compiled ruleset (indices never shift): `kept_indices`
lists the survivors, and `compiled.removals(c)` in Python says, for each removed type, the round,
the node whose sub-ball had no match, that sub-ball, and which removed types used to supply it.
Compiling reports each removal as a `pruned_type` warning, and `unrealizable` if nothing is left.
`tests/test_prune.py` checks the result against a direct NetworkX version of the rule, and checks
that the balls of any real graph are never pruned.

## Embedding check

`core/src/embed.rs` checks open and shadow nodes, whose visible ball may be incomplete: it must
embed into some kept type as a rooted induced subgraph, under the design doc's distance-window and
closed-degree rules. `EmbedQuery::new` builds the query from a `VisibleBall` plus each node's open
flag and distance to the open set (`Graph::distances_from`); `EmbeddingChecker::compatible` returns
the kept types it fits. The matcher backtracks over visible nodes in BFS order, giving interchangeable
"twin" nodes increasing images, and falls back to constraint propagation after `STEP_BUDGET` steps.
Both choices, and leaving out a memo, came from a benchmark (see the design doc). From Python,
`embed.compatible_types(compiled, G, open_nodes)` checks every node of a NetworkX patch at once;
pass all the nodes in one call, since each call rebuilds the checker. `tests/test_embed.py`
compares it with a NetworkX VF2 oracle that shares none of its code.

## Validity checker

`core/src/check.rs` decides whether a graph with an open set is valid for a ruleset. `Checker::check`
classifies every node by its distance to the open set (open, shadow, core), looks core nodes up by
canonical key, runs the embedding check on the rest, and returns a `CheckReport`: the verdict, each
node's class and outcome, the failing nodes, components, core fraction and depth, and the energy the
sampler will minimize. Only kept types count, so a core node whose ball is a pruned type fails.
`Checker::check_node` is the per-node function the incremental check will reuse. From Python,
`check.check_graph(compiled, G, open_nodes, max_open)` and `check.check_file(compiled, path)`;
graph files are read and written by `graph_io.py` (safetensors; needs `numpy` and `safetensors`).
`tests/test_check.py` compares the checker node by node with a NetworkX oracle in `tests/oracle.py`.

## Enumeration baseline

`core/src/enumeration.rs` reads a list of graphs in graph6 (`core/src/graph6.rs`), the format
nauty's geng writes, and keeps the ones valid in closed mode, each with its automorphism count
(`canonical::automorphism_count`, from nauty). It uses `Checker::passes`, which stops at the first
failing node. From Python, `enumeration.baseline(compiled, max_n)` runs `geng -c` with degrees
limited to the kept types' root degrees and returns the valid graphs by n, with `numberings`
(n!/|Aut|), the frequency the sampler should match. geng is found through `$GENG` or `PATH`
(`brew install nauty`); the geng tests are skipped without it.

Results are cached under `.cache/` at the repository root (or `$BCGS_CACHE`): geng's raw lists in
`geng/connected_n{n}_deg{lo}-{hi}.g6`, shared by every ruleset, and each ruleset's valid graphs in
`baselines/{key}/n{n}.json`, where the key hashes r and the ruleset's types (labels and order do
not matter). A second run reads the JSON and needs neither geng nor its lists. Checking costs about
5 µs per graph in a release build (`maturin develop --release`), many times slower in a debug build,
so build release before going past n = 9 or so. `tests/test_enumeration.py` compares the baseline
with the NetworkX graph atlas up to 7 nodes.

To look at the results, `draw.draw_baseline(b, title="showcase")` draws the valid graphs in a grid
titled with n and |Aut| (`sizes=[10]` for one n, `limit=None` for all of them; needs the `draw`
extra). Each panel uses whichever of a Tutte drawing (for 3-connected planar graphs), a circle or a
few spring layouts has the fewest edge crossings.

## Sampler

`core/src/sample.rs` is sampler version 1: one Metropolis-Hastings chain on n nodes in closed mode,
at a fixed β, adding or removing one edge per step with the proposal correction, and rechecking
every node after each move. The degree cap is the kept types' largest root degree. The energy is a
trait: `RulesetEnergy` is the design doc's (failing nodes + components − 1), and `ZeroEnergy` makes
the chain uniform over all graphs under the cap, for testing. `Recorder` runs a chain and counts
the steps spent on each valid unlabelled graph (keyed by canonical graph6, from
`canonical::canonical_graph`), takes snapshots at fixed step numbers, and records an energy trace.
Random numbers come from `core/src/rng.rs` (xoshiro256**), so a seed reproduces a run exactly.

From Python, `sampler.sample(compiled, n, steps, beta=..., seed=..., snapshot_every=...)` returns a
`Run` (print it for acceptance, time valid and graphs seen); `compare_with_baseline(run, level)`
checks its time on each graph against n!/|Aut| from the enumeration baseline; and
`save_snapshots(run, folder)` writes the snapshots as graph files with |Aut|, β, seed and step in
the metadata. Counting visits to each graph is on by default up to `COUNT_GRAPHS_MAX_NODES` (32)
and refused above it: each graph6 key takes about n²/12 bytes, and at large n graphs rarely recur. Runs check for Ctrl-C every 20,000 steps. A step costs about 7 µs at n = 10 and 40 µs
at n = 24 in a release build, so build release for real runs.

Whole-graph |Aut| and canonical forms use nauty's sparse mode (`sparsenauty`), which scales with
the number of edges; balls keep the dense mode, which is faster at their size. The Rust tests check
the transition rule's detailed balance exactly on 4 nodes, the chain's distribution with zero and
random energies, and the recorder's counting against a replayed chain. `tests/test_sampler.py`
compares real runs with the baseline on a ruleset whose valid graphs are all the cubic graphs.
