"""The sampler: run a chain, compare it with the baseline, save samples.

The chain itself is Rust (``rust/core/src/sample.rs``); this module starts it,
holds what it returns, and does the bookkeeping around it. One chain at a fixed
β, with a full recheck after every move (the design doc's "Sampler" section).
Closed mode by default; ``open_nodes=m`` samples graphs with exactly m open
nodes, which may have neighbors outside the patch::

    from compiled import load_and_compile
    from enumeration import baseline
    from sampler import compare_with_baseline, sample, save_snapshots

    compiled, _ = load_and_compile("rulesets/showcase.json")
    run = sample(compiled, n=8, steps=200_000, beta=1.0, seed=1, snapshot_every=1_000)
    print(run)                                   # acceptance, time valid, graphs seen
    print(compare_with_baseline(run, baseline(compiled, 8).levels[8]))
    save_snapshots(run, "samples/showcase-n8", ruleset_name="showcase")
    save_run(run, "runs/showcase-n8", ruleset_name="showcase")   # the whole run
    run = load_run("runs/showcase-n8")

    # Open mode: patches with 2 open nodes, checked against open_level.
    run = sample(compiled, n=6, steps=200_000, open_nodes=2, seed=1)
    print(compare_with_baseline(run, open_level(compiled, 6, 2)))

Counting. The chain is uniform over valid *numbered* graphs (with their open
sets), so an unlabelled one with automorphism group Aut turns up in proportion
to n!/|Aut|: symmetric graphs have fewer distinct numberings and get less time
(in open mode Aut counts only symmetries mapping open nodes to open nodes).
The numbering is arbitrary, so analysis counts each unlabelled state once by
default, weighting every sample by its |Aut|: ``Run.shares``,
``Run.average`` and ``compare_with_baseline`` take ``count="unlabelled"``
(the default) or ``count="numbered"`` for the chain's own weighting. The
reweighting is exact; its cost is noise when |Aut| varies a lot between
samples, which ``Run.effective_samples`` measures. At large n nearly every
graph has |Aut| = 1 and the two counts agree.

``visits`` counts the steps spent on each valid state up to isomorphism, keyed
by canonical graph6 and canonical open nodes, and ``compare_with_baseline``
checks those shares against the enumeration.

Counting is for small n only (``COUNT_GRAPHS_MAX_NODES``). A graph6 key takes
about n^2/12 bytes, and at large n nearly every valid state is a graph the
chain has not seen before, so the table would grow with the run while telling
nothing. Snapshots and diagnostics work at any n.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx

from extract import number_nodes

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset

    from enumeration import Level

DEFAULT_SWAP_RATE = 0.2
"""In open mode, the share of steps that propose swapping an open node's flag
with a non-open node's, rather than an edge change."""

Count = Literal["unlabelled", "numbered"]
"""How to count states: each unlabelled state once ("unlabelled", weighting a
sample by |Aut|), or each numbered state once ("numbered", the chain's own
weighting)."""

COUNTS: tuple[Count, ...] = ("unlabelled", "numbered")

COUNT_GRAPHS_MAX_NODES = 32
"""Largest n at which ``sample`` counts visits to each graph. On by default up to
here, refused above: a key is at most about 85 bytes at this size, graphs still
recur, and it is well past the n the enumeration baseline can reach."""


@dataclass(frozen=True)
class Snapshot:
    """A valid graph seen at a snapshot step."""

    step: int
    """The chain's step number, burn-in included."""
    edges: tuple[tuple[int, int], ...]
    automorphisms: int
    """Symmetries of the graph that keep the open set in place."""
    open: tuple[int, ...] = ()
    """The open nodes, sorted; empty in closed mode."""

    def to_networkx(self, num_nodes: int) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(self.edges)
        return graph


@dataclass(frozen=True)
class Visit:
    """Steps spent on one valid state up to isomorphism."""

    graph6: str
    """Canonical graph6."""
    open: tuple[int, ...]
    """The open nodes in the canonical numbering; empty in closed mode. With
    ``graph6`` it is equal exactly for isomorphic states."""
    steps: int
    automorphisms: int


@dataclass(frozen=True)
class Run:
    """Everything one chain returned."""

    num_nodes: int
    beta: float
    seed: int
    burn_in: int
    steps: int
    """Recorded steps, after the burn-in."""
    max_degree: int
    """The degree cap: the largest root degree among kept types."""
    open_nodes: int
    """m: the number of open nodes, fixed for the run. 0 is closed mode."""
    swap_rate: float
    proposed_remove: int
    accepted_remove: int
    proposed_add: int
    accepted_add: int
    proposed_swap: int
    accepted_swap: int
    capped: int
    """Additions rejected by the degree cap."""
    no_candidate: int
    """Steps with nothing to remove or add."""
    valid_steps: int
    """Recorded steps at which the graph was valid."""
    counted: bool
    """Whether visits to each graph were counted (``count_graphs``)."""
    visits: tuple[Visit, ...]
    """By time spent, most first; empty if ``counted`` is False."""
    snapshots: tuple[Snapshot, ...]
    trace: tuple[tuple[int, float], ...]
    """``(step, energy)`` every ``trace_every`` recorded steps."""
    final_edges: tuple[tuple[int, int], ...]
    final_open: tuple[int, ...]
    final_energy: float

    @property
    def valid_fraction(self) -> float:
        """Share of recorded steps spent on valid graphs."""
        return self.valid_steps / self.steps if self.steps else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Accepted moves over all steps, burn-in included."""
        total = self.proposed_add + self.proposed_remove + self.proposed_swap
        accepted = self.accepted_add + self.accepted_remove + self.accepted_swap
        return accepted / total if total else 0.0

    def shares(self, count: Count = "unlabelled") -> dict[tuple[str, tuple[int, ...]], float]:
        """Each visited valid state's share, keyed by (canonical graph6, open nodes).

        ``"numbered"`` is the share of valid time the chain spent there.
        ``"unlabelled"`` weights that time by |Aut| and renormalizes, which is
        the share a sampler counting each unlabelled state once would give.
        Empty if the run was never valid.

        Raises:
            ValueError: an unknown ``count``, or the run did not count graphs.
        """
        _check_count(count)
        if not self.counted:
            raise ValueError("the run did not count graphs; sample with count_graphs=True")
        weights = {
            (v.graph6, v.open): v.steps * (v.automorphisms if count == "unlabelled" else 1)
            for v in self.visits
        }
        total = sum(weights.values())
        return {key: w / total for key, w in weights.items()} if total else {}

    def snapshot_weights(self, count: Count = "unlabelled") -> tuple[float, ...]:
        """Each snapshot's weight, summing to 1: proportional to its |Aut| for
        ``"unlabelled"``, equal for ``"numbered"``. Empty with no snapshots."""
        _check_count(count)
        raw = [s.automorphisms if count == "unlabelled" else 1 for s in self.snapshots]
        total = sum(raw)
        return tuple(w / total for w in raw)

    def average(self, statistic: Callable[[Snapshot], float], count: Count = "unlabelled") -> float:
        """The weighted mean of ``statistic`` over the snapshots: its average over
        valid states with each unlabelled state counted once (``"unlabelled"``)
        or each numbered state once (``"numbered"``).

        ``statistic`` takes a ``Snapshot``; ``snapshot.to_networkx(run.num_nodes)``
        gives the graph. Raises ValueError for an unknown ``count`` or a run
        with no snapshots.
        """
        weights = self.snapshot_weights(count)
        if not weights:
            raise ValueError("the run has no snapshots; sample with snapshot_every set")
        return sum(w * statistic(s) for w, s in zip(weights, self.snapshots, strict=True))

    def effective_samples(self, count: Count = "unlabelled") -> float:
        """How many equally weighted snapshots the weighted ones are worth:
        (sum of weights)^2 / sum of squared weights. The number of snapshots for
        ``"numbered"``; fewer for ``"unlabelled"`` when |Aut| varies, which is
        the noise reweighting costs. This ignores correlation between
        snapshots, which lowers it further."""
        weights = self.snapshot_weights(count)
        return 1 / sum(w * w for w in weights) if weights else 0.0

    def __str__(self) -> str:
        graphs = (
            f"{len(self.visits)} distinct valid graphs" if self.counted else "graphs not counted"
        )
        mode = f"{self.open_nodes} open nodes" if self.open_nodes else "closed"
        swaps = (
            f", swaps {_share(self.accepted_swap, self.proposed_swap)}" if self.open_nodes else ""
        )
        lines = [
            (
                f"n = {self.num_nodes}, {mode}, beta = {self.beta}, seed = {self.seed}, "
                f"{self.burn_in} burn-in + {self.steps} recorded steps, "
                f"degree cap {self.max_degree}"
            ),
            (
                f"accepted {self.acceptance_rate:.1%} of moves "
                f"(removals {_share(self.accepted_remove, self.proposed_remove)}, "
                f"additions {_share(self.accepted_add, self.proposed_add)}{swaps}, "
                f"{_share(self.capped, self.proposed_add)} of additions capped)"
            ),
            (
                f"valid at {self.valid_fraction:.1%} of recorded steps; {graphs}; "
                f"{len(self.snapshots)} snapshots"
            ),
        ]
        return "\n".join(lines)


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def _check_count(count: str) -> None:
    if count not in COUNTS:
        raise ValueError(f"count must be one of {COUNTS}, got {count!r}")


def default_start_edges(compiled: CompiledRuleset, num_nodes: int) -> int:
    """Edges for the random start: about n times the kept types' mean root degree, halved."""
    degrees = [compiled.degrees(i)[0] for i in compiled.kept_indices]
    if not degrees:
        return 0
    return round(num_nodes * sum(degrees) / len(degrees) / 2)


def sample(
    compiled: CompiledRuleset,
    n: int,
    steps: int,
    *,
    beta: float = 1.0,
    seed: int = 0,
    burn_in: int = 0,
    snapshot_every: int | None = None,
    count_graphs: bool | None = None,
    trace_every: int | None = None,
    start: nx.Graph | Iterable[tuple[int, int]] | Literal["random", "empty"] = "random",
    open_nodes: int = 0,
    swap_rate: float = DEFAULT_SWAP_RATE,
    start_open: Iterable[Hashable] | None = None,
) -> Run:
    """Run one chain on n nodes, closed or with exactly ``open_nodes`` open nodes.

    Args:
        compiled: the ruleset. Its kept types set the energy and the degree cap.
        n: number of nodes.
        steps: steps to record, after ``burn_in`` unrecorded ones.
        beta: inverse temperature. Valid graphs are sampled uniformly at any
            β; it changes how often the chain is valid and how freely it moves.
        seed: the run is reproducible from it.
        snapshot_every: keep the graph every this many recorded steps, if it
            is valid then (fixed steps, so the snapshots are unbiased).
        count_graphs: count the steps spent on each unlabelled valid graph,
            for ``compare_with_baseline``. None (the default) counts when n is
            at most ``COUNT_GRAPHS_MAX_NODES``; True above that is refused.
        trace_every: record the energy every this many recorded steps.
        start: ``"random"`` for ``default_start_edges`` random edges under
            the cap, ``"empty"``, or a graph: a NetworkX graph on n nodes
            (renumbered in its own order) or edges on ``0..n-1``.
        open_nodes: m, the number of open nodes, fixed for the run. Open nodes
            may have unseen neighbors outside the patch, so the chain samples
            patches; 0 (the default) is closed mode.
        swap_rate: in open mode, the probability that a step proposes moving
            an open flag (swapping it with a non-open node) instead of an edge.
        start_open: the first open set, ``open_nodes`` nodes: labels of
            ``start`` if it is a NetworkX graph, else numbers in ``0..n-1``.
            A random set by default.

    Raises:
        ValueError: no type is kept, β is negative, ``open_nodes`` exceeds n,
            ``swap_rate`` is outside [0, 1], the starting graph or open set
            is the wrong size or invalid, or ``count_graphs`` is True with n
            above ``COUNT_GRAPHS_MAX_NODES``.
        KeyboardInterrupt: on Ctrl-C, checked every 20,000 steps.
    """
    if count_graphs is None:
        count_graphs = n <= COUNT_GRAPHS_MAX_NODES
    elif count_graphs and n > COUNT_GRAPHS_MAX_NODES:
        raise ValueError(
            f"count_graphs is for n up to {COUNT_GRAPHS_MAX_NODES}, not {n}: each key takes "
            f"about n^2/12 bytes and graphs rarely recur at this size; use snapshots instead"
        )
    start_edges: list[tuple[int, int]] | None = None
    random_edges = 0
    open_numbers: list[int] | None = None
    if isinstance(start, str):
        if start == "random":
            random_edges = default_start_edges(compiled, n)
        elif start != "empty":
            raise ValueError(f"start must be 'random', 'empty' or a graph, got {start!r}")
    elif isinstance(start, nx.Graph):
        if start.number_of_nodes() != n:
            raise ValueError(f"starting graph has {start.number_of_nodes()} nodes, not {n}")
        _, number, edges = number_nodes(start)
        start_edges = list(edges)
        if start_open is not None:
            missing = [v for v in start_open if v not in number]
            if missing:
                raise ValueError(f"open nodes not in the starting graph: {missing[:5]}")
            open_numbers = [number[v] for v in start_open]
    else:
        start_edges = [(int(u), int(v)) for u, v in start]
    if start_open is not None and open_numbers is None:
        open_numbers = [int(v) for v in start_open]

    raw = compiled.sample(
        n,
        steps,
        beta=beta,
        seed=seed,
        burn_in=burn_in,
        snapshot_every=snapshot_every,
        count_graphs=count_graphs,
        trace_every=trace_every,
        start_edges=start_edges,
        random_start_edges=random_edges,
        open_nodes=open_nodes,
        swap_rate=swap_rate,
        start_open=open_numbers,
    )
    stats = raw["stats"]
    return Run(
        num_nodes=n,
        beta=beta,
        seed=seed,
        burn_in=burn_in,
        steps=steps,
        max_degree=raw["max_degree"],
        open_nodes=open_nodes,
        swap_rate=swap_rate,
        proposed_remove=stats[0],
        accepted_remove=stats[1],
        proposed_add=stats[2],
        accepted_add=stats[3],
        proposed_swap=stats[4],
        accepted_swap=stats[5],
        capped=stats[6],
        no_candidate=stats[7],
        valid_steps=raw["valid_steps"],
        counted=count_graphs,
        visits=tuple(
            Visit(g6, tuple(open_set), count, round(aut))
            for g6, open_set, count, aut in raw["visits"]
        ),
        snapshots=tuple(
            Snapshot(step, tuple(map(tuple, edges)), round(aut), tuple(open_set))
            for step, edges, open_set, aut in raw["snapshots"]
        ),
        trace=tuple(raw["trace"]),
        final_edges=tuple(map(tuple, raw["final_edges"])),
        final_open=tuple(raw["final_open"]),
        final_energy=raw["final_energy"],
    )


def canonical_key(
    graph: nx.Graph | str, open_nodes: Iterable[Hashable] = ()
) -> tuple[str, tuple[int, ...], int]:
    """(canonical graph6, canonical open nodes, |Aut| keeping the open set in place)
    of a NetworkX graph (open nodes by label) or a graph6 line (open nodes by
    number). Equal for two states exactly when an isomorphism maps one to the
    other and open nodes to open nodes: the key ``Run.visits`` uses.
    """
    import graph_search_rs

    if isinstance(graph, str):
        num_nodes, edges = graph_search_rs.parse_graph6(graph)
        open_numbers = [int(v) for v in open_nodes]
    else:
        labels, number, edges = number_nodes(graph)
        num_nodes = len(labels)
        open_numbers = [number[v] for v in open_nodes]
    key, open_set, automorphisms = graph_search_rs.canonical_open_graph6(
        num_nodes, list(edges), open_numbers
    )
    return key, tuple(open_set), round(automorphisms)


def canonical_graph6(graph: nx.Graph | str) -> tuple[str, int]:
    """(canonical graph6, |Aut|) of a NetworkX graph or a graph6 line, from nauty.

    Equal strings exactly for isomorphic graphs: the key ``Run.visits`` uses.
    """
    import graph_search_rs

    if isinstance(graph, str):
        num_nodes, edges = graph_search_rs.parse_graph6(graph)
    else:
        labels, _, edges = number_nodes(graph)
        num_nodes = len(labels)
    key, automorphisms = graph_search_rs.canonical_graph6(num_nodes, list(edges))
    return key, round(automorphisms)


@dataclass(frozen=True)
class Row:
    graph6: str
    """Canonical graph6."""
    open: tuple[int, ...]
    """Canonical open nodes; empty in closed mode."""
    automorphisms: int
    expected: float
    """The share this state should have: equal for every state when counting
    unlabelled states, (1/|Aut|) normalized when counting numbered ones."""
    observed: float
    """``Run.shares`` for this state, in the same count."""


@dataclass(frozen=True)
class Comparison:
    """A run's time on each valid graph against the enumeration baseline."""

    num_nodes: int
    open_nodes: int
    count: Count
    """How states were counted: each unlabelled one once, or each numbering."""
    valid_steps: int
    rows: tuple[Row, ...]
    """One per baseline graph, most expected first."""
    unexpected: tuple[Visit, ...]
    """Graphs the chain counted as valid that the baseline does not list. Any
    is a bug in one or the other."""
    unexpected_share: float
    """Their combined observed share."""

    @property
    def total_variation(self) -> float:
        """Half the summed absolute differences between expected and observed
        shares: 0 is a perfect match, 1 no overlap."""
        if not self.valid_steps:
            return 0.0
        mismatch = sum(abs(r.expected - r.observed) for r in self.rows)
        return (mismatch + self.unexpected_share) / 2

    def __str__(self) -> str:
        mode = f", {self.open_nodes} open" if self.open_nodes else ""
        counting = (
            "each unlabelled state counted once"
            if self.count == "unlabelled"
            else "numbered states (the chain's own weighting)"
        )
        lines = [
            (
                f"n = {self.num_nodes}{mode}: {self.valid_steps} valid steps over "
                f"{len(self.rows)} baseline graphs, {counting}, "
                f"total variation {self.total_variation:.3f}"
            ),
            f"{'graph6':<16} {'open':<12} {'|Aut|':>6} {'expected':>9} {'observed':>9}",
        ]
        for row in self.rows:
            open_text = ",".join(map(str, row.open)) or "-"
            lines.append(
                f"{row.graph6:<16} {open_text:<12} {row.automorphisms:>6} "
                f"{row.expected:>9.3f} {row.observed:>9.3f}"
            )
        for visit in self.unexpected:
            open_text = ",".join(map(str, visit.open)) or "-"
            lines.append(
                f"{visit.graph6:<16} {open_text:<12} {visit.automorphisms:>6}  not in the baseline"
            )
        return "\n".join(lines)


def compare_with_baseline(run: Run, level: Level, count: Count = "unlabelled") -> Comparison:
    """Compare ``run.visits`` with the baseline's valid graphs at the same n.

    ``level`` comes from ``enumeration.baseline`` for a closed run, or from
    ``enumeration.open_level`` with the run's number of open nodes. Counting
    each unlabelled state once (the default), every valid state should have
    the same share; counting numbered states, each should have a share
    proportional to its numberings, n!/|Aut|. Both test the same chain; the
    numbered count is the less noisy test, the unlabelled one the answer to
    "is every graph equally likely". The run must have counted graphs
    (``Run.counted``).

    Raises:
        ValueError: the run and the level differ in n or in open nodes, the
            run did not count graphs, or ``count`` is unknown.
    """
    _check_count(count)
    if level.n != run.num_nodes:
        raise ValueError(f"run is for n = {run.num_nodes}, baseline level for n = {level.n}")
    if level.open_nodes != run.open_nodes:
        raise ValueError(f"run has {run.open_nodes} open nodes, baseline level {level.open_nodes}")
    observed = run.shares(count)
    # geng's closed-mode lines are not in canonical numbering; open-mode levels
    # already are, and canonicalizing them again changes nothing.
    keyed = [canonical_key(valid.graph6, valid.open) for valid in level.valid]
    weights = [1.0 if count == "unlabelled" else 1 / aut for _, _, aut in keyed]
    total_weight = sum(weights)
    rows = sorted(
        (
            Row(
                graph6=key,
                open=open_set,
                automorphisms=aut,
                expected=weight / total_weight,
                observed=observed.get((key, open_set), 0.0),
            )
            for (key, open_set, aut), weight in zip(keyed, weights, strict=True)
        ),
        key=lambda row: (-row.expected, row.graph6, row.open),
    )
    known = {(key, open_set) for key, open_set, _ in keyed}
    unexpected = tuple(v for v in run.visits if (v.graph6, v.open) not in known)
    return Comparison(
        num_nodes=run.num_nodes,
        open_nodes=run.open_nodes,
        count=count,
        valid_steps=run.valid_steps,
        rows=tuple(rows),
        unexpected=unexpected,
        unexpected_share=sum(observed[(v.graph6, v.open)] for v in unexpected),
    )


def save_snapshots(
    run: Run,
    directory: str | PathLike[str],
    ruleset_name: str | None = None,
    metadata: dict[str, Hashable] | None = None,
) -> list[Path]:
    """Write each snapshot as a graph file, ``step-<step>.safetensors``, with
    its open set in the file's ``open`` tensor.

    The metadata records the ruleset name, n, the number of open nodes, β,
    seed, the chain step and |Aut|, plus anything in ``metadata``, whose keys
    must not be those (or ``source`` or ``energy``). Returns the paths
    written.

    Raises:
        ValueError: ``metadata`` uses a reserved key.
    """
    from graph_io import save_graph

    _check_metadata_keys(metadata)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    width = len(str(run.burn_in + run.steps))
    paths = []
    for snapshot in run.snapshots:
        path = folder / f"step-{snapshot.step:0{width}d}.safetensors"
        save_graph(
            path,
            snapshot.to_networkx(run.num_nodes),
            open_nodes=snapshot.open,
            metadata={
                "source": "sampler",
                "ruleset": ruleset_name,
                "num_nodes": run.num_nodes,
                "open_nodes": run.open_nodes,
                "beta": run.beta,
                "seed": run.seed,
                "step": snapshot.step,
                "automorphisms": snapshot.automorphisms,
                **(metadata or {}),
            },
        )
        paths.append(path)
    return paths


RUN_FORMAT = 1
"""Version of the folder layout ``save_run`` writes."""

RESERVED_METADATA = frozenset(
    {
        "source",
        "ruleset",
        "num_nodes",
        "open_nodes",
        "beta",
        "seed",
        "step",
        "automorphisms",
        "energy",
    }
)
"""Keys the sampler writes into graph files itself; user metadata may not use them."""


def _check_metadata_keys(metadata: dict[str, Hashable] | None) -> None:
    clashes = sorted(RESERVED_METADATA.intersection(metadata or {}))
    if clashes:
        raise ValueError(f"metadata keys {clashes} are reserved for the sampler's own records")


RUN_FILE = "run.json"
FINAL_FILE = "final.safetensors"
SNAPSHOT_FOLDER = "snapshots"


def save_run(
    run: Run,
    folder: str | PathLike[str],
    ruleset_name: str | None = None,
    metadata: dict[str, Hashable] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a whole run in one folder, for ``load_run`` to read back.

    The folder holds:

    - ``run.json``: the settings, move statistics, visit counts, energy trace,
      final energy, the snapshot list, and ``ruleset_name`` and ``metadata``
      if given. Written last, so a folder with it is a complete save.
    - ``final.safetensors``: the chain's final graph and open set, as a graph
      file, since at large n its edge list would bloat the JSON.
    - ``snapshots/``: the snapshots as graph files, as ``save_snapshots``
      writes them.

    ``metadata`` must be JSON-serializable and may not use the keys in
    ``RESERVED_METADATA``. Returns the folder.

    Raises:
        FileExistsError: the folder already holds a saved run and
            ``overwrite`` is False. With ``overwrite``, the old run's files
            are removed first (only those a save writes).
        ValueError: ``metadata`` uses a reserved key.
    """
    from graph_io import save_graph

    _check_metadata_keys(metadata)
    root = Path(folder)
    record_path = root / RUN_FILE
    final_path = root / FINAL_FILE
    snapshot_dir = root / SNAPSHOT_FOLDER
    old_snapshots = sorted(snapshot_dir.glob("step-*.safetensors")) if snapshot_dir.is_dir() else []
    if record_path.exists() or final_path.exists() or old_snapshots:
        if not overwrite:
            raise FileExistsError(
                f"{root} already holds a saved run; pass overwrite=True to replace it"
            )
        # The record goes first, so a half-replaced folder never looks complete.
        record_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        for path in old_snapshots:
            path.unlink()
    root.mkdir(parents=True, exist_ok=True)

    snapshot_paths = save_snapshots(run, snapshot_dir, ruleset_name, metadata)
    final = nx.Graph()
    final.add_nodes_from(range(run.num_nodes))
    final.add_edges_from(run.final_edges)
    save_graph(
        final_path,
        final,
        open_nodes=run.final_open,
        metadata={
            "source": "sampler final state",
            "ruleset": ruleset_name,
            "step": run.burn_in + run.steps,
            "energy": run.final_energy,
            **(metadata or {}),
        },
    )

    record = {
        "format": RUN_FORMAT,
        "ruleset": ruleset_name,
        "metadata": dict(metadata or {}),
        "settings": {
            "num_nodes": run.num_nodes,
            "beta": run.beta,
            "seed": run.seed,
            "burn_in": run.burn_in,
            "steps": run.steps,
            "max_degree": run.max_degree,
            "open_nodes": run.open_nodes,
            "swap_rate": run.swap_rate,
        },
        "moves": {
            "proposed_remove": run.proposed_remove,
            "accepted_remove": run.accepted_remove,
            "proposed_add": run.proposed_add,
            "accepted_add": run.accepted_add,
            "proposed_swap": run.proposed_swap,
            "accepted_swap": run.accepted_swap,
            "capped": run.capped,
            "no_candidate": run.no_candidate,
        },
        "valid_steps": run.valid_steps,
        "counted": run.counted,
        "visits": [[v.graph6, list(v.open), v.steps, v.automorphisms] for v in run.visits],
        "trace": [[step, energy] for step, energy in run.trace],
        "final_energy": run.final_energy,
        "snapshots": [
            {
                "file": path.relative_to(root).as_posix(),
                "step": snapshot.step,
                "automorphisms": snapshot.automorphisms,
            }
            for path, snapshot in zip(snapshot_paths, run.snapshots, strict=True)
        ],
    }
    partial = root / (RUN_FILE + ".part")
    partial.write_text(json.dumps(record, indent=1))
    os.replace(partial, record_path)
    return root


def load_run(folder: str | PathLike[str]) -> Run:
    """Read a run saved by ``save_run``.

    Raises:
        FileNotFoundError: the folder has no ``run.json`` (not a complete
            save), or a graph file it lists is missing.
        ValueError: an unknown format version, a missing entry, or a graph
            file that is invalid or does not match the record.
    """
    root = Path(folder)
    record_path = root / RUN_FILE
    if not record_path.exists():
        raise FileNotFoundError(f"{record_path} not found: {root} is not a complete saved run")
    record = json.loads(record_path.read_text())
    if record.get("format") != RUN_FORMAT:
        raise ValueError(
            f"{record_path}: format {record.get('format')!r}, this code reads {RUN_FORMAT}"
        )
    try:
        settings, moves = record["settings"], record["moves"]
        n = settings["num_nodes"]
        snapshots = []
        for entry in record["snapshots"]:
            graph_file = _load_run_graph(root / entry["file"], n)
            if graph_file.metadata.get("step") != entry["step"]:
                raise ValueError(
                    f"{root / entry['file']}: step {graph_file.metadata.get('step')}, "
                    f"record says {entry['step']}"
                )
            snapshots.append(
                Snapshot(
                    entry["step"], graph_file.edges, entry["automorphisms"], graph_file.open_nodes
                )
            )
        final = _load_run_graph(root / FINAL_FILE, n)
        return Run(
            num_nodes=n,
            beta=settings["beta"],
            seed=settings["seed"],
            burn_in=settings["burn_in"],
            steps=settings["steps"],
            max_degree=settings["max_degree"],
            open_nodes=settings["open_nodes"],
            swap_rate=settings["swap_rate"],
            proposed_remove=moves["proposed_remove"],
            accepted_remove=moves["accepted_remove"],
            proposed_add=moves["proposed_add"],
            accepted_add=moves["accepted_add"],
            proposed_swap=moves["proposed_swap"],
            accepted_swap=moves["accepted_swap"],
            capped=moves["capped"],
            no_candidate=moves["no_candidate"],
            valid_steps=record["valid_steps"],
            counted=record["counted"],
            visits=tuple(
                Visit(g6, tuple(open_set), steps, aut)
                for g6, open_set, steps, aut in record["visits"]
            ),
            snapshots=tuple(snapshots),
            trace=tuple((step, energy) for step, energy in record["trace"]),
            final_edges=final.edges,
            final_open=final.open_nodes,
            final_energy=record["final_energy"],
        )
    except KeyError as missing:
        raise ValueError(f"{record_path}: missing entry {missing}") from None


def _load_run_graph(path: Path, num_nodes: int):
    """A graph file of a saved run, checked and matched to the run's n."""
    from graph_io import load_graph

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    graph_file, report = load_graph(path)
    if graph_file is None:
        raise ValueError(f"{path} is not a valid graph file:\n{report}")
    if graph_file.num_nodes != num_nodes:
        raise ValueError(f"{path} has {graph_file.num_nodes} nodes, the run {num_nodes}")
    return graph_file
