"""The sampler, version 1: run a chain, compare it with the baseline, save samples.

The chain itself is Rust (``rust/core/src/sample.rs``); this module starts it,
holds what it returns, and does the bookkeeping around it. One chain at a fixed
β, closed mode, with a full recheck after every move (the design doc's
"Sampler" section)::

    from compiled import load_and_compile
    from enumeration import baseline
    from sampler import compare_with_baseline, sample, save_snapshots

    compiled, _ = load_and_compile("rulesets/showcase.json")
    run = sample(compiled, n=8, steps=200_000, beta=1.0, seed=1, snapshot_every=1_000)
    print(run)                                   # acceptance, time valid, graphs seen
    print(compare_with_baseline(run, baseline(compiled, 8).levels[8]))
    save_snapshots(run, "samples/showcase-n8", ruleset_name="showcase")

The chain is uniform over valid *numbered* graphs, so an unlabelled graph with
automorphism group Aut turns up in proportion to n!/|Aut|. ``visits`` counts
the steps spent on each unlabelled valid graph, keyed by canonical graph6, and
``compare_with_baseline`` checks those shares against the enumeration.

Counting is for small n only (``COUNT_GRAPHS_MAX_NODES``). A graph6 key takes
about n^2/12 bytes, and at large n nearly every valid state is a graph the
chain has not seen before, so the table would grow with the run while telling
nothing. Snapshots and diagnostics work at any n.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx

from extract import number_nodes

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset

    from enumeration import Level

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

    def to_networkx(self, num_nodes: int) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(self.edges)
        return graph


@dataclass(frozen=True)
class Visit:
    """Steps spent on one unlabelled valid graph."""

    graph6: str
    """Canonical graph6: equal exactly for isomorphic graphs."""
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
    proposed_remove: int
    accepted_remove: int
    proposed_add: int
    accepted_add: int
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
    final_energy: float

    @property
    def valid_fraction(self) -> float:
        """Share of recorded steps spent on valid graphs."""
        return self.valid_steps / self.steps if self.steps else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Accepted moves over all steps, burn-in included."""
        total = self.proposed_add + self.proposed_remove
        return (self.accepted_add + self.accepted_remove) / total if total else 0.0

    def __str__(self) -> str:
        graphs = (
            f"{len(self.visits)} distinct valid graphs" if self.counted else "graphs not counted"
        )
        lines = [
            (
                f"n = {self.num_nodes}, beta = {self.beta}, seed = {self.seed}, "
                f"{self.burn_in} burn-in + {self.steps} recorded steps, "
                f"degree cap {self.max_degree}"
            ),
            (
                f"accepted {self.acceptance_rate:.1%} of moves "
                f"(removals {_share(self.accepted_remove, self.proposed_remove)}, "
                f"additions {_share(self.accepted_add, self.proposed_add)}, "
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
) -> Run:
    """Run one chain on n nodes in closed mode.

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

    Raises:
        ValueError: no type is kept, β is negative, the starting graph has
            the wrong size or breaks the degree cap, or ``count_graphs`` is
            True with n above ``COUNT_GRAPHS_MAX_NODES``.
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
    if isinstance(start, str):
        if start == "random":
            random_edges = default_start_edges(compiled, n)
        elif start != "empty":
            raise ValueError(f"start must be 'random', 'empty' or a graph, got {start!r}")
    elif isinstance(start, nx.Graph):
        if start.number_of_nodes() != n:
            raise ValueError(f"starting graph has {start.number_of_nodes()} nodes, not {n}")
        start_edges = list(number_nodes(start)[2])
    else:
        start_edges = [(int(u), int(v)) for u, v in start]

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
    )
    stats = raw["stats"]
    return Run(
        num_nodes=n,
        beta=beta,
        seed=seed,
        burn_in=burn_in,
        steps=steps,
        max_degree=raw["max_degree"],
        proposed_remove=stats[0],
        accepted_remove=stats[1],
        proposed_add=stats[2],
        accepted_add=stats[3],
        capped=stats[4],
        no_candidate=stats[5],
        valid_steps=raw["valid_steps"],
        counted=count_graphs,
        visits=tuple(Visit(g6, count, round(aut)) for g6, count, aut in raw["visits"]),
        snapshots=tuple(
            Snapshot(step, tuple(map(tuple, edges)), round(aut))
            for step, edges, aut in raw["snapshots"]
        ),
        trace=tuple(raw["trace"]),
        final_edges=tuple(map(tuple, raw["final_edges"])),
        final_energy=raw["final_energy"],
    )


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
    automorphisms: int
    expected: float
    """Share of valid time the chain should spend here: (1/|Aut|) normalized."""
    observed: float


@dataclass(frozen=True)
class Comparison:
    """A run's time on each valid graph against the enumeration baseline."""

    num_nodes: int
    valid_steps: int
    rows: tuple[Row, ...]
    """One per baseline graph, most expected first."""
    unexpected: tuple[Visit, ...]
    """Graphs the chain counted as valid that the baseline does not list. Any
    is a bug in one or the other."""

    @property
    def total_variation(self) -> float:
        """Half the summed absolute differences between expected and observed
        shares: 0 is a perfect match, 1 no overlap."""
        if not self.valid_steps:
            return 0.0
        unexpected = sum(v.steps for v in self.unexpected) / self.valid_steps
        return (sum(abs(r.expected - r.observed) for r in self.rows) + unexpected) / 2

    def __str__(self) -> str:
        lines = [
            (
                f"n = {self.num_nodes}: {self.valid_steps} valid steps over {len(self.rows)} "
                f"baseline graphs, total variation {self.total_variation:.3f}"
            ),
            f"{'graph6':<16} {'|Aut|':>8} {'expected':>9} {'observed':>9}",
        ]
        for row in self.rows:
            lines.append(
                f"{row.graph6:<16} {row.automorphisms:>8} {row.expected:>9.3f} {row.observed:>9.3f}"
            )
        for visit in self.unexpected:
            lines.append(f"{visit.graph6:<16} {visit.automorphisms:>8}  not in the baseline")
        return "\n".join(lines)


def compare_with_baseline(run: Run, level: Level) -> Comparison:
    """Compare ``run.visits`` with the baseline's valid graphs at the same n.

    The expected share of each graph is proportional to its numberings,
    n!/|Aut|. The run must have counted graphs (``Run.counted``).

    Raises:
        ValueError: the run and the level are for different n, or the run did
            not count graphs while it was valid.
    """
    if level.n != run.num_nodes:
        raise ValueError(f"run is for n = {run.num_nodes}, baseline level for n = {level.n}")
    if not run.counted:
        raise ValueError("the run did not count graphs; sample with count_graphs=True")
    keyed = [canonical_graph6(valid.graph6) for valid in level.valid]
    weights = [1 / aut for _, aut in keyed]
    total_weight = sum(weights)
    steps = {v.graph6: v.steps for v in run.visits}
    rows = sorted(
        (
            Row(
                graph6=key,
                automorphisms=aut,
                expected=weight / total_weight,
                observed=steps.get(key, 0) / run.valid_steps if run.valid_steps else 0.0,
            )
            for (key, aut), weight in zip(keyed, weights, strict=True)
        ),
        key=lambda row: (-row.expected, row.graph6),
    )
    known = {key for key, _ in keyed}
    return Comparison(
        num_nodes=run.num_nodes,
        valid_steps=run.valid_steps,
        rows=tuple(rows),
        unexpected=tuple(v for v in run.visits if v.graph6 not in known),
    )


def save_snapshots(
    run: Run,
    directory: str | PathLike[str],
    ruleset_name: str | None = None,
    metadata: dict[str, Hashable] | None = None,
) -> list[Path]:
    """Write each snapshot as a graph file, ``step-<step>.safetensors``.

    The metadata records the ruleset name, n, β, seed, the chain step and
    |Aut|, plus anything in ``metadata``. Returns the paths written.
    """
    from graph_io import save_graph

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    width = len(str(run.burn_in + run.steps))
    paths = []
    for snapshot in run.snapshots:
        path = folder / f"step-{snapshot.step:0{width}d}.safetensors"
        save_graph(
            path,
            snapshot.to_networkx(run.num_nodes),
            metadata={
                "source": "sampler",
                "ruleset": ruleset_name,
                "num_nodes": run.num_nodes,
                "beta": run.beta,
                "seed": run.seed,
                "step": snapshot.step,
                "automorphisms": snapshot.automorphisms,
                **(metadata or {}),
            },
        )
        paths.append(path)
    return paths
