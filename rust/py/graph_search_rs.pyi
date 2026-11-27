"""Type stub for the Rust extension. Keep in step with rust/py/src/lib.rs."""

import os
from typing import Any

CanonicalKey = tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
"""(layer_sizes, edges): equal keys mean isomorphic balls, root to root. Hashable."""

Removal = tuple[int, int, int, int, list[tuple[int, int]], list[int]]
"""(index, round, node, depth, sub_ball_edges, lost_suppliers); see CompiledRuleset.removals."""

class CheckReport:
    """The validity checker's report. check.check_graph wraps it with node labels."""

    @property
    def valid(self) -> bool: ...
    @property
    def components(self) -> int: ...
    @property
    def open_count(self) -> int: ...
    @property
    def max_open(self) -> int | None: ...
    @property
    def energy(self) -> int:
        """Failing nodes plus (components - 1)."""
    @property
    def core_fraction(self) -> float: ...
    @property
    def core_depth(self) -> int | None:
        """Largest distance to the open set minus r; None in closed mode."""
    @property
    def failing(self) -> list[int]: ...
    def classes(self) -> list[str]:
        """Per node: "open", "shadow" or "core"."""
    def distances(self) -> list[int | None]: ...
    def outcomes(self) -> list[tuple[str, list[int]]]:
        """Per node: ("matched", [i]), ("pruned", [i]), ("unmatched", []) or ("compatible", [...])."""

class CompiledRuleset:
    """A compiled ruleset, read-only. Ball indices count distinct types after merging
    duplicates, pruned or not; ``kept_indices`` lists those that survive pruning."""

    @property
    def r(self) -> int: ...
    @property
    def source_indices(self) -> list[int]:
        """Position in the file of each distinct ball type."""
    @property
    def max_root_degree(self) -> int:
        """Largest root degree over all types, pruned or not."""
    @property
    def kept_indices(self) -> list[int]:
        """Indices of the types that survive pruning, ascending."""
    def removals(self) -> list[Removal]:
        """Every type pruning removed, by round, then index. compiled.removals wraps these."""
    def __len__(self) -> int: ...
    def shallow_indices(self) -> list[int]: ...
    def layers(self, index: int) -> list[list[int]]: ...
    def degrees(self, index: int) -> list[int]: ...
    def depth(self, index: int) -> int: ...
    def edges(self, index: int) -> list[tuple[int, int]]: ...
    def canonical_key(self, index: int) -> CanonicalKey: ...
    def check(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        open_nodes: list[int],
        max_open: int | None = None,
    ) -> CheckReport:
        """The validity checker. check.check_graph wraps this for NetworkX graphs."""
    def valid_graph6(self, path: str | os.PathLike[str]) -> tuple[int, list[tuple[str, float]]]:
        """Closed-mode check of every graph in a graph6 file: (candidates, [(graph6, |Aut|)])."""
    def sample(
        self,
        num_nodes: int,
        steps: int,
        *,
        beta: float = 1.0,
        seed: int = 0,
        burn_in: int = 0,
        snapshot_every: int | None = None,
        count_graphs: bool = True,
        trace_every: int | None = None,
        start_edges: list[tuple[int, int]] | None = None,
        random_start_edges: int = 0,
        open_nodes: int = 0,
        swap_rate: float = 0.2,
        start_open: list[int] | None = None,
    ) -> dict[str, Any]:
        """One sampler chain with exactly open_nodes open nodes (0 is closed mode).
        sampler.sample wraps this; see it for the keys."""
    def valid_open_graph6(
        self, path: str | os.PathLike[str], m: int
    ) -> tuple[int, int, list[tuple[str, list[int], float]]]:
        """(candidates, open sets checked, [(canonical graph6, open, |Aut|)]) per valid class."""
    def compatible_types(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        open_nodes: list[int],
        centers: list[int],
    ) -> list[list[int]]:
        """Embedding check per center: indices of the kept types its visible ball fits.
        embed.compatible_types wraps this for NetworkX graphs."""
    def index_of(self, edges: list[tuple[int, int]]) -> int | None:
        """Index of the type isomorphic to this ball (pruned or not), or None.
        ValueError if not a valid ball."""

def compile_ruleset(
    r: int, types: list[list[tuple[int, int]]]
) -> tuple[CompiledRuleset | None, list[tuple[str, str, int | None]]]:
    """Compile from r and each type's edges as written. Raises ValueError if r is 0."""

def canonical_key(edges: list[tuple[int, int]]) -> CanonicalKey:
    """Canonical key of the ball with these edges. ValueError if not a valid ball."""

def extract_ball(
    num_nodes: int, edges: list[tuple[int, int]], center: int, r: int
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """The radius-r ball around center: (nodes, depths, edges) in ball numbering. ValueError if invalid."""

def ball_census(
    num_nodes: int, edges: list[tuple[int, int]], r: int
) -> list[tuple[list[tuple[int, int]], int, int]]:
    """Every node's ball grouped by shape: (canonical_edges, count, example_center), most common first."""

def parse_graph6(line: str) -> tuple[int, list[tuple[int, int]]]:
    """(num_nodes, edges) of one graph6 line. ValueError if not graph6."""

def automorphism_count(num_nodes: int, edges: list[tuple[int, int]]) -> float:
    """|Aut| from nauty's sparse mode, exact below 2^53."""

def canonical_graph6(num_nodes: int, edges: list[tuple[int, int]]) -> tuple[str, float]:
    """(canonical graph6, |Aut|): equal strings exactly for isomorphic graphs."""

def canonical_open_graph6(
    num_nodes: int, edges: list[tuple[int, int]], open_nodes: list[int] = ...
) -> tuple[str, list[int], float]:
    """(canonical graph6, canonical open nodes, |Aut| keeping the open set)."""

def issue_codes() -> list[tuple[str, str]]:
    """Every code compiling can report, as (code, severity) pairs."""
