"""Type stub for the Rust extension. Keep in step with rust/py/src/lib.rs."""

CanonicalKey = tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
"""(layer_sizes, edges): equal keys mean isomorphic balls, root to root. Hashable."""

Removal = tuple[int, int, int, int, list[tuple[int, int]], list[int]]
"""(index, round, node, depth, sub_ball_edges, lost_suppliers); see CompiledRuleset.removals."""

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

def issue_codes() -> list[tuple[str, str]]:
    """Every code compiling can report, as (code, severity) pairs."""
