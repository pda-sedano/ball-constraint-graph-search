"""The enumeration baseline: every valid graph at small n, for ground truth.

nauty's geng lists every connected graph on n nodes, up to isomorphism, within
a degree range. The Rust core checks each in closed mode and returns the valid
ones with |Aut|. In equilibrium the sampler visits each valid unlabelled graph
in proportion to n!/|Aut|, so this is what its frequencies are compared with.

Two levels of cache, under ``.cache/`` at the repository root (or
``$BCGS_CACHE``), so nothing is computed twice:

- ``geng/connected_n{n}_deg{lo}-{hi}.g6``: geng's output. It depends only on
  n and the degree range, so every ruleset with the same range shares it.
- ``baselines/{key}/n{n}.json``: one ruleset's valid graphs at one n. The key
  hashes r and every distinct type's edges, so editing a ruleset changes it,
  while relabelling, reordering or duplicating types does not. Bump
  ``BASELINE_FORMAT`` when the checker's rules change.

Every node of a valid closed graph has one of the kept types' root degrees,
so geng only generates graphs with degrees between the smallest and largest.

Open mode has its own ground truth, ``open_level``: every candidate graph with
every set of m open nodes, kept once per isomorphism class (open nodes
mapped to open nodes) with the symmetries that keep the open set in place.
An open node may be missing neighbors, so candidates there have degrees from
1 up to the largest root degree. Its results are not cached (geng's lists
are), since it is meant for the small n the sampler is checked at.

geng comes with nauty: ``brew install nauty`` on macOS, ``apt install nauty``
on Debian and Ubuntu (which name it ``nauty-geng``). Set ``$GENG`` to use a
particular binary.

Example::

    from compiled import load_and_compile
    from enumeration import baseline

    compiled, _ = load_and_compile("rulesets/path-ball-r2.json")
    print(baseline(compiled, max_n=10))   # one valid graph per n >= 6: the cycle
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from graph_search_rs import CompiledRuleset

BASELINE_FORMAT = 1
"""Version of the cached results. Bump when the checker's rules change."""

GENG_NAMES = ("geng", "nauty-geng")


class GengNotFound(RuntimeError):
    """geng is not installed, or not where ``$GENG`` says."""


@dataclass(frozen=True)
class ValidGraph:
    """One valid graph: as geng wrote it in closed mode, or in canonical form
    with its open set in open mode."""

    graph6: str
    automorphisms: int
    """Symmetries of the graph, keeping the open set in place if there is one."""
    open: tuple[int, ...] = ()
    """The open nodes, in ``graph6``'s numbering; empty in closed mode."""

    def to_networkx(self) -> nx.Graph:
        return nx.from_graph6_bytes(self.graph6.encode())

    @property
    def num_nodes(self) -> int:
        return self.to_networkx().number_of_nodes()

    @property
    def numberings(self) -> int:
        """n!/|Aut|: how many numbered graphs (with open sets) this class stands for."""
        return factorial(self.num_nodes) // self.automorphisms


@dataclass(frozen=True)
class Level:
    """All valid graphs on n nodes."""

    n: int
    candidates: int
    """Connected graphs geng generated within the degree range."""
    valid: tuple[ValidGraph, ...]
    open_nodes: int = 0
    """m: every valid graph here has exactly this many open nodes (0 is closed mode)."""


@dataclass(frozen=True)
class Baseline:
    """Valid graphs for one ruleset, n by n."""

    r: int
    degree_range: tuple[int, int] | None
    """Smallest and largest root degree among kept types; None if none is kept."""
    levels: dict[int, Level]

    def counts(self) -> dict[int, int]:
        """Number of valid unlabelled graphs at each n."""
        return {n: len(level.valid) for n, level in self.levels.items()}

    def __str__(self) -> str:
        lines = [f"r = {self.r}, degrees {self.degree_range}", "   n   candidates   valid"]
        for n, level in self.levels.items():
            lines.append(f"{n:4d} {level.candidates:12d} {len(level.valid):7d}")
        return "\n".join(lines)


def cache_dir() -> Path:
    """``$BCGS_CACHE``, or ``.cache`` at the repository root."""
    return Path(os.environ.get("BCGS_CACHE", Path(__file__).resolve().parent.parent / ".cache"))


def find_geng() -> str:
    """The geng binary: ``$GENG`` if set, else the first of geng or nauty-geng on PATH."""
    explicit = os.environ.get("GENG")
    if explicit:
        if shutil.which(explicit):
            return explicit
        raise GengNotFound(f"$GENG is {explicit!r}, which is not an executable")
    for name in GENG_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise GengNotFound(
        "geng not found: install nauty (brew install nauty, or apt install nauty) "
        "or set $GENG to its path"
    )


def degree_range(compiled: CompiledRuleset) -> tuple[int, int] | None:
    """Smallest and largest root degree among the kept types."""
    roots = [compiled.degrees(i)[0] for i in compiled.kept_indices]
    return (min(roots), max(roots)) if roots else None


def ruleset_key(compiled: CompiledRuleset) -> str:
    """A short hash of r and every distinct type's normalized edges."""
    types = sorted(compiled.edges(i) for i in range(len(compiled)))
    payload = json.dumps({"format": BASELINE_FORMAT, "r": compiled.r, "types": types})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def geng_list(n: int, min_degree: int, max_degree: int, root: Path | None = None) -> Path:
    """geng's list of connected graphs on n nodes with degrees in the range,
    generated once and then read from the cache."""
    path = (root or cache_dir()) / "geng" / f"connected_n{n}_deg{min_degree}-{max_degree}.g6"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [find_geng(), "-cq", f"-d{min_degree}", f"-D{max_degree}", str(n)]
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".part") as out:
        try:
            done = subprocess.run(command, stdout=out, stderr=subprocess.PIPE, check=False)
        except BaseException:
            Path(out.name).unlink(missing_ok=True)
            raise
    if done.returncode != 0:
        # geng refuses outright when the edge-count bounds that connectivity
        # and the degree range imply leave no graphs: an empty list.
        if b"impossible" not in done.stderr:
            Path(out.name).unlink(missing_ok=True)
            message = done.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"{' '.join(command)} failed: {message}")
        Path(out.name).write_bytes(b"")
    os.replace(out.name, path)
    return path


def baseline(
    compiled: CompiledRuleset,
    max_n: int,
    min_n: int = 1,
    root: Path | None = None,
    verbose: bool = False,
) -> Baseline:
    """Every valid closed graph on min_n to max_n nodes, reading cached results
    where they exist and caching new ones. ``verbose`` prints a line per n.

    Candidates grow fast with n and the degree range: with degrees 3 to 6
    there are 1.5 million at n = 10, checked at about 5 µs each in a release
    build (``maturin develop --release``); debug builds are far slower.
    """
    root = root or cache_dir()
    degrees = degree_range(compiled)
    results = root / "baselines" / ruleset_key(compiled)
    levels = {}
    for n in range(min_n, max_n + 1):
        path = results / f"n{n}.json"
        started = time.perf_counter()
        cached = path.exists()
        if cached:
            levels[n] = _read_level(path)
        else:
            levels[n] = _compute_level(compiled, n, degrees, root)
            _write_level(path, compiled.r, degrees, levels[n])
        if verbose:
            level = levels[n]
            source = "cached" if cached else f"{time.perf_counter() - started:.1f} s"
            print(f"n = {n}: {level.candidates} candidates, {len(level.valid)} valid ({source})")
    return Baseline(r=compiled.r, degree_range=degrees, levels=levels)


def open_level(
    compiled: CompiledRuleset,
    n: int,
    m: int,
    *,
    graphs: Iterable[nx.Graph] | None = None,
    root: Path | None = None,
) -> Level:
    """Every valid graph on n nodes with exactly m open nodes, once per class.

    Candidates come from geng (connected, degrees 1 up to the largest kept root
    degree; geng's list is cached) or, if given, from ``graphs``, of which only
    those on n nodes are used, e.g. ``nx.graph_atlas_g()`` to avoid needing
    geng up to 7 nodes. Each candidate is checked with every m-subset as its
    open set. The cost is the candidates times C(n, m), so keep n small.
    ``open_level(compiled, n, 0)`` gives the closed-mode valid graphs, in
    canonical form.
    """
    if graphs is None:
        degrees = degree_range(compiled)
        if degrees is None:
            return Level(n, 0, (), open_nodes=m)
        low = 0 if n == 1 else (1 if m > 0 else degrees[0])
        high = min(degrees[1], n - 1) if n > 1 else 0
        if low > high:
            return Level(n, 0, (), open_nodes=m)
        candidates, _, valid = compiled.valid_open_graph6(geng_list(n, low, high, root), m)
    else:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graphs.g6"
            lines = [
                nx.to_graph6_bytes(nx.convert_node_labels_to_integers(g), header=False)
                for g in graphs
                if g.number_of_nodes() == n
            ]
            path.write_bytes(b"".join(lines))
            candidates, _, valid = compiled.valid_open_graph6(path, m)
    return Level(
        n,
        candidates,
        tuple(ValidGraph(g6, round(aut), tuple(open_nodes)) for g6, open_nodes, aut in valid),
        open_nodes=m,
    )


def _compute_level(compiled, n, degrees, root) -> Level:
    if degrees is None:
        return Level(n, 0, ())
    low, high = degrees[0], min(degrees[1], n - 1)
    if low > high:
        return Level(n, 0, ())
    candidates, valid = compiled.valid_graph6(geng_list(n, low, high, root))
    return Level(n, candidates, tuple(ValidGraph(g6, round(aut)) for g6, aut in valid))


def _write_level(path: Path, r: int, degrees, level: Level) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": BASELINE_FORMAT,
        "r": r,
        "degree_range": degrees,
        "n": level.n,
        "candidates": level.candidates,
        "valid": [[g.graph6, g.automorphisms] for g in level.valid],
    }
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_level(path: Path) -> Level:
    document = json.loads(path.read_text(encoding="utf-8"))
    return Level(
        n=document["n"],
        candidates=document["candidates"],
        valid=tuple(ValidGraph(g6, aut) for g6, aut in document["valid"]),
    )
