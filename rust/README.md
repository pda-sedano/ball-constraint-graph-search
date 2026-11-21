# Rust core

Two crates in one workspace:

- `core/` (`graph-search-core`): plain Rust. Compiling rulesets and canonical form now; search later.
- `py/` (`graph-search-py`): PyO3 bindings, built by maturin into the Python module `graph_search_rs`.

Python's `src/compiled.py` is the only Python module that imports `graph_search_rs`.
Loading, checking and drawing rulesets work without it.

## Build and test

The core depends on nauty (through `nauty-Traces-sys`), compiled from bundled C source on first
build. That needs a C compiler and libclang; on macOS, `xcode-select --install` provides both.

From the repository root, with the venv active:

```sh
pip install maturin                       # once
maturin develop -m rust/py/Cargo.toml     # build the extension into the venv; rerun after Rust changes
pytest                                    # Python tests, including test_compiled.py and test_canonical.py
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
