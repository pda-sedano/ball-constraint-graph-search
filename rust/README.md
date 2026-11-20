# Rust core

Two crates in one workspace:

- `core/` (`graph-search-core`): plain Rust. Compiling rulesets now; search later.
- `py/` (`graph-search-py`): PyO3 bindings, built by maturin into the Python module `graph_search_rs`.

Python's `src/compiled.py` is the only Python module that imports `graph_search_rs`.
Loading, checking and drawing rulesets work without it.

## Build and test

From the repository root, with the venv active:

```sh
pip install maturin                       # once
maturin develop -m rust/py/Cargo.toml     # build the extension into the venv; rerun after Rust changes
pytest                                    # Python tests, including tests/test_compiled.py
cargo test --manifest-path rust/Cargo.toml                 # Rust tests
```

Add `--release` to `maturin develop` once speed matters; debug builds are much slower.

## Issue codes

`core/src/issue.rs` mirrors the compile codes in `src/report.py`. When adding a code, add it in both
places; `tests/test_compiled.py` checks they match.
