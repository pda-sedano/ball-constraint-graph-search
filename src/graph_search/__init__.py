"""Search over connected graphs satisfying radius-r ball constraints.

Modules:

- ``balltype``, ``ruleset``, ``ruleset_io``: ball types and ruleset files.
- ``compiled``: compiling a ruleset with the Rust core (``graph_search_rs``).
- ``report``: issue codes and load reports.
- ``extract``, ``graph_io``, ``check``, ``embed``: single graphs: ball
  extraction, graph files, the validity checker, the embedding check.
- ``enumeration``: the exhaustive baseline (needs geng from nauty).
- ``sampler``: the Metropolis-Hastings sampler and parallel tempering.
- ``draw``: pictures of balls, rulesets and graphs (needs matplotlib).

Import what you need from its module, e.g.
``from graph_search.sampler import temper``. Importing the package itself
loads nothing else.
"""
