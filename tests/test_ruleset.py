import pytest

TODO = pytest.mark.skip(reason="not written yet")


@TODO
def test_beyond_radius_is_error():
    """r = 2 with a path of length 3 from the root gives BEYOND_RADIUS."""


@TODO
def test_shallow_is_warning_and_kept():
    """r = 2 with the single-edge ball: SHALLOW_TYPE warning, report.ok, type still present."""


@TODO
def test_shallow_indices():
    """shallow_indices lists exactly the types with depth < r, in order."""


@TODO
def test_exact_duplicate_warns_once_per_later_copy():
    """Three identical types give two EXACT_DUPLICATE warnings, at indices 1 and 2."""


@TODO
def test_exact_duplicate_ignores_labels():
    """Same structure with different labels still counts as a duplicate."""


@TODO
def test_empty_ruleset_warns():
    """No types gives EMPTY_RULESET as a warning."""


@TODO
@pytest.mark.parametrize("r", [0, -1, True])
def test_bad_r(r):
    """r below 1, or a bool, gives BAD_VALUE."""


@TODO
def test_max_root_degree():
    """Largest root degree across types; used as the sampler's degree cap."""


@TODO
def test_check_ruleset_appends_to_given_report():
    """Passing an existing report appends to it and returns the same object."""
