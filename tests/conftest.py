from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
RULESETS_DIR = TESTS_DIR.parent / "rulesets"


@pytest.fixture
def fixtures_dir() -> Path:
    """Deliberately broken (or edge-case) ruleset files, one problem each."""
    return FIXTURES_DIR


@pytest.fixture
def rulesets_dir() -> Path:
    """Example rulesets that should load cleanly."""
    return RULESETS_DIR
