import sys
from pathlib import Path

# Ensure package import for tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from Causal_Web.engine.fields.density import get_field


@pytest.fixture(autouse=True)
def _clear_density_field() -> None:
    """Reset the global density field before each test."""

    field = get_field()
    field.clear()
    yield
    field.clear()
