import sys
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def n_seeds():
    """Fixture to provide the number of Monte Carlo seeds."""
    return int(os.environ.get("PYTEST_MONTE_CARLO_SEEDS", 5))
