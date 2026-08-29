"""Pytest configuration — add project root to sys.path."""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Pre-import model classes so joblib can deserialize pickled models in any test.
# Without this, joblib looks for FusedCalibratedClassifier in the test module
# (e.g., pytest.__main__) rather than in models.model_suite, and fails.
import models.model_suite  # noqa: F401 -- required for joblib unpickling
