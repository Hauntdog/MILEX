import pytest
import os
import shutil
from pathlib import Path

# Common fixtures for all tests
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Ensure tests run in an isolated environment by default."""
    # We can add global mocks here if needed
    pass
