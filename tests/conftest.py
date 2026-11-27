import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from bwbopt.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    return load_config(ROOT / "config" / "config.yaml",
                       ["geometry.backend=analytic", f"project.workdir={tmp_path / 'run'}"])
