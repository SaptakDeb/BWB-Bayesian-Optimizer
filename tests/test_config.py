import pytest

from bwbopt.config import ConfigError, load_config
from conftest import ROOT


def test_loads_and_validates(cfg):
    assert len(cfg.geometry.sections) == 5
    assert len(cfg.geometry.airfoil_stations) == 6
    assert cfg.objective.mode == "multi"
    assert cfg.flow.analysis_type == "incompressible"


def test_override_objective_and_analysis():
    c = load_config(ROOT / "config" / "config.yaml", ["objective.mode=max_ld", "flow.analysis_type=compressible"])
    assert c.objective.mode == "max_ld" and c.flow.analysis_type == "compressible"


def test_rejects_bad_values():
    with pytest.raises(ConfigError):
        load_config(ROOT / "config" / "config.yaml", ["objective.mode=nonsense"])
    with pytest.raises(ConfigError):
        load_config(ROOT / "config" / "config.yaml", ["geometry.airfoil_stations=['0012']"])
    with pytest.raises(ConfigError):
        load_config(ROOT / "config" / "config.yaml", ["geometry.origin=[0, 1, 0]"])


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("flow: {speeed: 3}\n")
    with pytest.raises(ConfigError):
        load_config(p)
