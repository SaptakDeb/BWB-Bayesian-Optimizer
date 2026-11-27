
import pytest

from bwbopt.atmosphere import flow_conditions
from bwbopt.constraints import check_geometry
from bwbopt.parametrisation import DesignSpace
from bwbopt.planform import compute_planform


def test_area_and_span(cfg):
    pf = compute_planform(cfg.geometry)
    g = cfg.geometry
    chords = [g.root_chord] + [s.tip_chord for s in g.sections]
    area = sum(0.5 * (chords[i] + chords[i + 1]) * s.span for i, s in enumerate(g.sections))
    assert pf.area_half == pytest.approx(area)
    assert pf.half_span == pytest.approx(sum(s.span for s in g.sections))
    assert pf.area_full == pytest.approx(2 * area)


def test_rectangular_wing_mac():
    from bwbopt.config import GeometryCfg, Section

    g = GeometryCfg(root_chord=2.0, airfoil_stations=["0012", "0012"],
                    sections=[Section(name="s", span=5.0, tip_chord=2.0, sweep_deg=0.0)])
    pf = compute_planform(g)
    assert pf.mac == pytest.approx(2.0)
    assert pf.aspect_ratio == pytest.approx(10.0 ** 2 / 20.0)


def test_isa_sea_level():
    fc = flow_conditions(0.0, 100.0, 0.0)
    assert fc.density == pytest.approx(1.225, rel=1e-3)
    assert fc.speed_of_sound == pytest.approx(340.3, rel=1e-3)
    assert fc.mach < 0.3


def test_design_space_roundtrip(cfg):
    sp = DesignSpace(cfg)
    x = sp.nominal()
    u = sp.to_unit(x)
    x2 = sp.from_unit(u)
    assert all(x2[k] == pytest.approx(x[k]) for k in x)
    g = sp.apply({**x, "twist_tip": -4.0})
    assert g.sections[4].twist_deg == -4.0
    assert cfg.geometry.sections[4].twist_deg == -2.5     # base config untouched


def test_constraints_flag_violation(cfg):
    sp = DesignSpace(cfg)
    g = sp.apply({**sp.nominal(), "chord_tip": 0.05})
    bad = check_geometry(g, compute_planform(g), cfg.optimisation.constraints)
    assert any("tip chord" in b for b in bad)
