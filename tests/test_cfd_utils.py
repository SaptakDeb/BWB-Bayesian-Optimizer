import math

import pytest

from bwbopt.atmosphere import flow_conditions
from bwbopt.cfd.base import reference_values
from bwbopt.cfd.mesh_requirements import (bl_thickness, first_layer_height, layers_needed, plan_mesh,
                                          stack_thickness)
from bwbopt.cfd.mock_backend import MockBackend
from bwbopt.cfd.results import coefficients_from_tables, parse_csv
from bwbopt.planform import compute_planform


def test_first_layer_scaling():
    nu, u, re = 1.5e-5, 100.0, 1e7
    y1 = first_layer_height(1.0, u, nu, re)
    assert 1e-6 < y1 < 1e-5
    assert first_layer_height(30.0, u, nu, re) == pytest.approx(30 * y1)


def test_layers_cover_boundary_layer():
    first, r = 5e-6, 1.2
    delta = bl_thickness(3.0, 2e7)
    n = layers_needed(first, r, delta)
    assert stack_thickness(first, r, n) >= delta
    assert stack_thickness(first, r, n - 1) < delta


def test_plan_wall_function_needs_fewer_layers(cfg):
    pf = compute_planform(cfg.geometry)
    fl = flow_conditions(0, 100, 8)
    resolved = plan_mesh(cfg, pf, fl)
    cfg.mesh.wall_treatment = "wall_function"
    cfg.mesh.target_y_plus = 30.0
    wf = plan_mesh(cfg, pf, fl)
    assert wf.layers < resolved.layers and wf.first_layer > 20 * resolved.first_layer


def test_force_csv_roundtrip(cfg):
    pf = compute_planform(cfg.geometry)
    fl = flow_conditions(0, 100, cfg.flow.aoa_deg)
    ref = reference_values(cfg, pf, fl)
    res = MockBackend().run(cfg, pf, fl, None, None, "t")
    q = ref.q_inf * ref.area
    a = math.radians(cfg.flow.aoa_deg)
    fx = res.cd * q * math.cos(a) - res.cl * q * math.sin(a)
    fz = res.cd * q * math.sin(a) + res.cl * q * math.cos(a)
    f = "Time,Fx total (N),Fy total (N),Fz total (N)\n" + "\n".join(f"{i},{fx},0,{fz}" for i in range(300))
    m = "Time,Mx,My,Mz\n" + "\n".join(f"{i},0,{res.cm * q * ref.length},0" for i in range(300))
    cl, cd, cm, info = coefficients_from_tables([parse_csv(f)], [parse_csv(m)], fl, ref, 100, {})
    assert (cl, cd, cm) == (pytest.approx(res.cl), pytest.approx(res.cd), pytest.approx(res.cm))
    assert info["force_rel_std"][0] == pytest.approx(0.0, abs=1e-12)
