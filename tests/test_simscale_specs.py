import pytest

sdk = pytest.importorskip("simscale_sdk")

from bwbopt.atmosphere import flow_conditions
from bwbopt.cfd import simscale_specs as S
from bwbopt.cfd.base import reference_values
from bwbopt.cfd.mesh_requirements import plan_mesh
from bwbopt.planform import compute_planform


def _setup(cfg):
    pf = compute_planform(cfg.geometry)
    fl = flow_conditions(0, cfg.flow.speed, cfg.flow.aoa_deg)
    ents = {k: [f"{k}_1"] for k in S.ENTITY_KEYS}
    return pf, fl, reference_values(cfg, pf, fl), plan_mesh(cfg, pf, fl), ents


@pytest.mark.parametrize("analysis,cls", [("incompressible", "Incompressible"), ("compressible", "Compressible")])
def test_flow_model_serialises(cfg, analysis, cls):
    cfg.flow.analysis_type = analysis
    pf, fl, ref, plan, ents = _setup(cfg)
    model = S.build_flow_model(cfg, pf, fl, ref, ents)
    assert type(model).__name__ == cls and model.turbulence_model == "KOMEGASST"
    d = sdk.ApiClient().sanitize_for_serialization(model)
    types = [b["type"] for b in d["boundaryConditions"]]
    assert "SYMMETRY" in types and "VELOCITY_INLET_V3" in types


def test_resolved_vs_wall_function(cfg):
    pf, fl, ref, plan, ents = _setup(cfg)
    walls = lambda m: [b for b in m.boundary_conditions if getattr(b, "name", "") == "Wing wall"][0]
    assert walls(S.build_flow_model(cfg, pf, fl, ref, ents)).velocity.turbulence_wall == "FULL_RESOLUTION"
    cfg.mesh.wall_treatment = "wall_function"
    assert walls(S.build_flow_model(cfg, pf, fl, ref, ents)).velocity.turbulence_wall == "WALL_FUNCTION"


def test_mesh_model_has_layers_matching_plan(cfg):
    pf, fl, ref, plan, ents = _setup(cfg)
    m = S.build_mesh_model(cfg, plan, ents, {"near_body": "a", "wake": "b"})
    layers = [r for r in m.refinements if type(r).__name__ == "LayerAdditionHexDominantSnappy"][0]
    assert layers.layers == plan.layers and layers.first_layer_thickness == pytest.approx(plan.first_layer)
