import pytest

from bwbopt.geometry import build_wing_surface
from bwbopt.geometry.analytic_backend import build_mesh
from bwbopt.geometry.enclosure import build_enclosure
from bwbopt.planform import compute_planform


def test_wing_is_watertight_solid(cfg):
    m = build_mesh(cfg.geometry)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    assert m.bounds[1][1] == pytest.approx(compute_planform(cfg.geometry).half_span)


@pytest.mark.parametrize("mode", ["half", "full"])
def test_enclosure_patches_and_volume(cfg, tmp_path, mode):
    cfg.geometry.symmetry_mode = mode
    pf = compute_planform(cfg.geometry)
    wing = build_wing_surface(cfg.geometry, tmp_path)
    rep = build_enclosure(wing, cfg.geometry, pf, cfg.domain, tmp_path / "domain.stl")
    assert rep.watertight
    for p in ("inlet", "outlet", "farfield", "wing"):
        assert rep.patch_faces[p] > 0
    assert (rep.patch_faces["symmetry"] > 0) == (mode == "half")
    lo, hi = rep.bounds.lo, rep.bounds.hi
    box = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    wing_vol = build_mesh(cfg.geometry, with_symmetry_mirror=(mode == "full")).volume
    assert rep.volume == pytest.approx(box - wing_vol, rel=1e-6)
    text = (tmp_path / "domain.stl").read_text()
    assert text.count("solid ") >= 4 and "solid wing" in text and "solid inlet" in text
