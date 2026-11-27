"""OpenVSP backend: builds the parametric half-wing through the OpenVSP Python API.

Requires the `openvsp` Python package (see README).  Every parameter lookup is
checked, so a wrong parameter name raises immediately instead of silently
leaving a default value in the model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..config import GeometryCfg


def _vsp() -> Any:
    try:
        import openvsp as vsp  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the user's machine
        raise ImportError(
            "The 'openvsp' Python package is not installed. Install it (see README) or set "
            "geometry.backend: analytic in the config to use the built-in lofter."
        ) from exc
    return vsp


def _parm(vsp: Any, geom_id: str, name: str, group: str) -> str:
    pid = vsp.GetParm(geom_id, name, group)
    if not pid:
        raise RuntimeError(f"OpenVSP parameter not found: {group}/{name}")
    return pid


def _set(vsp: Any, geom_id: str, name: str, group: str, value: float) -> None:
    vsp.SetParmVal(_parm(vsp, geom_id, name, group), float(value))


def build_vsp_model(g: GeometryCfg) -> str:
    """Create the wing in the current OpenVSP model and return its geometry id."""
    vsp = _vsp()
    vsp.VSPCheckSetup()
    vsp.VSPRenew()
    vsp.ClearVSPModel()

    n = len(g.sections)
    wing = vsp.AddGeom("WING", "")
    vsp.SetGeomName(wing, g.name)
    # a new wing already owns one section; add the remaining n-1
    for _ in range(n - 1):
        vsp.InsertXSec(wing, 1, vsp.XS_FOUR_SERIES)
    vsp.Update()

    # airfoil of every station (n + 1 cross-sections)
    xsec_surf = vsp.GetXSecSurf(wing, 0)
    for i, code in enumerate(g.airfoil_stations):
        vsp.ChangeXSecShape(xsec_surf, i, vsp.XS_FOUR_SERIES)
    vsp.Update()
    for i, code in enumerate(g.airfoil_stations):
        grp = f"XSecCurve_{i}"
        _set(vsp, wing, "Camber", grp, int(code[0]) / 100.0)
        _set(vsp, wing, "CamberLoc", grp, int(code[1]) / 10.0)
        _set(vsp, wing, "ThickChord", grp, int(code[2:]) / 100.0)
    vsp.Update()

    # sections: span / chords / sweep / twist / dihedral
    for i, sec in enumerate(g.sections, start=1):
        grp = f"XSec_{i}"
        vsp.SetDriverGroup(wing, i, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
        gam = math.radians(sec.dihedral_deg)
        _set(vsp, wing, "Span", grp, sec.span / math.cos(gam))      # config span is the y-projection
        if i == 1:
            _set(vsp, wing, "Root_Chord", grp, g.root_chord)
        _set(vsp, wing, "Tip_Chord", grp, sec.tip_chord)
        _set(vsp, wing, "Sweep_Location", grp, sec.sweep_location)
        _set(vsp, wing, "Sweep", grp, sec.sweep_deg)
        _set(vsp, wing, "Dihedral", grp, sec.dihedral_deg)
        _set(vsp, wing, "Twist_Location", grp, 0.25)
        _set(vsp, wing, "Twist", grp, sec.twist_deg)
        _set(vsp, wing, "SectTess_U", grp, g.tess.span_per_section)
        vsp.Update()

    # placement, incidence, symmetry
    _set(vsp, wing, "X_Rel_Location", "XForm", g.origin[0])
    _set(vsp, wing, "Y_Rel_Location", "XForm", g.origin[1])
    _set(vsp, wing, "Z_Rel_Location", "XForm", g.origin[2])
    _set(vsp, wing, "Y_Rel_Rotation", "XForm", g.root_incidence_deg)
    # half model: no mirror in OpenVSP (the CFD symmetry plane does it); full model: mirror about xz
    _set(vsp, wing, "Sym_Planar_Flag", "Sym", vsp.SYM_XZ if g.symmetry_mode == "full" else 0)

    # watertight solid: flat caps at root and tip
    _set(vsp, wing, "CapUMinOption", "EndCap", vsp.FLAT_END_CAP)
    _set(vsp, wing, "CapUMaxOption", "EndCap", vsp.FLAT_END_CAP)

    # surface tessellation used by the STL export
    _set(vsp, wing, "Tess_W", "Shape", g.tess.chordwise)
    _set(vsp, wing, "LECluster", "WingGeom", g.tess.le_cluster)
    _set(vsp, wing, "TECluster", "WingGeom", g.tess.te_cluster)
    vsp.Update()
    return wing


def build_wing_stl(g: GeometryCfg, out_dir: Path) -> Path:
    vsp = _vsp()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_vsp_model(g)
    vsp.WriteVSPFile(str(out_dir / f"{g.name}.vsp3"), vsp.SET_ALL)
    stl = out_dir / f"{g.name}_wing.stl"
    vsp.ExportFile(str(stl), vsp.SET_ALL, vsp.EXPORT_STL)
    if not stl.is_file() or stl.stat().st_size == 0:
        raise RuntimeError(f"OpenVSP did not produce {stl}")
    return stl
