"""Geometry generation: wing surface (OpenVSP or analytic) + CFD enclosure."""
from __future__ import annotations

from pathlib import Path

from ..config import GeometryCfg


def build_wing_surface(g: GeometryCfg, out_dir: Path) -> Path:
    """Dispatch on `geometry.backend` and return the path of the wing STL."""
    if g.backend == "openvsp":
        from .vsp_backend import build_wing_stl
    elif g.backend == "analytic":
        from .analytic_backend import build_wing_stl
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown geometry backend '{g.backend}'")
    return build_wing_stl(g, Path(out_dir))
