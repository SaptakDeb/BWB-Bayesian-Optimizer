"""Cheap geometric feasibility checks, run BEFORE any CFD is spent."""
from __future__ import annotations

from typing import List

from .config import ConstraintsCfg, GeometryCfg
from .planform import Planform


def check_geometry(g: GeometryCfg, pf: Planform, c: ConstraintsCfg) -> List[str]:
    """Return a list of human-readable violations (empty list = feasible)."""
    bad: List[str] = []
    if c.half_span_range is not None:
        lo, hi = c.half_span_range
        if not lo <= pf.half_span <= hi:
            bad.append(f"half span {pf.half_span:.3f} m outside [{lo}, {hi}]")
    if c.area_range is not None:
        lo, hi = c.area_range
        if not lo <= pf.area_full <= hi:
            bad.append(f"planform area {pf.area_full:.3f} m^2 outside [{lo}, {hi}]")
    if c.min_tip_chord is not None and pf.stations[-1].chord < c.min_tip_chord:
        bad.append(f"tip chord {pf.stations[-1].chord:.3f} m < {c.min_tip_chord}")
    if c.min_root_thickness_m is not None:
        root = pf.stations[0]
        t = root.chord * root.thickness_ratio
        if t < c.min_root_thickness_m:
            bad.append(f"root thickness {t:.3f} m < {c.min_root_thickness_m}")
    if c.max_le_sweep_deg is not None:
        import math

        for a, b in zip(pf.stations[:-1], pf.stations[1:]):
            sw = math.degrees(math.atan2(b.x_le - a.x_le, b.y - a.y))
            if abs(sw) > c.max_le_sweep_deg:
                bad.append(f"LE sweep {sw:.1f} deg between stations {a.index}-{b.index} exceeds {c.max_le_sweep_deg}")
    for s in pf.stations:
        if s.chord <= 0:
            bad.append(f"station {s.index}: non-positive chord")
    return bad
