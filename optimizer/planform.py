"""Analytic planform model of the half-wing.

Gives station coordinates and the reference quantities (S_ref, MAC, b, AR,
moment reference) straight from the config, so CFD coefficients never depend
on what the CAD tool reports.

Conventions: x aft, y span (right), z up.  `span` of a section is its
projected span along y.  Sweep is taken at `sweep_location` of the local
chord and acts along the dihedral-tilted section axis (length span/cos(dihedral)),
which is how the OpenVSP backend sets the section.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

from .config import GeometryCfg


def _trapz(f, x):
    """Trapezoidal rule (np.trapz was renamed in NumPy 2)."""
    f = np.asarray(f, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


@dataclass(frozen=True)
class Station:
    index: int
    y: float
    x_le: float
    z: float
    chord: float
    twist_deg: float
    airfoil: str

    @property
    def thickness_ratio(self) -> float:
        return int(self.airfoil[2:]) / 100.0


@dataclass
class Planform:
    stations: List[Station]
    half_span: float
    area_half: float
    area_full: float
    mac: float
    x_mac_le: float
    y_mac: float
    aspect_ratio: float
    taper_ratio: float
    x_max: float
    x_min: float
    z_min: float
    z_max: float
    mean_qc_sweep_deg: float
    moment_ref: List[float]

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["stations"] = [asdict(s) for s in self.stations]
        return d


def build_stations(g: GeometryCfg) -> List[Station]:
    ox, _, oz = g.origin
    chord = g.root_chord
    x_le, y, z = ox, 0.0, oz
    st = [Station(0, y, x_le, z, chord, 0.0, g.airfoil_stations[0])]
    for i, sec in enumerate(g.sections, start=1):
        gam = math.radians(sec.dihedral_deg)
        lam = math.radians(sec.sweep_deg)
        axial = sec.span / math.cos(gam)            # length along the tilted section axis
        p = sec.sweep_location
        x_le = x_le + p * chord + axial * math.tan(lam) - p * sec.tip_chord
        y = y + sec.span
        z = z + sec.span * math.tan(gam)
        chord = sec.tip_chord
        st.append(Station(i, y, x_le, z, chord, sec.twist_deg, g.airfoil_stations[i]))
    return st


def compute_planform(g: GeometryCfg, samples: int = 200) -> Planform:
    st = build_stations(g)
    half_span = st[-1].y
    area_half = 0.0
    ys, cs, xs = [], [], []
    for a, b in zip(st[:-1], st[1:]):
        dy = b.y - a.y
        area_half += 0.5 * (a.chord + b.chord) * dy
        t = np.linspace(0.0, 1.0, samples, endpoint=False)
        ys.append(a.y + t * dy)
        cs.append(a.chord + t * (b.chord - a.chord))
        xs.append(a.x_le + t * (b.x_le - a.x_le))
    ys.append(np.array([st[-1].y]))
    cs.append(np.array([st[-1].chord]))
    xs.append(np.array([st[-1].x_le]))
    y = np.concatenate(ys)
    c = np.concatenate(cs)
    x = np.concatenate(xs)
    mac = float(_trapz(c ** 2, y) / area_half)
    x_mac_le = float(_trapz(x * c, y) / area_half)
    y_mac = float(_trapz(y * c, y) / area_half)
    ar = (2.0 * half_span) ** 2 / (2.0 * area_half)
    taper = st[-1].chord / st[0].chord
    qc = [(b.x_le + 0.25 * b.chord) - (a.x_le + 0.25 * a.chord) for a, b in zip(st[:-1], st[1:])]
    dys = [b.y - a.y for a, b in zip(st[:-1], st[1:])]
    qc_sweep = math.degrees(math.atan2(sum(qc), sum(dys)))
    x_ref = x_mac_le + g.cg_x_frac_mac * mac
    return Planform(
        stations=st,
        half_span=half_span,
        area_half=area_half,
        area_full=2.0 * area_half,
        mac=mac,
        x_mac_le=x_mac_le,
        y_mac=y_mac,
        aspect_ratio=ar,
        taper_ratio=taper,
        x_max=max(s.x_le + s.chord for s in st),
        x_min=min(s.x_le for s in st),
        z_min=min(s.z for s in st) - 0.5 * g.root_chord * 0.2,
        z_max=max(s.z for s in st) + 0.5 * g.root_chord * 0.2,
        mean_qc_sweep_deg=qc_sweep,
        moment_ref=[x_ref, 0.0, g.origin[2]],
    )
