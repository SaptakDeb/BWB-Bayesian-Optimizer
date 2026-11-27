"""Pure-numpy lofted wing (NACA 4-digit sections) -> closed STL.

Used when OpenVSP is not installed, for unit tests and for the mock dry-run.
It follows exactly the same parametrisation as the OpenVSP backend (shared
station layout from `planform.build_stations`).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import trimesh

from ..config import GeometryCfg
from ..planform import Station, build_stations
from .stl_io import write_single_solid_stl


def naca4_loop(code: str, n: int = 41) -> np.ndarray:
    """Closed airfoil loop (x/c, z/c): upper TE -> LE -> lower TE, finite-thickness TE."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    xs = np.concatenate([xu[::-1], xl[1:]])
    zs = np.concatenate([zu[::-1], zl[1:]])
    return np.column_stack([xs, zs])


def _place(loop: np.ndarray, s: Station) -> np.ndarray:
    th = math.radians(s.twist_deg)
    xc = loop[:, 0] - 0.25
    zc = loop[:, 1]
    xr = 0.25 + xc * math.cos(th) + zc * math.sin(th)       # nose-up positive twist about c/4
    zr = -xc * math.sin(th) + zc * math.cos(th)
    pts = np.empty((len(loop), 3))
    pts[:, 0] = s.x_le + s.chord * xr
    pts[:, 1] = s.y
    pts[:, 2] = s.z + s.chord * zr
    return pts


def _mirror(stations: List[Station]) -> List[Station]:
    out = []
    for s in reversed(stations[1:]):
        out.append(Station(-1, -s.y, s.x_le, s.z, s.chord, s.twist_deg, s.airfoil))
    out.extend(stations)
    return out


def build_mesh(g: GeometryCfg, with_symmetry_mirror: bool = False) -> trimesh.Trimesh:
    stations = build_stations(g)
    if with_symmetry_mirror:
        stations = _mirror(stations)
    n = g.tess.chordwise
    # refine spanwise by linear interpolation of the section parameters
    rings = []
    for a, b in zip(stations[:-1], stations[1:]):
        for k in range(g.tess.span_per_section):
            f = k / g.tess.span_per_section
            s = Station(
                -1,
                a.y + f * (b.y - a.y),
                a.x_le + f * (b.x_le - a.x_le),
                a.z + f * (b.z - a.z),
                a.chord + f * (b.chord - a.chord),
                a.twist_deg + f * (b.twist_deg - a.twist_deg),
                a.airfoil,
            )
            rings.append(_place(naca4_loop(a.airfoil, n) * (1 - f) + naca4_loop(b.airfoil, n) * f, s))
    rings.append(_place(naca4_loop(stations[-1].airfoil, n), stations[-1]))
    P = np.array(rings)                       # (S, K, 3)
    S, K, _ = P.shape
    verts = [P.reshape(-1, 3)]
    faces = []

    def vid(s, k):
        return s * K + (k % K)

    for s in range(S - 1):
        for k in range(K):
            a, b, c, d = vid(s, k), vid(s, k + 1), vid(s + 1, k + 1), vid(s + 1, k)
            faces.append([a, b, c])
            faces.append([a, c, d])
    base = S * K
    cap0 = P[0].mean(axis=0)
    capN = P[-1].mean(axis=0)
    verts.append(cap0[None, :])
    verts.append(capN[None, :])
    for k in range(K):
        faces.append([base, vid(0, k + 1), vid(0, k)])
        faces.append([base + 1, vid(S - 1, k), vid(S - 1, k + 1)])
    mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.array(faces), process=False)
    mesh.merge_vertices()
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()
    if g.root_incidence_deg != 0.0:
        th = math.radians(g.root_incidence_deg)
        ox, _, oz = g.origin
        v = mesh.vertices.copy()
        x, z = v[:, 0] - ox, v[:, 2] - oz
        v[:, 0] = ox + x * math.cos(th) + z * math.sin(th)
        v[:, 2] = oz - x * math.sin(th) + z * math.cos(th)
        mesh.vertices = v
    return mesh


def build_wing_stl(g: GeometryCfg, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_mesh(g, with_symmetry_mirror=(g.symmetry_mode == "full"))
    path = out_dir / f"{g.name}_wing.stl"
    write_single_solid_stl(path, np.asarray(mesh.vertices), np.asarray(mesh.faces), "wing")
    return path
