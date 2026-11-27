"""Builds the fluid domain: far-field box minus the wing, with named patches.

The SimScale OpenFOAM (hex-dominant) workflow needs the fluid volume in the
CAD itself.  We take the wing STL, boolean-subtract it from a box and split
the boundary into named solids:

    inlet, outlet, farfield, symmetry (half model only), wing

The result is ONE multi-solid ASCII STL whose enclosed volume is the fluid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh

from ..config import DomainCfg, GeometryCfg
from ..planform import Planform
from .stl_io import write_multisolid_stl

PATCHES = ("inlet", "outlet", "farfield", "symmetry", "wing")


@dataclass
class DomainBounds:
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]

    def extents(self):
        return tuple(h - l for l, h in zip(self.lo, self.hi))


@dataclass
class EnclosureReport:
    bounds: DomainBounds
    watertight: bool
    volume: float
    patch_faces: Dict[str, int]
    patch_area: Dict[str, float]
    path: Path


def domain_bounds(g: GeometryCfg, pf: Planform, d: DomainCfg) -> DomainBounds:
    x_lo = pf.x_min - d.upstream_root_chords * g.root_chord
    x_hi = pf.x_max + d.downstream_root_chords * g.root_chord
    y_hi = d.lateral_half_spans * pf.half_span
    y_lo = 0.0 if g.symmetry_mode == "half" else -y_hi
    zc = g.origin[2]
    z_half = d.vertical_half_spans * pf.half_span
    return DomainBounds((x_lo, y_lo, zc - z_half), (x_hi, y_hi, zc + z_half))


def _load_solid(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        raise RuntimeError(
            "Wing surface is not watertight - check the end caps (OpenVSP: flat caps at root and tip)."
        )
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _boolean(op: str, meshes, engine: str = "manifold") -> trimesh.Trimesh:
    fn = {"union": trimesh.boolean.union, "difference": trimesh.boolean.difference,
          "intersection": trimesh.boolean.intersection}[op]
    return fn(meshes, engine=engine)


def build_enclosure(wing_stl: Path, g: GeometryCfg, pf: Planform, d: DomainCfg, out_stl: Path) -> EnclosureReport:
    wing = _load_solid(Path(wing_stl))
    bounds = domain_bounds(g, pf, d)
    lo, hi = np.array(bounds.lo), np.array(bounds.hi)
    size = float(np.max(hi - lo))
    tol = 1e-6 * size

    if g.symmetry_mode == "half":
        # keep only y >= 0 so the root section lies exactly in the symmetry plane
        clip = trimesh.creation.box(bounds=[[lo[0], 0.0, lo[2]], [hi[0], hi[1], hi[2]]])
        wing = _boolean("intersection", [wing, clip])
        v = wing.vertices.copy()
        v[np.abs(v[:, 1]) < tol, 1] = 0.0
        wing.vertices = v

    box = trimesh.creation.box(bounds=[lo.tolist(), hi.tolist()])
    fluid = _boolean("difference", [box, wing])
    fluid.merge_vertices()

    tri = fluid.vertices[fluid.faces]              # (m,3,3)
    on = {
        "inlet": np.all(np.abs(tri[:, :, 0] - lo[0]) < tol, axis=1),
        "outlet": np.all(np.abs(tri[:, :, 0] - hi[0]) < tol, axis=1),
        "ymax": np.all(np.abs(tri[:, :, 1] - hi[1]) < tol, axis=1),
        "ymin": np.all(np.abs(tri[:, :, 1] - lo[1]) < tol, axis=1),
        "zmin": np.all(np.abs(tri[:, :, 2] - lo[2]) < tol, axis=1),
        "zmax": np.all(np.abs(tri[:, :, 2] - hi[2]) < tol, axis=1),
    }
    label = np.full(len(fluid.faces), "wing", dtype=object)
    label[on["zmin"] | on["zmax"] | on["ymax"]] = "farfield"
    if g.symmetry_mode == "half":
        label[on["ymin"]] = "symmetry"
    else:
        label[on["ymin"]] = "farfield"
    label[on["inlet"]] = "inlet"
    label[on["outlet"]] = "outlet"

    patches: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    areas = fluid.area_faces
    faces_count: Dict[str, int] = {}
    area: Dict[str, float] = {}
    for name in PATCHES:
        mask = label == name
        faces_count[name] = int(mask.sum())
        area[name] = float(areas[mask].sum())
        if mask.any():
            patches[name] = (np.asarray(fluid.vertices), np.asarray(fluid.faces)[mask])
    if faces_count["wing"] == 0 or faces_count["inlet"] == 0 or faces_count["outlet"] == 0:
        raise RuntimeError(f"enclosure is missing mandatory patches: {faces_count}")
    write_multisolid_stl(Path(out_stl), patches)
    return EnclosureReport(
        bounds=bounds,
        watertight=bool(fluid.is_watertight),
        volume=float(abs(fluid.volume)),
        patch_faces=faces_count,
        patch_area=area,
        path=Path(out_stl),
    )
