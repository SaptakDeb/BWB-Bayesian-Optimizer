"""Mesh requirements for SimScale hex-dominant (snappyHexMesh) + regular k-omega SST.

Regular SST (no transition model) is run wall-resolved (y+ ~ 1, `resolved`) or
with wall functions (y+ ~ 30, `wall_function`).  Everything below is derived
from the config + planform + flow, so the mesh follows the geometry the
optimiser proposes.

Boundary-layer estimates use flat-plate turbulent correlations:
    Cf     = 0.026 Re_x^(-1/7)          (skin friction, Schlichting)
    delta  = 0.37 x Re_x^(-1/5)         (boundary-layer thickness)
    y1     = y+ * nu / u_tau,  u_tau = U sqrt(Cf/2)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..atmosphere import FlowConditions
from ..config import Config
from ..planform import Planform


def flat_plate_cf(re_x: float) -> float:
    return 0.026 * re_x ** (-1.0 / 7.0)


def bl_thickness(x: float, re_x: float) -> float:
    return 0.37 * x * re_x ** (-0.2)


def first_layer_height(y_plus: float, speed: float, nu: float, re_x: float) -> float:
    u_tau = speed * math.sqrt(0.5 * flat_plate_cf(re_x))
    return y_plus * nu / u_tau


def layers_needed(first: float, ratio: float, thickness: float) -> int:
    """Smallest n with first*(r^n - 1)/(r - 1) >= thickness."""
    if thickness <= first:
        return 1
    return int(math.ceil(math.log(1.0 + thickness * (ratio - 1.0) / first) / math.log(ratio)))


def stack_thickness(first: float, ratio: float, n: int) -> float:
    return first * (ratio ** n - 1.0) / (ratio - 1.0)


@dataclass
class MeshPlan:
    wall_treatment: str
    target_y_plus: float
    re_mac: float
    re_root: float
    first_layer: float
    min_layer_thickness: float
    expansion_ratio: float
    layers: int
    layers_wanted: int
    stack_thickness: float
    bl_thickness_root: float
    coverage: float
    surface_min: float
    surface_max: float
    farfield_max: float
    near_body_size: float
    wake_size: float
    near_body_box: Dict[str, List[float]]
    wake_box: Dict[str, List[float]]
    feature_distance: float
    feature_length: float
    est_wetted_area: float
    est_surface_cells: float
    est_cells: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def plan_mesh(cfg: Config, pf: Planform, flow: FlowConditions) -> MeshPlan:
    m = cfg.mesh
    g = cfg.geometry
    nu, u = flow.kinematic_viscosity, flow.speed
    re_mac = flow.reynolds(pf.mac)
    re_root = flow.reynolds(g.root_chord)
    warnings: List[str] = []

    # size the first cell for the most demanding station (largest Re = root chord)
    y_plus = m.target_y_plus
    first = first_layer_height(y_plus, u, nu, re_root)
    delta_root = bl_thickness(g.root_chord, re_root)
    wanted = layers_needed(first, m.expansion_ratio, m.bl_coverage * delta_root)
    layers = min(wanted, m.max_layers)
    stack = stack_thickness(first, m.expansion_ratio, layers)
    coverage = stack / delta_root
    if layers < wanted:
        warnings.append(
            f"layer stack capped at {layers} (wanted {wanted}); covers {coverage:.0%} of the root-chord BL. "
            "Raise mesh.max_layers or expansion_ratio, or use wall_function."
        )
    if m.wall_treatment == "resolved" and y_plus > 1.5:
        warnings.append(f"wall_treatment=resolved but target_y_plus={y_plus}; SST low-Re wants y+ <= 1")
    if m.wall_treatment == "wall_function" and not 30.0 <= y_plus <= 100.0:
        warnings.append(f"wall_treatment=wall_function but target_y_plus={y_plus}; use 30-100")
    if m.expansion_ratio > 1.3:
        warnings.append("expansion_ratio > 1.3 degrades snappy layer quality")

    s_min = m.surface_min_mac_frac * pf.mac
    s_max = m.surface_max_mac_frac * pf.mac
    far = m.farfield_max_mac_frac * pf.mac
    near = m.near_body_size_mac_frac * pf.mac
    wake = m.wake_size_mac_frac * pf.mac
    if s_min < 4.0 * first:
        warnings.append("surface_min is within ~4 first-layer heights; layer addition may collapse near sharp edges")

    pad = m.near_body_pad_root_chords * g.root_chord
    near_box = {
        "min": [pf.x_min - pad, 0.0 if g.symmetry_mode == "half" else -(pf.half_span + pad), pf.z_min - pad],
        "max": [pf.x_max + pad, pf.half_span + pad, pf.z_max + pad],
    }
    wake_box = {
        "min": [pf.x_max - 0.5 * g.root_chord, 0.0 if g.symmetry_mode == "half" else -(pf.half_span + pad),
                g.origin[2] - m.wake_half_height_root_chords * g.root_chord],
        "max": [pf.x_max + m.wake_length_root_chords * g.root_chord, pf.half_span + pad,
                g.origin[2] + m.wake_half_height_root_chords * g.root_chord],
    }

    # rough cell budget (order of magnitude only)
    mean_t = sum(s.thickness_ratio for s in pf.stations) / len(pf.stations)
    wetted = 2.0 * pf.area_half * (1.0 + 0.25 * mean_t)
    if g.symmetry_mode == "full":
        wetted *= 2.0
    h_s = 0.5 * (s_min + s_max)
    surf_cells = wetted / (h_s ** 2)
    vol = lambda b: max(0.0, (b["max"][0] - b["min"][0]) * (b["max"][1] - b["min"][1]) * (b["max"][2] - b["min"][2]))
    dom_vol = (pf.x_max - pf.x_min + (cfg.domain.upstream_root_chords + cfg.domain.downstream_root_chords) * g.root_chord) \
        * cfg.domain.lateral_half_spans * pf.half_span * 2.0 * cfg.domain.vertical_half_spans * pf.half_span
    if g.symmetry_mode == "full":
        dom_vol *= 2.0
    cells = (
        surf_cells * (layers + 6.0)
        + vol(near_box) / near ** 3
        + vol(wake_box) / wake ** 3
        + dom_vol / far ** 3
    )
    if cells > m.max_cells:
        warnings.append(f"rough cell estimate {cells:.2e} exceeds mesh.max_cells={m.max_cells:.1e}")

    return MeshPlan(
        wall_treatment=m.wall_treatment,
        target_y_plus=y_plus,
        re_mac=re_mac,
        re_root=re_root,
        first_layer=first,
        min_layer_thickness=m.min_thickness_ratio * first,
        expansion_ratio=m.expansion_ratio,
        layers=layers,
        layers_wanted=wanted,
        stack_thickness=stack,
        bl_thickness_root=delta_root,
        coverage=coverage,
        surface_min=s_min,
        surface_max=s_max,
        farfield_max=far,
        near_body_size=near,
        wake_size=wake,
        near_body_box=near_box,
        wake_box=wake_box,
        feature_distance=m.feature_distance_mac_frac * pf.mac,
        feature_length=s_min,
        est_wetted_area=wetted,
        est_surface_cells=surf_cells,
        est_cells=cells,
        warnings=warnings,
    )
