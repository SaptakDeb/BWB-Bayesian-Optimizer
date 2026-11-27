"""Typed configuration loader for the whole project.

Everything the pipeline needs comes from one YAML file (see config/config.yaml).
No secrets live here: the SimScale API key is read from the environment only.
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """One wing section (segment between two airfoil stations).

    span          : projected span of the section along y [m]
    tip_chord     : chord at the outboard end of the section [m]
    sweep_deg     : sweep of the line at `sweep_location` (0 = leading edge) [deg]
    twist_deg     : twist of the OUTBOARD airfoil, nose-up positive [deg]
    dihedral_deg  : dihedral of the section [deg]
    """

    name: str
    span: float
    tip_chord: float
    sweep_deg: float
    twist_deg: float = 0.0
    dihedral_deg: float = 0.0
    sweep_location: float = 0.0


@dataclass
class TessCfg:
    span_per_section: int = 12
    chordwise: int = 41
    le_cluster: float = 0.25
    te_cluster: float = 0.25


@dataclass
class GeometryCfg:
    name: str = "BWB_halfwing"
    backend: str = "openvsp"            # openvsp | analytic
    symmetry_mode: str = "half"         # half (symmetry plane at y=0) | full
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    root_chord: float = 3.0
    root_incidence_deg: float = 0.0
    airfoil_stations: List[str] = field(default_factory=list)   # NACA 4-digit, root -> tip
    sections: List[Section] = field(default_factory=list)
    cg_x_frac_mac: float = 0.30         # moment reference as fraction of MAC
    tess: TessCfg = field(default_factory=TessCfg)


# --------------------------------------------------------------------------- #
# Flow / domain / mesh
# --------------------------------------------------------------------------- #
@dataclass
class FlowCfg:
    speed: float = 100.0                # [m/s]
    aoa_deg: float = 4.0
    altitude_m: float = 0.0
    analysis_type: str = "incompressible"   # incompressible | compressible
    turbulence_intensity: float = 0.001     # inlet freestream TI (fraction)


@dataclass
class DomainCfg:
    upstream_root_chords: float = 8.0
    downstream_root_chords: float = 16.0
    lateral_half_spans: float = 3.0
    vertical_half_spans: float = 3.0


@dataclass
class MeshCfg:
    sizing_mode: str = "manual"            # manual | automatic
    automatic_fineness: str = "FINE"       # VERY_COARSE|COARSE|MODERATE|FINE|VERY_FINE
    wall_treatment: str = "resolved"       # resolved (y+~1, SST low-Re) | wall_function
    target_y_plus: float = 1.0
    expansion_ratio: float = 1.2
    max_layers: int = 40
    min_thickness_ratio: float = 0.2       # min layer thickness / first layer thickness
    bl_coverage: float = 1.0               # layer stack thickness / estimated BL thickness
    surface_min_mac_frac: float = 0.004    # surface refinement min edge / MAC
    surface_max_mac_frac: float = 0.020    # surface refinement max edge / MAC
    farfield_max_mac_frac: float = 0.60    # global max edge / MAC
    near_body_size_mac_frac: float = 0.030
    near_body_pad_root_chords: float = 0.5
    wake_size_mac_frac: float = 0.060
    wake_length_root_chords: float = 4.0
    wake_half_height_root_chords: float = 0.6
    use_feature_refinement: bool = True
    feature_distance_mac_frac: float = 0.05
    processors: int = 16                   # SimScale: -1, 4, 8, 16, 32, 48, 64, 96
    max_run_time_s: float = 36000.0
    max_cells: float = 4.0e7               # local guard on the rough cell estimate


@dataclass
class SimScaleCfg:
    project_name: str = "bwb-bayesopt"
    project_id: Optional[str] = None       # reuse an existing project (optional)
    geometry_format: str = "STL"
    geometry_unit: str = "m"
    facet_split: bool = False
    sewing: bool = False
    improve: bool = True
    material_name: str = "Air"
    iterations: int = 1500
    write_interval: int = 250
    averaging_window: int = 200            # average the last N iterations of the force history
    poll_interval_s: float = 30.0
    geometry_timeout_s: float = 900.0
    mesh_timeout_s: float = 14400.0
    sim_timeout_s: float = 43200.0
    max_compute_units: Optional[float] = 100.0   # abort if SimScale's estimate is higher
    entity_resolution: str = "auto"        # auto | manual
    manual_entities: Dict[str, List[str]] = field(default_factory=dict)
    results_column_map: Dict[str, str] = field(default_factory=dict)
    force_scale: float = 1.0               # multiply downloaded forces/moments (e.g. rho if they are kinematic)


# --------------------------------------------------------------------------- #
# Objective / optimisation
# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveWeights:
    ld: float = 0.5
    cd: float = 0.2
    cl: float = 0.2
    cm: float = 0.1


@dataclass
class ObjectivePenalties:
    cl: float = 5.0
    cm: float = 20.0
    failure_value: float = 10.0


@dataclass
class ObjectiveCfg:
    mode: str = "multi"                    # max_ld | min_cd | max_ld_cm | multi
    cl_target: float = 0.45
    cm_target: float = 0.0
    cm_scale: float = 0.05
    cm_bounds: List[float] = field(default_factory=lambda: [-0.04, 0.04])
    normalise_by_baseline: bool = True
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    penalties: ObjectivePenalties = field(default_factory=ObjectivePenalties)


@dataclass
class DesignVar:
    name: str
    param: str                             # see parametrisation.PARAMS
    bounds: List[float]
    section: Optional[int] = None          # 1-based section index
    station: Optional[int] = None          # 0-based station index (airfoil params)


@dataclass
class ConstraintsCfg:
    half_span_range: Optional[List[float]] = None
    area_range: Optional[List[float]] = None      # full-wing planform area [m^2]
    min_tip_chord: Optional[float] = None
    min_root_thickness_m: Optional[float] = None
    max_le_sweep_deg: Optional[float] = None


@dataclass
class OptCfg:
    seed: int = 7
    n_initial: int = 8
    n_iterations: int = 20
    include_baseline: bool = True
    acquisition: str = "ei"                # ei | ucb | pi
    xi: float = 0.01
    kappa: float = 2.0
    n_restarts: int = 4                    # GP hyper-parameter restarts
    n_candidates: int = 8192               # quasi-random pool the acquisition is screened on
    acq_starts: int = 8                    # multi-start gradient ascent on the acquisition
    acq_steps: int = 80
    batch_size: int = 1                    # designs proposed per BO step (parallel CFD runs)
    device: str = "auto"                   # auto | cpu | cuda
    patience: Optional[int] = None
    design_variables: List[DesignVar] = field(default_factory=list)
    constraints: ConstraintsCfg = field(default_factory=ConstraintsCfg)


@dataclass
class ProjectCfg:
    name: str = "bwb_bo"
    workdir: str = "runs/test_case"


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    flow: FlowCfg = field(default_factory=FlowCfg)
    domain: DomainCfg = field(default_factory=DomainCfg)
    mesh: MeshCfg = field(default_factory=MeshCfg)
    simscale: SimScaleCfg = field(default_factory=SimScaleCfg)
    objective: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    optimisation: OptCfg = field(default_factory=OptCfg)


# --------------------------------------------------------------------------- #
# Generic dict -> dataclass builder
# --------------------------------------------------------------------------- #
def _build(tp: Any, data: Any, path: str = "") -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if data is None:
            return None
        return _build(args[0], data, path)
    if origin in (list, List):
        (inner,) = get_args(tp)
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list")
        return [_build(inner, d, f"{path}[{i}]") for i, d in enumerate(data)]
    if origin in (dict, Dict):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return dict(data)
    if dataclasses.is_dataclass(tp):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path or tp.__name__}: expected a mapping")
        hints = get_type_hints(tp)
        known = {f.name for f in dataclasses.fields(tp)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"{path or tp.__name__}: unknown keys {sorted(unknown)}")
        kwargs = {k: _build(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
        try:
            return tp(**kwargs)
        except TypeError as exc:
            raise ConfigError(f"{path or tp.__name__}: {exc}") from exc
    if tp is float:
        return float(data)
    if tp is int:
        return int(data)
    if tp is bool:
        if not isinstance(data, bool):
            raise ConfigError(f"{path}: expected true/false")
        return data
    if tp is str:
        return str(data)
    return data


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: Any = d
    for k in keys[:-1]:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur.setdefault(k, {})
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def apply_overrides(raw: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Apply `a.b.c=value` overrides (value parsed as YAML) to a raw config dict."""
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        key, val = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(val))
    return raw


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
_NACA4 = re.compile(r"^\d{4}$")
BACKENDS_GEOM = ("openvsp", "analytic")
OBJECTIVE_MODES = ("max_ld", "min_cd", "max_ld_cm", "multi")
ANALYSES = ("incompressible", "compressible")


def validate(cfg: Config) -> Config:
    g = cfg.geometry
    if g.backend not in BACKENDS_GEOM:
        raise ConfigError(f"geometry.backend must be one of {BACKENDS_GEOM}")
    if g.symmetry_mode not in ("half", "full"):
        raise ConfigError("geometry.symmetry_mode must be 'half' or 'full'")
    if len(g.origin) != 3:
        raise ConfigError("geometry.origin must have 3 entries [x, y, z]")
    if abs(g.origin[1]) > 0.0:
        raise ConfigError("geometry.origin[1] (y) must be 0: the wing is mirrored about the xz plane")
    if not g.sections:
        raise ConfigError("geometry.sections is empty")
    if len(g.airfoil_stations) != len(g.sections) + 1:
        raise ConfigError(
            f"geometry.airfoil_stations needs {len(g.sections) + 1} entries "
            f"(sections + 1), got {len(g.airfoil_stations)}"
        )
    for code in g.airfoil_stations:
        if not _NACA4.match(str(code)):
            raise ConfigError(f"airfoil '{code}' is not a NACA 4-digit string like '0012'")
    if g.root_chord <= 0:
        raise ConfigError("geometry.root_chord must be > 0")
    for s in g.sections:
        if s.span <= 0 or s.tip_chord <= 0:
            raise ConfigError(f"section '{s.name}': span and tip_chord must be > 0")
        if not 0.0 <= s.sweep_location <= 1.0:
            raise ConfigError(f"section '{s.name}': sweep_location must be in [0, 1]")
        if abs(s.dihedral_deg) >= 80.0 or abs(s.sweep_deg) >= 85.0:
            raise ConfigError(f"section '{s.name}': dihedral/sweep out of range")
    if cfg.flow.analysis_type not in ANALYSES:
        raise ConfigError(f"flow.analysis_type must be one of {ANALYSES}")
    if cfg.flow.speed <= 0:
        raise ConfigError("flow.speed must be > 0")
    if cfg.mesh.wall_treatment not in ("resolved", "wall_function"):
        raise ConfigError("mesh.wall_treatment must be 'resolved' or 'wall_function'")
    if cfg.mesh.sizing_mode not in ("manual", "automatic"):
        raise ConfigError("mesh.sizing_mode must be 'manual' or 'automatic'")
    if cfg.mesh.processors not in (-1, 4, 8, 16, 32, 48, 64, 96):
        raise ConfigError("mesh.processors must be one of -1, 4, 8, 16, 32, 48, 64, 96")
    if cfg.mesh.expansion_ratio <= 1.0:
        raise ConfigError("mesh.expansion_ratio must be > 1")
    if cfg.objective.mode not in OBJECTIVE_MODES:
        raise ConfigError(f"objective.mode must be one of {OBJECTIVE_MODES}")
    if len(cfg.objective.cm_bounds) != 2 or cfg.objective.cm_bounds[0] >= cfg.objective.cm_bounds[1]:
        raise ConfigError("objective.cm_bounds must be [low, high] with low < high")
    if cfg.optimisation.acquisition not in ("ei", "ucb", "pi"):
        raise ConfigError("optimisation.acquisition must be ei, ucb or pi")
    if cfg.simscale.entity_resolution not in ("auto", "manual"):
        raise ConfigError("simscale.entity_resolution must be 'auto' or 'manual'")
    names = set()
    for dv in cfg.optimisation.design_variables:
        if dv.name in names:
            raise ConfigError(f"duplicate design variable name '{dv.name}'")
        names.add(dv.name)
        if len(dv.bounds) != 2 or dv.bounds[0] >= dv.bounds[1]:
            raise ConfigError(f"design variable '{dv.name}': bounds must be [low, high], low < high")
        if dv.section is not None and not 1 <= dv.section <= len(g.sections):
            raise ConfigError(f"design variable '{dv.name}': section {dv.section} out of range")
        if dv.station is not None and not 0 <= dv.station <= len(g.sections):
            raise ConfigError(f"design variable '{dv.name}': station {dv.station} out of range")
    return cfg


def load_config(path: Union[str, Path], overrides: Optional[List[str]] = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = apply_overrides(raw, overrides)
    cfg = _build(Config, raw)
    return validate(cfg)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)
