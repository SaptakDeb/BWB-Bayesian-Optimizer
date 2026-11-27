"""Maps optimiser design vectors onto the geometry configuration."""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, DesignVar, GeometryCfg

# parameter name -> (needs section?, needs station?)
PARAMS = {
    "span": (True, False),
    "tip_chord": (True, False),
    "sweep_deg": (True, False),
    "twist_deg": (True, False),
    "dihedral_deg": (True, False),
    "sweep_location": (True, False),
    "root_chord": (False, False),
    "root_incidence_deg": (False, False),
    "thickness_pct": (False, True),
}


def _validate_var(dv: DesignVar) -> None:
    if dv.param not in PARAMS:
        raise ValueError(f"design variable '{dv.name}': unknown param '{dv.param}' (choose from {sorted(PARAMS)})")
    needs_section, needs_station = PARAMS[dv.param]
    if needs_section and dv.section is None:
        raise ValueError(f"design variable '{dv.name}': param '{dv.param}' needs 'section'")
    if needs_station and dv.station is None:
        raise ValueError(f"design variable '{dv.name}': param '{dv.param}' needs 'station'")


def get_value(g: GeometryCfg, dv: DesignVar) -> float:
    _validate_var(dv)
    if dv.param == "root_chord":
        return g.root_chord
    if dv.param == "root_incidence_deg":
        return g.root_incidence_deg
    if dv.param == "thickness_pct":
        return float(int(g.airfoil_stations[dv.station][2:]))
    return float(getattr(g.sections[dv.section - 1], dv.param))


def set_value(g: GeometryCfg, dv: DesignVar, value: float) -> None:
    _validate_var(dv)
    if dv.param == "root_chord":
        g.root_chord = float(value)
    elif dv.param == "root_incidence_deg":
        g.root_incidence_deg = float(value)
    elif dv.param == "thickness_pct":
        code = g.airfoil_stations[dv.station]
        t = int(round(min(max(value, 1.0), 40.0)))
        g.airfoil_stations[dv.station] = f"{code[:2]}{t:02d}"
    else:
        setattr(g.sections[dv.section - 1], dv.param, float(value))


class DesignSpace:
    """Box-bounded design space with unit-cube normalisation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vars: List[DesignVar] = list(cfg.optimisation.design_variables)
        if not self.vars:
            raise ValueError("optimisation.design_variables is empty")
        for dv in self.vars:
            _validate_var(dv)
        self.names = [v.name for v in self.vars]
        self.lower = np.array([v.bounds[0] for v in self.vars], dtype=float)
        self.upper = np.array([v.bounds[1] for v in self.vars], dtype=float)

    @property
    def dim(self) -> int:
        return len(self.vars)

    def nominal(self) -> Dict[str, float]:
        """Design vector of the geometry exactly as written in the config."""
        return {v.name: get_value(self.cfg.geometry, v) for v in self.vars}

    def clip(self, x: Dict[str, float]) -> Dict[str, float]:
        return {n: float(min(max(x[n], lo), hi)) for n, lo, hi in zip(self.names, self.lower, self.upper)}

    def to_unit(self, x: Dict[str, float]) -> np.ndarray:
        arr = np.array([x[n] for n in self.names], dtype=float)
        return (arr - self.lower) / (self.upper - self.lower)

    def from_unit(self, u: np.ndarray) -> Dict[str, float]:
        arr = self.lower + np.asarray(u, dtype=float) * (self.upper - self.lower)
        return {n: float(v) for n, v in zip(self.names, arr)}

    def apply(self, x: Dict[str, float]) -> GeometryCfg:
        """Return a COPY of the base geometry with design values applied."""
        g = copy.deepcopy(self.cfg.geometry)
        for dv in self.vars:
            if dv.name not in x:
                raise KeyError(f"design vector is missing '{dv.name}'")
            set_value(g, dv, x[dv.name])
        return g

    def nominal_outside_bounds(self) -> List[Tuple[str, float, float, float]]:
        out = []
        nom = self.nominal()
        for n, lo, hi in zip(self.names, self.lower, self.upper):
            if not lo <= nom[n] <= hi:
                out.append((n, nom[n], lo, hi))
        return out
