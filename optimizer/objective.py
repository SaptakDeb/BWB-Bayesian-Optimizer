"""Selectable objective: every mode returns a COST that the optimiser MINIMISES.

Modes (objective.mode, or --objective on the command line):
    max_ld     maximise L/D, with a quadratic penalty keeping CL near cl_target
    min_cd     minimise CD, with the same CL penalty
    max_ld_cm  maximise L/D, penalise CL deviation and CM outside cm_bounds
    multi      weighted sum: w.ld*(-L/D) + w.cd*CD + w.cl*dCL^2 + w.cm*dCM^2

With normalise_by_baseline the L/D and CD terms are divided by the baseline
design's values, so the terms are O(1) and the weights are meaningful.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .config import ObjectiveCfg
from .cfd.base import CFDResult


@dataclass
class Baseline:
    cl: float
    cd: float
    cm: float

    @property
    def ld(self) -> float:
        return self.cl / self.cd

    def to_dict(self) -> Dict[str, float]:
        return {"cl": self.cl, "cd": self.cd, "cm": self.cm}


@dataclass
class Score:
    total: float
    terms: Dict[str, float]
    mode: str
    valid: bool = True


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def score(cfg: ObjectiveCfg, res: CFDResult, baseline: Optional[Baseline]) -> Score:
    if not res.ok or not _finite(res.cl, res.cd, res.cm) or res.cd <= 0.0:
        return Score(cfg.penalties.failure_value, {}, cfg.mode, valid=False)

    ld0 = cd0 = 1.0
    if cfg.normalise_by_baseline and baseline is not None:
        ld0 = max(abs(baseline.ld), 1e-9)
        cd0 = max(abs(baseline.cd), 1e-12)

    ld_term = -(res.cl / res.cd) / ld0
    cd_term = res.cd / cd0
    cl_term = ((res.cl - cfg.cl_target) / cfg.cl_target) ** 2 if cfg.cl_target else 0.0
    cm_term = ((res.cm - cfg.cm_target) / cfg.cm_scale) ** 2
    lo, hi = cfg.cm_bounds
    cm_excess = max(0.0, lo - res.cm, res.cm - hi)
    cm_bound = (cm_excess / cfg.cm_scale) ** 2

    w, p = cfg.weights, cfg.penalties
    if cfg.mode == "max_ld":
        terms = {"ld": ld_term, "cl_pen": p.cl * cl_term}
    elif cfg.mode == "min_cd":
        terms = {"cd": cd_term, "cl_pen": p.cl * cl_term}
    elif cfg.mode == "max_ld_cm":
        terms = {"ld": ld_term, "cl_pen": p.cl * cl_term, "cm_pen": p.cm * cm_bound}
    elif cfg.mode == "multi":
        terms = {"ld": w.ld * ld_term, "cd": w.cd * cd_term, "cl": w.cl * cl_term, "cm": w.cm * cm_term}
    else:
        raise ValueError(f"unknown objective mode '{cfg.mode}'")
    return Score(sum(terms.values()), terms, cfg.mode)
