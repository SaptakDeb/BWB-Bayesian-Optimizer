"""Common CFD interface and result container."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol

from ..atmosphere import FlowConditions
from ..config import Config
from ..planform import Planform


@dataclass
class CFDResult:
    ok: bool
    cl: float = float("nan")
    cd: float = float("nan")
    cm: float = float("nan")
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ld(self) -> float:
        if self.cd and self.cd == self.cd and abs(self.cd) > 1e-12:
            return self.cl / self.cd
        return float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "cl": self.cl, "cd": self.cd, "cm": self.cm, "ld": self.ld,
                "message": self.message, "extra": self.extra}


class CFDBackend(Protocol):
    name: str

    def run(self, cfg: Config, pf: Planform, flow: FlowConditions, domain_stl: Path,
            work_dir: Path, label: str) -> CFDResult: ...


@dataclass
class ReferenceValues:
    """Reference quantities used to non-dimensionalise forces on the SIMULATED model."""

    area: float          # S of the simulated model (half area for half model) [m^2]
    length: float        # MAC [m]
    moment_ref: tuple    # (x, y, z) [m]
    q_inf: float         # dynamic pressure [Pa]


def reference_values(cfg: Config, pf: Planform, flow: FlowConditions) -> ReferenceValues:
    area = pf.area_half if cfg.geometry.symmetry_mode == "half" else pf.area_full
    return ReferenceValues(area=area, length=pf.mac, moment_ref=tuple(pf.moment_ref), q_inf=flow.dynamic_pressure)
