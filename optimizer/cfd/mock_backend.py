"""Smooth analytic surrogate used for dry runs and tests (NOT physics-grade CFD).

It reads the planform (aspect ratio, sweep, taper, washout, thickness) and
returns plausible CL/CD/CM so the whole optimisation loop can be exercised
without spending any SimScale credits.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..atmosphere import FlowConditions
from ..config import Config
from ..planform import Planform
from .base import CFDResult


class MockBackend:
    name = "mock"

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def run(self, cfg: Config, pf: Planform, flow: FlowConditions, domain_stl: Path,
            work_dir: Path, label: str) -> CFDResult:
        st = pf.stations
        ar = pf.aspect_ratio
        lam = math.radians(pf.mean_qc_sweep_deg)
        mach = min(flow.mach, 0.6)
        a_rad = math.radians(flow.aoa_deg)
        eta = 0.95
        cl_alpha = 2.0 * math.pi * ar / (2.0 + math.sqrt(4.0 + (ar / eta) ** 2 * (1.0 + math.tan(lam) ** 2 - mach ** 2)))

        # area-weighted mean twist (deg) and thickness
        tw, area = 0.0, 0.0
        for a, b in zip(st[:-1], st[1:]):
            s = 0.5 * (a.chord + b.chord) * (b.y - a.y)
            tw += s * 0.5 * (a.twist_deg + b.twist_deg)
            area += s
        tw /= area
        t_mean = sum(s.thickness_ratio for s in st) / len(st)

        cl = cl_alpha * (a_rad + math.radians(0.3) + math.radians(0.6 * tw))
        washout_opt = -2.2
        e = max(0.5, 0.97 - 0.015 * (tw - washout_opt) ** 2 - 0.0015 * abs(pf.mean_qc_sweep_deg - 38.0))
        cd0 = 0.0058 * (1.0 + 2.0 * t_mean + 60.0 * t_mean ** 4) * (1.0 + 0.1 * (pf.mean_qc_sweep_deg / 60.0) ** 2)
        cd = cd0 + cl ** 2 / (math.pi * ar * e)

        x_ac = pf.x_mac_le + (0.25 + 0.10 * math.tan(lam) * 0.3) * pf.mac
        cm0 = -0.018 - 0.008 * tw
        cm = cm0 + cl * (pf.moment_ref[0] - x_ac) / pf.mac

        if self.noise > 0.0:
            cl *= 1.0 + self.noise * self.rng.standard_normal()
            cd *= 1.0 + self.noise * self.rng.standard_normal()
            cm += self.noise * 0.01 * self.rng.standard_normal()
        return CFDResult(ok=True, cl=cl, cd=cd, cm=cm, message="mock surrogate (not CFD)",
                         extra={"cl_alpha": cl_alpha, "e": e, "mean_twist_deg": tw})
