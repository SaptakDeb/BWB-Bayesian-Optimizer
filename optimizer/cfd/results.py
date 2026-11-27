"""Turn SimScale force/moment histories into CL, CD, CM.

SimScale returns the force and moment histories as CSV downloads.  Column
naming is detected heuristically (override with simscale.results_column_map if
your export differs; `scripts/inspect_run.py` prints the real headers).
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..atmosphere import FlowConditions
from .base import ReferenceValues


@dataclass
class Table:
    headers: List[str]
    data: np.ndarray            # (rows, cols) float, NaN where not numeric


def parse_csv(text: str) -> Table:
    text = text.lstrip("\ufeff")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("empty CSV")
    # the header is the last non-numeric row before the first numeric row
    def is_numeric_row(r: Sequence[str]) -> bool:
        ok = 0
        for c in r:
            try:
                float(c)
                ok += 1
            except ValueError:
                pass
        return ok >= max(1, len(r) // 2)

    first_num = next((i for i, r in enumerate(rows) if is_numeric_row(r)), None)
    if first_num is None:
        raise ValueError("CSV has no numeric rows")
    headers = [h.strip() for h in rows[first_num - 1]] if first_num > 0 else [f"col{i}" for i in range(len(rows[0]))]
    width = max(len(r) for r in rows[first_num:])
    if len(headers) < width:
        headers = headers + [f"col{i}" for i in range(len(headers), width)]
    out = np.full((len(rows) - first_num, width), np.nan)
    for i, r in enumerate(rows[first_num:]):
        for j, c in enumerate(r):
            try:
                out[i, j] = float(c)
            except ValueError:
                pass
    return Table(headers, out)


def _norm(h: str) -> str:
    h = re.sub(r"[\(\[].*?[\)\]]", " ", h.lower())
    return re.sub(r"[^a-z0-9]+", " ", h).strip()


def _component_regex(kind: str, axis: str) -> "re.Pattern[str]":
    if kind == "force":
        return re.compile(rf"(?:^|\b)(?:f\s?{axis}|force\s?{axis}|{axis}\s?force|force\s+in\s+{axis})(?:\b|$)")
    return re.compile(rf"(?:^|\b)(?:m\s?{axis}|moment\s?{axis}|{axis}\s?moment|moment\s+about\s+{axis})(?:\b|$)")


def find_column(table: Table, kind: str, axis: str, override: Optional[str] = None) -> np.ndarray:
    """Return the history of one force/moment component as a 1-D array."""
    if override:
        for i, h in enumerate(table.headers):
            if h.strip().lower() == override.strip().lower():
                return table.data[:, i]
        raise KeyError(f"column '{override}' not found in {table.headers}")
    rx = _component_regex(kind, axis)
    cands = [(i, _norm(h)) for i, h in enumerate(table.headers) if rx.search(_norm(h))]
    if not cands:
        raise KeyError(f"no {kind}-{axis} column in {table.headers}")
    total = [i for i, h in cands if "total" in h]
    if total:
        return table.data[:, total[0]]
    pres = [i for i, h in cands if "press" in h]
    visc = [i for i, h in cands if "visc" in h or "fric" in h]
    if pres and visc:
        return table.data[:, pres[0]] + table.data[:, visc[0]]
    return table.data[:, cands[0][0]]


def window_mean(x: np.ndarray, window: int) -> Tuple[float, float]:
    """Mean of the last `window` finite samples and its relative std (convergence indicator)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite samples in the history")
    tail = x[-window:] if window > 0 else x
    mean = float(tail.mean())
    rel = float(tail.std() / abs(mean)) if abs(mean) > 1e-30 else float("inf")
    return mean, rel


def coefficients(force: Sequence[float], moment_y: float, flow: FlowConditions, ref: ReferenceValues):
    """Body-axis force (x aft, y span, z up) and pitching moment about the reference point -> CL, CD, CM."""
    d = np.array(flow.flow_direction())
    l = np.array(flow.lift_direction())
    f = np.asarray(force, dtype=float)
    qs = ref.q_inf * ref.area
    cl = float(f @ l / qs)
    cd = float(f @ d / qs)
    cm = float(moment_y / (qs * ref.length))
    return cl, cd, cm


def coefficients_from_tables(force_tables: Sequence[Table], moment_tables: Sequence[Table], flow: FlowConditions,
                             ref: ReferenceValues, window: int, column_map: Dict[str, str], scale: float = 1.0):
    """Combine the force and moment tables into converged coefficients plus a convergence indicator."""
    def pick(tables: Sequence[Table], kind: str, axis: str, key: str) -> np.ndarray:
        last: Optional[Exception] = None
        for t in tables:
            try:
                return find_column(t, kind, axis, column_map.get(key))
            except KeyError as exc:
                last = exc
        raise KeyError(str(last))

    comps = []
    rel = []
    for axis in ("x", "y", "z"):
        mean, r = window_mean(pick(force_tables, "force", axis, f"f{axis}"), window)
        comps.append(mean * scale)
        rel.append(r)
    my, r_my = window_mean(pick(moment_tables, "moment", "y", "my"), window)
    cl, cd, cm = coefficients(comps, my * scale, flow, ref)
    return cl, cd, cm, {"force_rel_std": rel, "moment_rel_std": r_my, "forces": comps, "my": my * scale}
