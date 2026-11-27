"""STAGE 3 - REVIEW: no CFD is run here.  Reads evaluations.jsonl and reports what was learned.

Outputs in <workdir>/review/: summary.md, evaluations.csv, best_design.yaml, review.png
* table of every evaluation, re-scored under the CURRENT objective (--objective to change it)
* best design vs baseline (% change in CL, CD, L/D, CM)
* --compare-objectives: the winner under each of the four objective modes (re-scored, no CFD)
* Pareto set over (max L/D, min |CM - target|, min |CL - target|)
* PyTorch GP diagnostics: parameter relevance (ARD) and leave-one-out accuracy of the surrogate
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from ..cfd import make_backend
from ..cli import base_parser, load
from ..config import OBJECTIVE_MODES
from ..optimiser import BayesianOptimiser, Database, EvalRecord
from ..parametrisation import DesignSpace
from ..pipeline import Evaluator


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows when EVERY column is to be minimised."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def _pct(a: float, b: float) -> str:
    return "n/a" if b == 0 else f"{100.0 * (a - b) / abs(b):+.1f}%"


def main(argv: Optional[List[str]] = None) -> int:
    p = base_parser(__doc__)
    p.add_argument("--compare-objectives", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top", type=int, default=10, help="rows shown in the table")
    args = p.parse_args(argv)
    args.backend = "mock"                                   # nothing is run here
    cfg = load(args)
    space = DesignSpace(cfg)
    ev = Evaluator(cfg, space, make_backend("mock", cfg), cfg.project.workdir)
    rows: List[EvalRecord] = Database(Path(cfg.project.workdir) / "evaluations.jsonl").load()
    out = Path(cfg.project.workdir) / "review"
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no evaluations yet - run stage 2 first")
        return 1

    costs = [ev.rescore(r) for r in rows]
    ok_idx = [i for i, r in enumerate(rows) if r.status == "ok" and costs[i] is not None]
    base = next((r for r in rows if r.label == "baseline" and r.metrics), None)
    n_fail = sum(r.status == "failed" for r in rows)
    n_inf = sum(r.status == "infeasible" for r in rows)
    lines: List[str] = [f"# Review: {cfg.project.name}", "",
                        f"objective **{cfg.objective.mode}**, analysis **{cfg.flow.analysis_type}**, "
                        f"{len(rows)} records ({len(ok_idx)} ok, {n_fail} failed, {n_inf} infeasible)", ""]

    # ---- table ------------------------------------------------------------ #
    order = sorted(ok_idx, key=lambda i: costs[i])
    print(f"== STAGE 3: REVIEW  (objective={cfg.objective.mode}, {len(rows)} records: {len(ok_idx)} ok, "
          f"{n_fail} failed, {n_inf} infeasible)")
    hdr = f"{'label':12s} {'CL':>7s} {'CD':>8s} {'CM':>8s} {'L/D':>7s} {'cost':>9s}"
    print(hdr)
    lines += ["| label | CL | CD | CM | L/D | cost |", "|---|---|---|---|---|---|"]
    for i in order[: args.top]:
        m = rows[i].metrics or {}
        print(f"{rows[i].label:12s} {m['cl']:7.4f} {m['cd']:8.5f} {m['cm']:8.4f} {m['ld']:7.2f} {costs[i]:9.4f}")
        lines.append(f"| {rows[i].label} | {m['cl']:.4f} | {m['cd']:.5f} | {m['cm']:.4f} | {m['ld']:.2f} | {costs[i]:.4f} |")
    if not ok_idx:
        print("no successful evaluations to review")
        return 1

    # ---- best vs baseline -------------------------------------------------- #
    bi = order[0]
    best = rows[bi]
    lines += ["", f"## Best design: `{best.label}` (cost {costs[bi]:.4f})", ""]
    if base is not None and best.metrics:
        bm, mm = base.metrics, best.metrics
        print(f"\nbest = {best.label}; vs baseline:  " + "  ".join(
            f"{k.upper()} {bm[k]:.4f}->{mm[k]:.4f} ({_pct(mm[k], bm[k])})" for k in ("cl", "cd", "cm", "ld")))
        for k in ("cl", "cd", "cm", "ld"):
            lines.append(f"- {k.upper()}: {bm[k]:.5f} -> {mm[k]:.5f} ({_pct(mm[k], bm[k])})")
    lines += ["", "design variables: " + ", ".join(f"{k}={v:.3f}" for k, v in best.x.items())]
    g = space.apply(best.x)
    import dataclasses

    (out / "best_design.yaml").write_text(yaml.safe_dump({
        "label": best.label, "objective": cfg.objective.mode, "cost": costs[bi], "metrics": best.metrics,
        "design": best.x, "geometry": {"root_chord": g.root_chord, "airfoil_stations": g.airfoil_stations,
                                       "sections": [dataclasses.asdict(s) for s in g.sections]}}, sort_keys=False))

    # ---- objective comparison ---------------------------------------------- #
    if args.compare_objectives:
        print("\nwinner under each objective (re-scored, no CFD):")
        lines += ["", "## Winner per objective", ""]
        for mode in OBJECTIVE_MODES:
            obj = copy.deepcopy(cfg.objective)
            obj.mode = mode
            sc = [(ev.rescore(rows[i], obj), i) for i in ok_idx]
            c, i = min(sc)
            print(f"  {mode:10s} -> {rows[i].label:12s} cost {c:.4f}  L/D {rows[i].metrics['ld']:.2f}")
            lines.append(f"- **{mode}**: {rows[i].label} (cost {c:.4f}, L/D {rows[i].metrics['ld']:.2f})")

    # ---- Pareto -------------------------------------------------------------- #
    pts = np.array([[-rows[i].metrics["ld"], abs(rows[i].metrics["cm"] - cfg.objective.cm_target),
                     abs(rows[i].metrics["cl"] - cfg.objective.cl_target)] for i in ok_idx])
    pmask = pareto_mask(pts)
    pareto = [rows[i].label for i, k in zip(ok_idx, pmask) if k]
    print(f"\nPareto set (max L/D, min |CM-target|, min |CL-target|): {pareto}")
    lines += ["", "## Pareto set", "", ", ".join(pareto)]

    # ---- surrogate diagnostics (PyTorch GP) ------------------------------------ #
    relevance: Optional[np.ndarray] = None
    loo: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if len(ok_idx) >= 4:
        bo = BayesianOptimiser(space, cfg.optimisation)
        for i, r in enumerate(rows):
            if costs[i] is not None and r.status in ("ok", "failed"):
                bo.tell(r.x, costs[i], failed=(r.status == "failed"))
        gp = bo.fit()
        relevance = gp.relevance().cpu().numpy()
        mu, _ = gp.loo()
        y = bo.training_targets()
        loo = (y, mu.cpu().numpy())
        ss_res = float(np.sum((y - loo[1]) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        rank = sorted(zip(space.names, relevance), key=lambda t: -t[1])
        print(f"\nsurrogate (PyTorch GP on {gp.X.shape[0]} points, device {gp.device}): LOO R^2 = {r2:.2f}, "
              f"LOO RMSE = {np.sqrt(ss_res / len(y)):.4f}")
        print("most influential variables: " + ", ".join(f"{n} ({v:.0%})" for n, v in rank[:5]))
        lines += ["", "## Surrogate", "", f"LOO R^2 = {r2:.2f}", "",
                  "relevance: " + ", ".join(f"{n} {v:.0%}" for n, v in rank)]
        if r2 < 0.3:
            print("  note: surrogate is weak - more evaluations (stage 2/4) are needed before trusting its trends")
    else:
        print("\n(fewer than 4 successful evaluations: surrogate diagnostics skipped)")

    # ---- CSV + plots --------------------------------------------------------------- #
    with (out / "evaluations.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "status", "cl", "cd", "cm", "ld", "cost"] + space.names)
        for i, r in enumerate(rows):
            m = r.metrics or {}
            w.writerow([r.label, r.status, m.get("cl"), m.get("cd"), m.get("cm"), m.get("ld"), costs[i]]
                       + [r.x.get(n) for n in space.names])
    if not args.no_plots:
        try:
            _plots(out / "review.png", rows, costs, ok_idx, pmask, space, relevance, loo)
            print(f"\nplots: {out / 'review.png'}")
        except ImportError:
            print("\n(matplotlib not installed: plots skipped)")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"report: {out / 'summary.md'}   csv: {out / 'evaluations.csv'}   best: {out / 'best_design.yaml'}")
    return 0


def _plots(path: Path, rows, costs, ok_idx, pmask, space, relevance, loo) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    seq = [i for i in range(len(rows)) if rows[i].status == "ok" and costs[i] is not None]
    c = [costs[i] for i in seq]
    ax[0, 0].plot(range(len(c)), c, "o", alpha=0.6, label="evaluation")
    ax[0, 0].plot(range(len(c)), np.minimum.accumulate(c), "-", lw=2, label="best so far")
    ax[0, 0].set(title="Convergence", xlabel="evaluation #", ylabel="cost (minimised)")
    ax[0, 0].legend()
    ld = [rows[i].metrics["ld"] for i in ok_idx]
    cm = [rows[i].metrics["cm"] for i in ok_idx]
    sc = ax[0, 1].scatter(cm, ld, c=[costs[i] for i in ok_idx], cmap="viridis_r")
    px = [m for m, k in zip(cm, pmask) if k]
    py = [l for l, k in zip(ld, pmask) if k]
    ax[0, 1].plot(px, py, "r^", ms=9, mfc="none", label="Pareto")
    ax[0, 1].set(title="L/D vs CM", xlabel="CM", ylabel="L/D")
    ax[0, 1].legend()
    fig.colorbar(sc, ax=ax[0, 1], label="cost")
    if relevance is not None:
        idx = np.argsort(relevance)
        ax[1, 0].barh([space.names[i] for i in idx], relevance[idx])
        ax[1, 0].set(title="GP parameter relevance (1/lengthscale)")
        y, mu = loo
        ax[1, 1].plot(y, mu, "o")
        lim = [min(y.min(), mu.min()), max(y.max(), mu.max())]
        ax[1, 1].plot(lim, lim, "k--")
        ax[1, 1].set(title="Surrogate leave-one-out", xlabel="actual cost", ylabel="LOO prediction")
    else:
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
