"""One design evaluation, batch evaluation, and the optimisation driver.

evaluate(x):  design vector -> candidate geometry -> feasibility check -> wing surface (OpenVSP)
              -> CFD enclosure -> CFD backend (SimScale / mock) -> objective cost
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .atmosphere import flow_conditions
from .cfd.base import CFDResult
from .cfd.mesh_requirements import plan_mesh
from .config import Config, ObjectiveCfg, config_to_dict
from .constraints import check_geometry
from .geometry import build_wing_surface
from .geometry.enclosure import build_enclosure
from .objective import Baseline, Score, score
from .optimiser import BayesianOptimiser, Database, EvalRecord
from .parametrisation import DesignSpace
from .planform import compute_planform

log = logging.getLogger(__name__)
Job = Tuple[Dict[str, float], int, Optional[str]]          # (design vector, index, label)


class Evaluator:
    def __init__(self, cfg: Config, space: DesignSpace, backend: Any, workdir: Path):
        self.cfg = cfg
        self.space = space
        self.backend = backend
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.baseline: Optional[Baseline] = None
        self._baseline_path = self.workdir / "baseline.json"
        self._geom_lock = threading.Lock()               # OpenVSP keeps ONE global model: build serially
        if self._baseline_path.is_file():
            self.baseline = Baseline(**json.loads(self._baseline_path.read_text()))

    # -- cheap feasibility (no CFD) ---------------------------------------- #
    def violations(self, x: Dict[str, float]) -> List[str]:
        g = self.space.apply(x)
        return check_geometry(g, compute_planform(g), self.cfg.optimisation.constraints)

    def feasible(self, x: Dict[str, float]) -> bool:
        try:
            return not self.violations(x)
        except Exception:                                  # noqa: BLE001 - malformed geometry = infeasible
            return False

    # -- full evaluation --------------------------------------------------- #
    def evaluate(self, x: Dict[str, float], index: int, label: Optional[str] = None) -> EvalRecord:
        label = label or f"eval_{index:04d}"
        t0 = time.time()
        edir = self.workdir / "evals" / label
        edir.mkdir(parents=True, exist_ok=True)
        cfg_i = copy.deepcopy(self.cfg)
        cfg_i.geometry = self.space.apply(x)
        pf = compute_planform(cfg_i.geometry)
        bad = check_geometry(cfg_i.geometry, pf, cfg_i.optimisation.constraints)
        rec = EvalRecord(index=index, label=label, x=dict(x), status="infeasible", mode=self.cfg.objective.mode)
        if bad:
            rec.message = "; ".join(bad)
        else:
            try:
                flow = flow_conditions(cfg_i.flow.altitude_m, cfg_i.flow.speed, cfg_i.flow.aoa_deg)
                with self._geom_lock:
                    wing = build_wing_surface(cfg_i.geometry, edir / "geometry")
                    enc = build_enclosure(wing, cfg_i.geometry, pf, cfg_i.domain, edir / "domain.stl")
                (edir / "planform.json").write_text(json.dumps(pf.to_dict(), indent=2))
                (edir / "mesh_plan.json").write_text(json.dumps(plan_mesh(cfg_i, pf, flow).to_dict(), indent=2))
                res: CFDResult = self.backend.run(cfg_i, pf, flow, enc.path, edir, label)
            except Exception as exc:                       # noqa: BLE001
                log.exception("evaluation %s failed before/while running CFD", label)
                res = CFDResult(ok=False, message=f"{type(exc).__name__}: {exc}")
            self._fill(rec, res, self.baseline)
        rec.wall_s = time.time() - t0
        (edir / "record.json").write_text(rec.to_json())
        return rec

    def _fill(self, rec: EvalRecord, res: CFDResult, baseline: Optional[Baseline]) -> None:
        rec.message = res.message
        keep = ("project_id", "geometry_id", "simulation_id", "mesh_operation_id", "run_id", "force_rel_std",
                "moment_rel_std")
        rec.extra = {k: v for k, v in res.extra.items() if k in keep}
        if res.ok:
            rec.status = "ok"
            rec.metrics = {"cl": res.cl, "cd": res.cd, "cm": res.cm, "ld": res.ld}
        else:
            rec.status = "failed"
        sc: Score = score(self.cfg.objective, res, baseline)
        rec.score, rec.terms = sc.total, sc.terms

    def set_baseline(self, rec: EvalRecord) -> None:
        if rec.status != "ok" or not rec.metrics:
            raise RuntimeError(f"baseline evaluation failed: {rec.message}")
        self.baseline = Baseline(rec.metrics["cl"], rec.metrics["cd"], rec.metrics["cm"])
        self._baseline_path.write_text(json.dumps(self.baseline.to_dict()))

    def rescore(self, rec: EvalRecord, objective: Optional[ObjectiveCfg] = None) -> Optional[float]:
        """Cost of a stored record under `objective` (default: the current one). None for infeasible."""
        obj = objective or self.cfg.objective
        if rec.status == "infeasible":
            return None
        if rec.status == "failed" or not rec.metrics:
            return obj.penalties.failure_value
        res = CFDResult(ok=True, cl=rec.metrics["cl"], cd=rec.metrics["cd"], cm=rec.metrics["cm"])
        return score(obj, res, self.baseline).total


# --------------------------------------------------------------------------- #
def next_index(records: Sequence[EvalRecord]) -> int:
    return max((r.index for r in records), default=0) + 1


def evaluate_many(evaluator: Evaluator, db: Database, jobs: Sequence[Job], parallel: int = 1) -> List[EvalRecord]:
    """Evaluate designs, optionally concurrently (geometry is serialised, CFD runs overlap)."""
    jobs = list(jobs)
    if not jobs:
        return []
    if hasattr(evaluator.backend, "project_id"):            # create the SimScale project once, before threads
        evaluator.backend.project_id()

    def work(job: Job) -> EvalRecord:
        x, idx, label = job
        rec = evaluator.evaluate(x, idx, label)
        db.append(rec)
        log.info("%s: %s %s", rec.label, rec.status, rec.metrics or rec.message)
        return rec

    if parallel <= 1 or len(jobs) == 1:
        return [work(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(work, jobs))


def run_baseline(cfg: Config, evaluator: Evaluator, db: Database) -> EvalRecord:
    existing = [r for r in db.load() if r.label == "baseline"]
    if existing and evaluator.baseline is not None:
        return existing[-1]
    rec = evaluator.evaluate(evaluator.space.nominal(), index=0, label="baseline")
    evaluator.set_baseline(rec)                              # needs ok metrics, else raises
    evaluator._fill(rec, CFDResult(True, rec.metrics["cl"], rec.metrics["cd"], rec.metrics["cm"]), evaluator.baseline)
    db.append(rec)
    return rec


def restore_optimiser(opt: BayesianOptimiser, evaluator: Evaluator, db: Database, base_rec: EvalRecord) -> List[EvalRecord]:
    """Feed every stored evaluation (rescored with the CURRENT objective) back into the optimiser."""
    space, cfg = evaluator.space, evaluator.cfg
    history = [r for r in db.load() if r.label != "baseline"]
    in_bounds = not space.nominal_outside_bounds()
    if cfg.optimisation.include_baseline and in_bounds and base_rec.status == "ok":
        y0 = evaluator.rescore(base_rec)
        if y0 is not None:
            opt.tell(space.nominal(), y0)
    for r in history:
        y = evaluator.rescore(r)
        if y is not None:
            opt.tell(r.x, y, failed=(r.status == "failed"))
    return history


def run_optimisation(cfg: Config, evaluator: Evaluator, max_evals: Optional[int] = None,
                     batch: Optional[int] = None, parallel: int = 1) -> Dict[str, Any]:
    space, workdir = evaluator.space, evaluator.workdir
    (workdir / "config_used.json").write_text(json.dumps(config_to_dict(cfg), indent=2, default=str))
    db = Database(workdir / "evaluations.jsonl")
    opt = BayesianOptimiser(space, cfg.optimisation, feasible=evaluator.feasible)

    base_rec = run_baseline(cfg, evaluator, db)
    in_bounds = not space.nominal_outside_bounds()
    opt.plan_initial(space.nominal() if (cfg.optimisation.include_baseline and in_bounds) else None)
    history = restore_optimiser(opt, evaluator, db, base_rec)

    budget = cfg.optimisation.n_initial + cfg.optimisation.n_iterations
    if max_evals is not None:
        budget = min(budget, max_evals)
    batch = max(1, batch or cfg.optimisation.batch_size)
    index = next_index(history + [base_rec])
    best_y = min(opt.y) if opt.y else float("inf")
    since_improve, infeasible_streak = 0, 0

    while opt.n_told < budget:
        q = min(batch, budget - opt.n_told)
        xs = opt.ask_batch(q)
        recs = evaluate_many(evaluator, db, [(x, index + i, None) for i, x in enumerate(xs)], parallel)
        index += len(xs)
        for x, rec in zip(xs, recs):
            if rec.status == "infeasible":
                infeasible_streak += 1
                if infeasible_streak > 50:
                    raise RuntimeError("too many infeasible proposals in a row")
                continue
            infeasible_streak = 0
            y = rec.score if rec.score is not None else cfg.objective.penalties.failure_value
            opt.tell(x, y, failed=(rec.status == "failed"))
            if y < best_y - 1e-9:
                best_y, since_improve = y, 0
            else:
                since_improve += 1
        log.info("BO: %d/%d evaluations, best cost %.4f", opt.n_told, budget, min(opt.y))
        pat = cfg.optimisation.patience
        if pat is not None and opt.n_told > cfg.optimisation.n_initial and since_improve >= pat:
            log.info("stopping: no improvement for %d evaluations", pat)
            break

    bx, by = opt.best()
    summary = {"best_x": bx, "best_cost": by, "n_evaluations": opt.n_told, "objective_mode": cfg.objective.mode,
               "device": str(opt.device)}
    (workdir / "best.json").write_text(json.dumps(summary, indent=2))
    return summary
