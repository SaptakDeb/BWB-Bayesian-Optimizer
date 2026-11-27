"""STAGE 2 - RUN ALL: push the baseline and a whole design-of-experiments through geometry -> mesh -> CFD.

Default = baseline + a scrambled-Sobol DOE of `optimisation.n_initial` designs (this is also the warm start
for stage 4).  Resumable: designs already in evaluations.jsonl are skipped.  --parallel N overlaps N CFD runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..cfd import make_backend
from ..cli import base_parser, load
from ..optimiser import Database
from ..optimiser.bayes import initial_design
from ..parametrisation import DesignSpace
from ..pipeline import Evaluator, evaluate_many, next_index, run_baseline


def _already_done(x: Dict[str, float], done: List[Dict[str, float]], space: DesignSpace) -> bool:
    u = space.to_unit(x)
    return any(float(np.linalg.norm(space.to_unit(d) - u)) < 1e-3 for d in done)


def main(argv: Optional[List[str]] = None) -> int:
    p = base_parser(__doc__)
    p.add_argument("--only", choices=["baseline", "doe", "all"], default="all")
    p.add_argument("--n", type=int, default=None, help="number of DOE designs (default optimisation.n_initial)")
    p.add_argument("--parallel", type=int, default=1, help="concurrent CFD runs")
    p.add_argument("--designs-file", help="JSON list of design vectors {name: value} to run instead of a DOE")
    args = p.parse_args(argv)
    cfg = load(args)
    space = DesignSpace(cfg)
    evaluator = Evaluator(cfg, space, make_backend(args.backend, cfg), cfg.project.workdir)
    db = Database(Path(cfg.project.workdir) / "evaluations.jsonl")

    print(f"== STAGE 2: RUN  (backend={args.backend}, objective={cfg.objective.mode}, analysis={cfg.flow.analysis_type})")
    base = run_baseline(cfg, evaluator, db)
    print(f"  baseline: {base.status} {base.metrics or base.message}")
    if args.only == "baseline":
        return 0 if base.status == "ok" else 1

    if args.designs_file:
        designs = [space.clip(d) for d in json.loads(Path(args.designs_file).read_text())]
    else:
        n = args.n or cfg.optimisation.n_initial
        nominal = space.nominal() if not space.nominal_outside_bounds() else None
        designs = initial_design(space, n, cfg.optimisation.seed, evaluator.feasible,
                                 nominal if cfg.optimisation.include_baseline else None)
        if nominal is not None and cfg.optimisation.include_baseline:
            designs = designs[1:]                       # the baseline was already run above
    records = db.load()
    done = [r.x for r in records]
    todo = [d for d in designs if not _already_done(d, done, space)]
    print(f"  designs: {len(designs)} planned, {len(designs) - len(todo)} already done, {len(todo)} to run "
          f"(parallel={args.parallel})")
    start = next_index(records)
    new = evaluate_many(evaluator, db, [(x, start + i, None) for i, x in enumerate(todo)], args.parallel)

    ok = sum(r.status == "ok" for r in new)
    print(f"  finished: {ok} ok, {sum(r.status == 'failed' for r in new)} failed, "
          f"{sum(r.status == 'infeasible' for r in new)} infeasible")
    summary = {"baseline": base.metrics, "n_new": len(new), "n_ok": ok}
    (Path(cfg.project.workdir) / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  log: {db.path}")
    return 0 if base.status == "ok" else 1
