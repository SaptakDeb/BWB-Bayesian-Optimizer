"""STAGE 4 - OPTIMISE: PyTorch Bayesian optimisation, warm-started from everything stage 2 produced.

GP surrogate (exact, float64, CPU or CUDA) + log-EI / LCB / log-PI maximised with autograd.
--batch q proposes q designs per step (Kriging believer) and --parallel runs them concurrently.
Re-run to resume; change --objective to re-score stored CFD results without repeating any CFD.
"""
from __future__ import annotations

import json
from typing import List, Optional

from ..cfd import make_backend
from ..cli import base_parser, load
from ..parametrisation import DesignSpace
from ..pipeline import Evaluator, run_optimisation


def main(argv: Optional[List[str]] = None) -> int:
    p = base_parser(__doc__)
    p.add_argument("--max-evals", type=int, default=None, help="cap on the total number of evaluations")
    p.add_argument("--batch", type=int, default=None, help="designs proposed per BO step")
    p.add_argument("--parallel", type=int, default=1, help="concurrent CFD runs within a batch")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="PyTorch device for the surrogate")
    args = p.parse_args(argv)
    if args.device:
        args.overrides.append(f"optimisation.device={args.device}")
    cfg = load(args)
    space = DesignSpace(cfg)
    evaluator = Evaluator(cfg, space, make_backend(args.backend, cfg), cfg.project.workdir)
    print(f"== STAGE 4: OPTIMISE  (backend={args.backend}, objective={cfg.objective.mode}, "
          f"acquisition={cfg.optimisation.acquisition}, device={cfg.optimisation.device})")
    summary = run_optimisation(cfg, evaluator, args.max_evals, args.batch, args.parallel)
    print(json.dumps(summary, indent=2))
    return 0
