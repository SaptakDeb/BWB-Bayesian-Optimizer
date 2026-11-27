#!/usr/bin/env python3
"""Run the bundled test case through all four stages:  1 setup -> 2 run -> 4 optimise -> 3 review.

    python scripts/run_test_case.py                     # FREE: mock CFD surrogate + analytic geometry, ~1 minute
    python scripts/run_test_case.py --objective max_ld  # same, other objective
    python scripts/run_test_case.py --backend simscale --yes
                                                         # REAL CFD with the budgets in config.yaml (costs credits!)

The test case is config/config.yaml: a 5-section BWB half-wing (4 m half-span, mirrored about xz), multi-objective
(L/D, CD, CL target, CM target), incompressible at 100 m/s, SimScale hex-dominant mesh with regular k-omega SST.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bwbopt.stages import optimise_stage, review_stage, run_stage, setup_stage  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "simscale"], default="mock")
    p.add_argument("--objective", choices=["max_ld", "min_cd", "max_ld_cm", "multi"], default=None)
    p.add_argument("--analysis", choices=["incompressible", "compressible"], default=None)
    p.add_argument("--yes", action="store_true", help="confirm spending SimScale credits")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    common = ["--backend", args.backend]
    if args.config:
        common += ["--config", args.config]
    if args.objective:
        common += ["--objective", args.objective]
    if args.analysis:
        common += ["--analysis", args.analysis]
    if args.backend == "mock":                       # tiny budgets so the dry run is quick
        common += ["--set", "project.workdir=runs/test_case_mock", "--set", "optimisation.n_initial=6",
                   "--set", "optimisation.n_iterations=6"]
    elif not args.yes:
        sys.exit("The simscale backend spends compute credits. Re-run with --yes to confirm "
                 "(consider `python scripts/04_optimise.py --max-evals 3` first).")

    for name, stage, extra in (("setup", setup_stage, ["--check-api"] if args.backend == "simscale" else []),
                               ("run", run_stage, []), ("optimise", optimise_stage, []),
                               ("review", review_stage, ["--compare-objectives"])):
        print(f"\n{'=' * 78}\n>>> {name}\n{'=' * 78}")
        code = stage.main(common + extra)
        if code != 0:
            print(f"stage '{name}' returned {code}; stopping")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
