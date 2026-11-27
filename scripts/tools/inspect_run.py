#!/usr/bin/env python3
"""Print the force/moment result headers of a finished SimScale run (to fix column detection).

    python scripts/tools/inspect_run.py --project-id <id> --simulation-id <id> --run-id <id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bwbopt.cli import base_parser, load  # noqa: E402

from bwbopt.cfd.results import parse_csv
from bwbopt.cfd.simscale_client import make_client


def main() -> None:
    p = base_parser(__doc__)
    for a in ("project-id", "simulation-id", "run-id"):
        p.add_argument(f"--{a}", required=True)
    args = p.parse_args()
    load(args)
    c = make_client()
    for cat in ("FORCE_PLOT", "MOMENT_PLOT"):
        res = c.simulation_run_api.get_simulation_run_results(
            args.project_id, args.simulation_id, args.run_id, page=1, limit=100, category=cat)
        for item in res.embedded or []:
            t = parse_csv(c.download(item.download.url).decode("utf-8", errors="replace"))
            print(cat, getattr(item, "name", ""), "headers:", t.headers, "rows:", len(t.data))


if __name__ == "__main__":
    main()
