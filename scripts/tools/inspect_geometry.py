#!/usr/bin/env python3
"""List the entities SimScale created for an imported geometry (use it once after your first import).

    python scripts/tools/inspect_geometry.py --project-id <id> --geometry-id <id>

If your named STL patches are not found automatically, copy the entity names printed here into
simscale.manual_entities and set simscale.entity_resolution: manual.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bwbopt.cli import base_parser, load  # noqa: E402

from bwbopt.cfd.simscale_backend import _labels, _query_entities
from bwbopt.cfd.simscale_client import make_client


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--project-id", required=True)
    p.add_argument("--geometry-id", required=True)
    args = p.parse_args()
    load(args)
    client = make_client()
    for e in _query_entities(client, args.project_id, args.geometry_id):
        print(f"{e.name:24s} class={getattr(e, '_class', '?'):5s} labels={_labels(e)}")


if __name__ == "__main__":
    main()
