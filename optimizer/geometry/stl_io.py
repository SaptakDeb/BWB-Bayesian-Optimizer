"""Multi-solid ASCII STL writer (one named solid per boundary patch).

SimScale / snappyHexMesh pick patch names up from the `solid <name>` records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def write_multisolid_stl(path: Path, patches: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """patches: name -> (vertices (n,3), faces (m,3) int)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for name, (verts, faces) in patches.items():
            if len(faces) == 0:
                continue
            fh.write(f"solid {name}\n")
            tri = verts[faces]                                   # (m,3,3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            n = n / norm
            for k in range(len(faces)):
                fh.write(f"  facet normal {n[k, 0]:.9e} {n[k, 1]:.9e} {n[k, 2]:.9e}\n")
                fh.write("    outer loop\n")
                for v in tri[k]:
                    fh.write(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {name}\n")


def write_single_solid_stl(path: Path, verts: np.ndarray, faces: np.ndarray, name: str = "wing") -> None:
    write_multisolid_stl(path, {name: (verts, faces)})
