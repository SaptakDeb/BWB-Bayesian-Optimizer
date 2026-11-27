"""STAGE 1 - SETUP: verify the environment, validate the config, derive everything, build the baseline CAD.

Writes <workdir>/setup/: setup_report.json, config_resolved.yaml, planform.json, mesh_plan.json, wing/domain STL.
Spends no CFD credits.  Exit code 1 when any check FAILS.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..atmosphere import flow_conditions
from ..cfd.mesh_requirements import plan_mesh
from ..cli import base_parser, load
from ..config import config_to_dict
from ..constraints import check_geometry
from ..parametrisation import DesignSpace
from ..planform import compute_planform

Check = Tuple[str, str, str]                      # (status OK|WARN|FAIL, name, detail)


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def check_environment(cfg: Any, backend: str, check_api: bool) -> List[Check]:
    out: List[Check] = []
    ok = lambda n, d="": out.append(("OK", n, d))
    warn = lambda n, d="": out.append(("WARN", n, d))
    fail = lambda n, d="": out.append(("FAIL", n, d))

    if sys.version_info >= (3, 9):
        ok("python", platform.python_version())
    else:
        fail("python", f"{platform.python_version()} (need >= 3.9)")
    for mod in ("numpy", "scipy", "yaml", "trimesh", "manifold3d"):
        (ok if _has(mod) else fail)(f"module {mod}", "" if _has(mod) else "pip install -r requirements.txt")
    if _has("torch"):
        import torch

        dev = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ok("module torch", f"{torch.__version__}, optimiser device: {dev}")
        if cfg.optimisation.device == "cuda" and not torch.cuda.is_available():
            fail("optimisation.device", "cuda requested but not available")
    else:
        fail("module torch", "the Bayesian optimiser needs PyTorch: pip install torch")
    (ok if _has("matplotlib") else warn)("module matplotlib", "" if _has("matplotlib") else "review plots disabled")

    if cfg.geometry.backend == "openvsp":
        if _has("openvsp"):
            ok("module openvsp", "OpenVSP Python API found")
        else:
            fail("module openvsp", "install the OpenVSP Python API or set geometry.backend: analytic")
    else:
        warn("geometry backend", "analytic lofter in use (OpenVSP not used)")

    if backend == "simscale":
        if _has("simscale_sdk"):
            ok("module simscale_sdk")
        else:
            fail("module simscale_sdk", "pip install simscale-sdk")
        if os.getenv("SIMSCALE_API_KEY"):
            ok("SIMSCALE_API_KEY", "set (value never shown)")
        elif Path(".env").is_file():
            ok("SIMSCALE_API_KEY", ".env file present (loaded at run time)")
        else:
            fail("SIMSCALE_API_KEY", "copy .env.example to .env and add your key")
        if check_api and _has("simscale_sdk"):
            try:
                from ..cfd.simscale_client import make_client

                make_client().project_api.get_projects(limit=1)
                ok("SimScale API", "authenticated")
            except Exception as exc:               # noqa: BLE001
                fail("SimScale API", f"{type(exc).__name__}: {str(exc)[:120]}")
    else:
        ok("CFD backend", "mock surrogate (no SimScale needed)")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = base_parser(__doc__)
    p.add_argument("--no-geometry", action="store_true", help="skip building the baseline wing/enclosure")
    p.add_argument("--check-api", action="store_true", help="also call the SimScale API once to test the key")
    p.add_argument("--create-project", action="store_true", help="create the SimScale project now")
    args = p.parse_args(argv)
    cfg = load(args)
    out_dir = Path(cfg.project.workdir) / "setup"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = check_environment(cfg, args.backend, args.check_api)
    pf = compute_planform(cfg.geometry)
    fl = flow_conditions(cfg.flow.altitude_m, cfg.flow.speed, cfg.flow.aoa_deg)
    plan = plan_mesh(cfg, pf, fl)
    space = DesignSpace(cfg)

    if cfg.flow.analysis_type == "incompressible" and fl.mach > 0.3:
        checks.append(("WARN", "compressibility", f"Mach {fl.mach:.2f} > 0.3: use --analysis compressible"))
    for n, v, lo, hi in space.nominal_outside_bounds():
        checks.append(("WARN", "design space", f"nominal {n}={v} is outside bounds [{lo}, {hi}]"))
    bad = check_geometry(cfg.geometry, pf, cfg.optimisation.constraints)
    checks.append(("FAIL" if bad else "OK", "baseline constraints", "; ".join(bad) or "baseline is feasible"))
    for w in plan.warnings:
        checks.append(("WARN", "mesh plan", w))

    geom_info: Dict[str, Any] = {}
    if not args.no_geometry and not any(s == "FAIL" and ("module" in n or "openvsp" in n) for s, n, _ in checks):
        try:
            from ..geometry import build_wing_surface
            from ..geometry.enclosure import build_enclosure

            wing = build_wing_surface(cfg.geometry, out_dir / "geometry")
            rep = build_enclosure(wing, cfg.geometry, pf, cfg.domain, out_dir / "domain.stl")
            geom_info = {"wing_stl": str(wing), "domain_stl": str(rep.path), "watertight": rep.watertight,
                         "patch_faces": rep.patch_faces, "fluid_volume_m3": rep.volume}
            checks.append(("OK" if rep.watertight else "FAIL", "CFD enclosure",
                           f"watertight={rep.watertight}, patches={rep.patch_faces}"))
        except Exception as exc:                   # noqa: BLE001
            checks.append(("FAIL", "geometry build", f"{type(exc).__name__}: {exc}"))

    if args.create_project and args.backend == "simscale":
        try:
            from ..cfd.simscale_backend import SimScaleBackend

            checks.append(("OK", "SimScale project", f"id {SimScaleBackend(cfg).project_id()}"))
        except Exception as exc:                   # noqa: BLE001
            checks.append(("FAIL", "SimScale project", f"{type(exc).__name__}: {str(exc)[:160]}"))

    (out_dir / "planform.json").write_text(json.dumps(pf.to_dict(), indent=2))
    (out_dir / "mesh_plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
    report = {"checks": [{"status": s, "name": n, "detail": d} for s, n, d in checks], "geometry": geom_info,
              "flow": {"mach": fl.mach, "re_mac": fl.reynolds(pf.mac), "q_pa": fl.dynamic_pressure},
              "planform": {k: getattr(pf, k) for k in ("half_span", "area_full", "mac", "aspect_ratio")}}
    (out_dir / "setup_report.json").write_text(json.dumps(report, indent=2))

    print(f"== STAGE 1: SETUP  ({cfg.project.name}, objective={cfg.objective.mode}, analysis={cfg.flow.analysis_type})")
    for s, n, d in checks:
        print(f"  [{s:4s}] {n:22s} {d}")
    print(f"  half-span {pf.half_span:.3f} m | S_full {pf.area_full:.3f} m^2 | MAC {pf.mac:.3f} m | AR {pf.aspect_ratio:.2f} "
          f"| Mach {fl.mach:.3f} | Re_MAC {fl.reynolds(pf.mac):.2e}")
    print(f"  mesh: wall={plan.wall_treatment}, y+={plan.target_y_plus}, first layer {plan.first_layer:.2e} m, "
          f"{plan.layers} layers (BL coverage {plan.coverage:.0%}), ~{plan.est_cells:.1e} cells")
    print(f"  design space: {space.dim} variables | report: {out_dir / 'setup_report.json'}")
    failed = [c for c in checks if c[0] == "FAIL"]
    print("  RESULT:", "FAILED - fix the items above" if failed else "ready for stage 2 (run)")
    return 1 if failed else 0
