"""SimScale CFD backend: upload -> import -> mesh (hex-dominant) -> steady SST run -> CL/CD/CM.

Workflow per design (all calls go through the official simscale-python-sdk):
    1. upload + import the enclosure STL
    2. resolve entity names of the named patches (wing, inlet, outlet, farfield, symmetry, fluid)
    3. create refinement boxes, simulation spec (incompressible | compressible, k-omega SST), material
    4. hex-dominant mesh operation (check -> estimate -> cost guard -> start -> wait)
    5. attach mesh, check, estimate, run, wait
    6. download force/moment histories, average, convert to CL/CD/CM
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..atmosphere import FlowConditions
from ..config import Config
from ..planform import Planform
from .base import CFDResult, reference_values
from .mesh_requirements import plan_mesh
from .results import coefficients_from_tables, parse_csv
from .simscale_client import SimScaleAuthError, SimScaleClient, make_client
from . import simscale_specs as specs

log = logging.getLogger(__name__)
TERMINAL = ("FINISHED", "CANCELED", "FAILED")


class SimScaleRunError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _wait(fetch: Callable[[], Any], label: str, timeout: float, interval: float) -> Any:
    start = time.time()
    obj = fetch()
    while obj.status not in TERMINAL:
        if time.time() - start > timeout:
            raise TimeoutError(f"{label}: no terminal status after {timeout:.0f}s (last: {obj.status})")
        time.sleep(interval)
        obj = fetch()
        log.info("%s: %s %s", label, obj.status, getattr(obj, "progress", ""))
    return obj


def _check_entries(check: Any, label: str) -> None:
    entries = list(getattr(check, "entries", None) or [])
    for e in entries:
        if e.severity == "WARNING":
            log.warning("%s check warning: %s", label, getattr(e, "message", e))
    errors = [e for e in entries if e.severity == "ERROR"]
    if errors:
        raise SimScaleRunError(f"{label} check failed: " + "; ".join(str(getattr(e, "message", e)) for e in errors))


def _estimate(call: Callable[[], Any], label: str, limit: Optional[float], default_runtime: float) -> float:
    """Run an estimation, enforce the cost guard, return a max runtime in seconds."""
    from simscale_sdk import ApiException

    try:
        est = call()
    except ApiException as exc:
        if exc.status == 422:                          # estimation not available for this setup
            log.info("%s estimation not available", label)
            return default_runtime
        raise
    cr = getattr(est, "compute_resource", None)
    if limit is not None and cr is not None and cr.value is not None and cr.value > limit:
        raise SimScaleRunError(f"{label} estimate {cr.value:.1f} compute units exceeds max_compute_units={limit}")
    dur = getattr(est, "duration", None)
    if dur is not None and getattr(dur, "interval_max", None):
        try:
            import isodate

            return max(3600.0, isodate.parse_duration(dur.interval_max).total_seconds() * 2.0)
        except Exception:                              # noqa: BLE001 - estimation parsing is best effort
            return default_runtime
    return default_runtime


def _query_entities(client: SimScaleClient, project_id: str, geometry_id: str, **kwargs: Any) -> List[Any]:
    out: List[Any] = []
    page = 1
    while True:
        res = client.geometry_api.get_geometry_mappings(project_id, geometry_id, page=page, limit=100, **kwargs)
        batch = list(res.embedded or [])
        out.extend(batch)
        total = res.meta.total if getattr(res, "meta", None) is not None else len(out)
        if not batch or len(out) >= total:
            return out
        page += 1


def _labels(entity: Any) -> List[str]:
    labels: List[str] = []
    for ref in getattr(entity, "originate_from", None) or []:
        for v in (getattr(ref, "entity", None), getattr(ref, "body", None)):
            if v:
                labels.append(str(v))
        for attr in getattr(ref, "attribute_list", None) or []:
            if getattr(attr, "value", None):
                labels.append(str(attr.value))
    return labels


def resolve_entities(client: SimScaleClient, project_id: str, geometry_id: str, cfg: Config) -> Dict[str, List[str]]:
    """Map the patch names written into the STL to SimScale's internal entity names."""
    need = ["wing", "inlet", "outlet", "farfield", "fluid"]
    if cfg.geometry.symmetry_mode == "half":
        need.append("symmetry")

    if cfg.simscale.entity_resolution == "manual":
        manual = cfg.simscale.manual_entities
        missing = [k for k in need if not manual.get(k)]
        if missing:
            raise SimScaleRunError(f"simscale.manual_entities is missing {missing}")
        return {k: list(v) for k, v in manual.items()}

    found: Dict[str, List[str]] = {}
    all_entities: Optional[List[Any]] = None
    for key in need:
        if key == "fluid":
            continue
        ents = _query_entities(client, project_id, geometry_id, attributes=["SDL/TYSA_NAME"], values=[key])
        names = [e.name for e in ents if str(getattr(e, "_class", "face")).lower() != "body"]
        if not names:                                    # fall back to a local scan of all entity labels
            if all_entities is None:
                all_entities = _query_entities(client, project_id, geometry_id)
            names = [e.name for e in all_entities
                     if str(getattr(e, "_class", "")).lower() != "body"
                     and any(l.lower() == key for l in _labels(e))]
        if names:
            found[key] = names
    if all_entities is None:
        all_entities = _query_entities(client, project_id, geometry_id)
    bodies = [e.name for e in all_entities if str(getattr(e, "_class", "")).lower() == "body"]
    if bodies:
        found["fluid"] = bodies
    missing = [k for k in need if not found.get(k)]
    if missing:
        sample = sorted({l for e in all_entities[:200] for l in _labels(e)})[:30]
        raise SimScaleRunError(
            f"could not resolve entities {missing} in the imported geometry. Labels seen: {sample}. "
            "Run scripts/inspect_geometry.py, then set simscale.entity_resolution: manual."
        )
    return found


# --------------------------------------------------------------------------- #
# backend
# --------------------------------------------------------------------------- #
class SimScaleBackend:
    name = "simscale"

    def __init__(self, cfg: Config, client: Optional[SimScaleClient] = None):
        self.cfg = cfg
        self.client = client or make_client()
        self._state_path = Path(cfg.project.workdir) / "simscale_state.json"
        self._project_id: Optional[str] = None

    # ---- project -------------------------------------------------------- #
    def project_id(self) -> str:
        if self._project_id:
            return self._project_id
        if self.cfg.simscale.project_id:
            self._project_id = self.cfg.simscale.project_id
        elif self._state_path.is_file():
            self._project_id = json.loads(self._state_path.read_text())["project_id"]
        else:
            from simscale_sdk import Project

            proj = self.client.project_api.create_project(
                Project(name=self.cfg.simscale.project_name, description="BWB Bayesian optimisation",
                        measurement_system="SI")
            )
            self._project_id = proj.project_id
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({"project_id": self._project_id}))
        return self._project_id

    # ---- public API ----------------------------------------------------- #
    def run(self, cfg: Config, pf: Planform, flow: FlowConditions, domain_stl: Path,
            work_dir: Path, label: str) -> CFDResult:
        try:
            return self._run(cfg, pf, flow, Path(domain_stl), Path(work_dir), label)
        except SimScaleAuthError:
            raise
        except Exception as exc:                          # noqa: BLE001 - a failed design must not kill the loop
            status = getattr(exc, "status", None)
            if status in (401, 403):
                raise SimScaleAuthError(f"SimScale rejected the credentials (HTTP {status})") from exc
            log.exception("SimScale run failed for %s", label)
            return CFDResult(ok=False, message=f"{type(exc).__name__}: {exc}")

    # ---- workflow ------------------------------------------------------- #
    def _run(self, cfg: Config, pf: Planform, flow: FlowConditions, domain_stl: Path,
             work_dir: Path, label: str) -> CFDResult:
        import simscale_sdk as sdk

        c, s = self.client, cfg.simscale
        pid = self.project_id()
        ids: Dict[str, Any] = {"project_id": pid}
        plan = plan_mesh(cfg, pf, flow)
        ref = reference_values(cfg, pf, flow)

        # 1. geometry
        storage = c.storage_api.create_storage()
        c.upload(storage.url, domain_stl.read_bytes())
        imp = c.geometry_import_api.import_geometry(
            pid,
            sdk.GeometryImportRequest(
                name=f"{label}_domain",
                location=sdk.GeometryImportRequestLocation(storage.storage_id),
                format=s.geometry_format,
                input_unit=s.geometry_unit,
                options=sdk.GeometryImportRequestOptions(facet_split=s.facet_split, sewing=s.sewing, improve=s.improve),
            ),
        )
        imp = _wait(lambda: c.geometry_import_api.get_geometry_import(pid, imp.geometry_import_id),
                    "geometry import", s.geometry_timeout_s, 10.0)
        if imp.status != "FINISHED":
            raise SimScaleRunError(f"geometry import ended with status {imp.status}")
        geometry_id = imp.geometry_id
        ids["geometry_id"] = geometry_id

        # 2. entities + refinement regions
        entities = resolve_entities(c, pid, geometry_id, cfg)
        prim_ids = {k: c.simulation_api.create_geometry_primitive(pid, box).geometry_primitive_id
                    for k, box in specs.refinement_boxes(plan).items()}

        # 3. simulation spec + material
        model = specs.build_flow_model(cfg, pf, flow, ref, entities)
        spec = sdk.SimulationSpec(name=f"{label}_{cfg.flow.analysis_type}", geometry_id=geometry_id, model=model)
        sim_id = c.simulation_api.create_simulation(pid, spec).simulation_id
        ids["simulation_id"] = sim_id
        self._assign_material(pid, sim_id, entities["fluid"])

        # 4. mesh
        mesh_model = specs.build_mesh_model(cfg, plan, entities, prim_ids)
        mesh_op = c.mesh_operation_api.create_mesh_operation(
            pid, sdk.MeshOperation(name=f"{label}_mesh", geometry_id=geometry_id, model=mesh_model))
        mesh_id_op = mesh_op.mesh_operation_id
        ids["mesh_operation_id"] = mesh_id_op
        _check_entries(c.mesh_operation_api.check_mesh_operation_setup(pid, mesh_id_op, simulation_id=sim_id), "mesh")
        mesh_runtime = _estimate(lambda: c.mesh_operation_api.estimate_mesh_operation(pid, mesh_id_op),
                                 "mesh", s.max_compute_units, s.mesh_timeout_s)
        c.mesh_operation_api.start_mesh_operation(pid, mesh_id_op, simulation_id=sim_id)
        mesh_op = _wait(lambda: c.mesh_operation_api.get_mesh_operation(pid, mesh_id_op),
                        "meshing", min(mesh_runtime, s.mesh_timeout_s), s.poll_interval_s)
        if mesh_op.status != "FINISHED":
            raise SimScaleRunError(f"meshing ended with status {mesh_op.status}")

        # 5. attach mesh and run
        spec = c.simulation_api.get_simulation(pid, sim_id)
        spec.mesh_id = mesh_op.mesh_id
        c.simulation_api.update_simulation(pid, sim_id, spec)
        _check_entries(c.simulation_api.check_simulation_setup(pid, sim_id), "simulation")
        run_timeout = _estimate(lambda: c.simulation_api.estimate_simulation_setup(pid, sim_id),
                                "simulation", s.max_compute_units, s.sim_timeout_s)
        run = c.simulation_run_api.create_simulation_run(pid, sim_id, sdk.SimulationRun(name=f"{label}_run"))
        run_id = run.run_id
        ids["run_id"] = run_id
        c.simulation_run_api.start_simulation_run(pid, sim_id, run_id)
        run = _wait(lambda: c.simulation_run_api.get_simulation_run(pid, sim_id, run_id),
                    "simulation", min(run_timeout, s.sim_timeout_s), s.poll_interval_s)
        if run.status != "FINISHED":
            return CFDResult(ok=False, message=f"simulation run ended with status {run.status}", extra=ids)

        # 6. results
        force_csv = self._download(pid, sim_id, run_id, "FORCE_PLOT", work_dir)
        moment_csv = self._download(pid, sim_id, run_id, "MOMENT_PLOT", work_dir)
        cl, cd, cm, info = coefficients_from_tables(
            [parse_csv(t) for t in force_csv], [parse_csv(t) for t in moment_csv], flow, ref,
            s.averaging_window, s.results_column_map, s.force_scale)
        info.update(ids)
        info["mesh_plan"] = plan.to_dict()
        ok = all(map(lambda v: v == v and abs(v) < 1e3, (cl, cd, cm))) and cd > 0.0
        return CFDResult(ok=ok, cl=cl, cd=cd, cm=cm,
                         message="" if ok else "non-physical coefficients (check force_scale / column map)", extra=info)

    # ---- pieces --------------------------------------------------------- #
    def _assign_material(self, pid: str, sim_id: str, fluid_entities: List[str]) -> None:
        import simscale_sdk as sdk

        c = self.client
        groups = c.materials_api.get_material_groups().embedded
        default = next((g for g in groups if g.group_type == sdk.MaterialGroupType.SIMSCALE_DEFAULT), None)
        if default is None:
            raise SimScaleRunError("default SimScale material group not found")
        mats = c.materials_api.get_materials(material_group_id=default.material_group_id).embedded
        mat = next((m for m in mats if m.name == self.cfg.simscale.material_name), None)
        if mat is None:
            raise SimScaleRunError(f"material '{self.cfg.simscale.material_name}' not found in the default library")
        data = c.materials_api.get_material_data(material_group_id=default.material_group_id, material_id=mat.id)
        request = sdk.MaterialUpdateRequest(
            operations=[sdk.MaterialUpdateOperation(
                path="/materials/fluids",
                material_data=data,
                reference=sdk.MaterialUpdateOperationReference(
                    material_group_id=default.material_group_id, material_id=mat.id),
            )]
        )
        c.simulation_api.update_simulation_materials(pid, sim_id, request)
        spec = c.simulation_api.get_simulation(pid, sim_id)
        spec.model.materials.fluids[0].topological_reference = sdk.TopologicalReference(entities=fluid_entities)
        c.simulation_api.update_simulation(pid, sim_id, spec)

    def _download(self, pid: str, sim_id: str, run_id: str, category: str, work_dir: Path) -> List[str]:
        c = self.client
        res = c.simulation_run_api.get_simulation_run_results(pid, sim_id, run_id, page=1, limit=100, category=category)
        items = list(res.embedded or [])
        if not items:
            raise SimScaleRunError(f"no results of category {category} - is the force/moment result control present?")
        out_dir = Path(work_dir) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        texts: List[str] = []
        for i, item in enumerate(items):
            text = c.download(item.download.url).decode("utf-8", errors="replace")
            (out_dir / f"{category.lower()}_{i}.csv").write_text(text)
            texts.append(text)
        return texts
