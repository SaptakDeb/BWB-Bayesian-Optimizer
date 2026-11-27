"""Pure builders for the SimScale API objects (no network access in here).

Every SDK class name below was checked against the simscale-python-sdk source.
SDK model names change between releases: if a newer SDK renames something, this
is the only file that needs touching.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..atmosphere import FlowConditions
from ..config import Config
from ..planform import Planform
from .base import ReferenceValues
from .mesh_requirements import MeshPlan

ENTITY_KEYS = ("wing", "inlet", "outlet", "farfield", "symmetry", "fluid")


def _length(sdk: Any, value: float):
    return sdk.DimensionalLength(value=float(value), unit="m")


def _vec_length(sdk: Any, v) -> Any:
    return sdk.DimensionalVectorLength(value=sdk.DecimalVector(x=float(v[0]), y=float(v[1]), z=float(v[2])), unit="m")


def _const(sdk: Any, v: float):
    return sdk.ConstantFunction(value=float(v))


def refinement_boxes(plan: MeshPlan) -> Dict[str, Any]:
    """Geometry primitives (Cartesian boxes) for the near-body and wake refinement regions."""
    import simscale_sdk as sdk

    return {
        "near_body": sdk.CartesianBox(
            name="Near-body refinement",
            min=_vec_length(sdk, plan.near_body_box["min"]),
            max=_vec_length(sdk, plan.near_body_box["max"]),
        ),
        "wake": sdk.CartesianBox(
            name="Wake refinement",
            min=_vec_length(sdk, plan.wake_box["min"]),
            max=_vec_length(sdk, plan.wake_box["max"]),
        ),
    }


def build_mesh_model(cfg: Config, plan: MeshPlan, entities: Dict[str, List[str]], primitive_ids: Dict[str, str]):
    """SimScale hex-dominant (snappyHexMesh) model with sizing, refinements and wall layers."""
    import simscale_sdk as sdk

    m = cfg.mesh
    wing_ref = sdk.TopologicalReference(entities=entities["wing"])

    if m.sizing_mode == "manual":
        sizing = sdk.ManualMeshSizingHexDominantSnappy(
            minimum_edge_length=_length(sdk, plan.surface_min),
            maximum_edge_length=_length(sdk, plan.farfield_max),
        )
    else:
        sizing = sdk.AutomaticMeshSizingHexDominantSnappy(fineness=m.automatic_fineness)

    refinements: List[Any] = [
        sdk.SurfaceRefinementHexDominantSnappy(
            name="Wing surface",
            min_length=_length(sdk, plan.surface_min),
            max_length=_length(sdk, plan.surface_max),
            cell_zone=sdk.WithoutCellZone(),
            topological_reference=wing_ref,
        ),
        sdk.LayerAdditionHexDominantSnappy(
            name="Wing boundary layers",
            layers=int(plan.layers),
            expansion_ratio=float(plan.expansion_ratio),
            min_thickness=float(plan.min_layer_thickness),
            first_layer_thickness=float(plan.first_layer),
            topological_reference=wing_ref,
        ),
        sdk.RegionRefinementWithLength(
            name="Near-body region",
            refinement=sdk.InsideRegionRefinementWithLength(length=_length(sdk, plan.near_body_size)),
            geometry_primitive_uuids=[primitive_ids["near_body"]],
        ),
        sdk.RegionRefinementWithLength(
            name="Wake region",
            refinement=sdk.InsideRegionRefinementWithLength(length=_length(sdk, plan.wake_size)),
            geometry_primitive_uuids=[primitive_ids["wake"]],
        ),
    ]
    if m.use_feature_refinement:
        refinements.append(
            sdk.FeatureRefinementHexDominantSnappy(
                name="Wing edges",
                distance_refinement_lengths=[
                    sdk.RefinementLength(
                        distance=_length(sdk, plan.feature_distance),
                        length=_length(sdk, plan.feature_length),
                    )
                ],
                topological_reference=wing_ref,
            )
        )
    return sdk.HexDominantSnappy(
        meshing_mode="INTERNAL",
        sizing=sizing,
        physics_based_meshing=False,
        num_of_processors=int(m.processors),
        max_meshing_run_time=sdk.DimensionalTime(value=float(m.max_run_time_s), unit="s"),
        refinements=refinements,
    )


def _numerics(sdk: Any):
    return sdk.FluidNumerics(
        relaxation_factor=sdk.RelaxationFactor(),
        pressure_reference_value=sdk.DimensionalPressure(value=0, unit="Pa"),
        residual_controls=sdk.ResidualControls(
            velocity=sdk.Tolerance(),
            pressure=sdk.Tolerance(),
            turbulent_kinetic_energy=sdk.Tolerance(),
            omega_dissipation_rate=sdk.Tolerance(),
        ),
        solvers=sdk.FluidSolvers(),
        schemes=sdk.Schemes(
            time_differentiation=sdk.TimeDifferentiationSchemes(),
            gradient=sdk.GradientSchemes(),
            divergence=sdk.DivergenceSchemes(),
            laplacian=sdk.LaplacianSchemes(),
            interpolation=sdk.InterpolationSchemes(),
            surface_normal_gradient=sdk.SurfaceNormalGradientSchemes(),
        ),
    )


def build_flow_model(cfg: Config, pf: Planform, flow: FlowConditions, ref: ReferenceValues,
                     entities: Dict[str, List[str]]):
    """Steady RANS, regular k-omega SST (no transition model).

    analysis_type selects `Incompressible` (gauge pressure, no energy equation) or
    `Compressible` (absolute pressure, energy equation, fixed inlet temperature).
    """
    import simscale_sdk as sdk

    compressible = cfg.flow.analysis_type == "compressible"
    vx, vy, vz = (flow.speed * c for c in flow.flow_direction())

    def ref_to(key: str):
        return sdk.TopologicalReference(entities=entities[key])

    inlet_kwargs: Dict[str, Any] = dict(
        name="Velocity inlet",
        velocity=sdk.FixedValueVBC(
            value=sdk.DimensionalVectorFunctionSpeed(
                value=sdk.ComponentVectorFunction(x=_const(sdk, vx), y=_const(sdk, vy), z=_const(sdk, vz))
            )
        ),
        turbulence_intensity=sdk.TurbulenceIntensityTIBC(
            value=sdk.DimensionalFunctionDimensionless(value=_const(sdk, cfg.flow.turbulence_intensity), unit="")
        ),
        dissipation_type=sdk.AutomaticOmegaDissipation(),
        topological_reference=ref_to("inlet"),
    )
    if compressible:
        inlet_kwargs["temperature"] = sdk.FixedValueTBC(
            value=sdk.DimensionalFunctionTemperature(value=_const(sdk, flow.temperature), unit="K")
        )
        outlet = sdk.PressureOutletBC(
            name="Pressure outlet",
            pressure=sdk.FixedValuePBC(
                value=sdk.DimensionalFunctionPressure(value=_const(sdk, flow.pressure), unit="Pa")
            ),
            topological_reference=ref_to("outlet"),
        )
    else:
        outlet = sdk.PressureOutletBC(
            name="Pressure outlet",
            gauge_pressure=sdk.FixedValuePBC(
                value=sdk.DimensionalFunctionPressure(value=_const(sdk, 0.0), unit="Pa")
            ),
            topological_reference=ref_to("outlet"),
        )

    wing_wall = dict(
        name="Wing wall",
        velocity=sdk.NoSlipVBC(
            turbulence_wall="FULL_RESOLUTION" if cfg.mesh.wall_treatment == "resolved" else "WALL_FUNCTION"
        ),
        topological_reference=ref_to("wing"),
    )
    far_wall = dict(name="Far-field slip", velocity=sdk.SlipVBC(), topological_reference=ref_to("farfield"))
    if compressible:
        wing_wall["temperature"] = sdk.AdiabaticTBC()
        far_wall["temperature"] = sdk.AdiabaticTBC()

    bcs: List[Any] = [sdk.VelocityInletBC(**inlet_kwargs), outlet, sdk.WallBC(**wing_wall), sdk.WallBC(**far_wall)]
    if cfg.geometry.symmetry_mode == "half":
        if not entities.get("symmetry"):
            raise ValueError("half model needs a 'symmetry' entity set")
        bcs.append(sdk.SymmetryBC(name="Symmetry plane", topological_reference=ref_to("symmetry")))

    sim_control = sdk.FluidSimulationControl(
        end_time=sdk.DimensionalTime(value=float(cfg.simscale.iterations), unit="s"),
        delta_t=sdk.DimensionalTime(value=1.0, unit="s"),
        write_control=sdk.TimeStepWriteControl(write_interval=int(cfg.simscale.write_interval)),
        max_run_time=sdk.DimensionalTime(value=float(cfg.simscale.sim_timeout_s), unit="s"),
        decompose_algorithm=sdk.ScotchDecomposeAlgorithm(),
    )
    result_control = sdk.FluidResultControls(
        forces_moments=[
            sdk.ForcesMomentsResultControl(
                name="Wing forces and moments",
                center_of_rotation=_vec_length(sdk, ref.moment_ref),
                write_control=sdk.TimeStepWriteControl(write_interval=1),
                topological_reference=ref_to("wing"),
            )
        ]
    )
    common = dict(
        turbulence_model="KOMEGASST",
        model=sdk.FluidModel(),
        initial_conditions=sdk.FluidInitialConditions(),
        advanced_concepts=sdk.AdvancedConcepts(),
        numerics=_numerics(sdk),
        boundary_conditions=bcs,
        simulation_control=sim_control,
        result_control=result_control,
    )
    if compressible:
        return sdk.Compressible(materials=sdk.CompressibleFluidMaterials(), **common)
    return sdk.Incompressible(materials=sdk.IncompressibleFluidMaterials(), **common)
