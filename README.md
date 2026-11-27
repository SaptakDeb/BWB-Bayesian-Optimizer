# BWB half-wing Bayesian optimisation (OpenVSP + SimScale)

Parametric blended-wing-body **half-wing** (y = span, mirrored about the xz plane) built in **OpenVSP**,
meshed with **SimScale hex-dominant** and solved with **regular k-omega SST** (steady RANS, no transition
model) through the **SimScale API**, optimised with **PyTorch Gaussian-process Bayesian optimisation**.
Fully standalone: nothing here depends on any other project.

```
config/config.yaml            ONE file: geometry, flow, domain, mesh, SimScale, objective, optimiser  (= the test case)
scripts/
  01_setup.py                 STAGE 1  environment + config checks, derived planform/mesh plan, builds baseline CAD
  02_run.py                   STAGE 2  runs the baseline + a whole DOE through geometry -> mesh -> CFD (resumable, parallel)
  03_review.py                STAGE 3  tables, best-vs-baseline, Pareto, PyTorch GP diagnostics, plots, best_design.yaml
  04_optimise.py              STAGE 4  PyTorch Bayesian optimisation (warm-started, batch/parallel, resumable)
  run_test_case.py            runs 1 -> 2 -> 4 -> 3 on the bundled test case (free mock CFD by default)
  tools/                      inspect_geometry.py, inspect_run.py (SimScale debugging helpers)
bwbopt/
  stages/                     the four stage implementations (scripts are thin wrappers)
  config.py  atmosphere.py  planform.py  parametrisation.py  constraints.py  objective.py  pipeline.py  cli.py
  geometry/                   vsp_backend (OpenVSP API) | analytic_backend (numpy) | enclosure (domain - wing, named patches)
  cfd/                        mesh_requirements | simscale_specs | simscale_client | simscale_backend | results | mock_backend
  optimiser/                  torch_gp (exact GP, autograd) | bayes (log-EI/LCB/log-PI, batches) | database (JSONL log)
tests/                        38 tests
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# if pip cannot find simscale-sdk:
pip install git+https://github.com/SimScaleGmbH/simscale-python-sdk.git
```
**OpenVSP's Python API** (only for `geometry.backend: openvsp`): install the OpenVSP Python bindings that match your
OpenVSP version (see the OpenVSP docs, "Python API"), and make sure `import openvsp` works in this venv.
No OpenVSP yet? Set `geometry.backend: analytic` to run everything else.

## SimScale credentials (never share the key)
```bash
cp .env.example .env      # then edit .env and paste YOUR key after SIMSCALE_API_KEY=
```
`.env` is git-ignored; the key is only read from the environment, is never printed or written to run
folders, and the client object hides it from logs/tracebacks. Don't paste it into chat or commit it.

## The four stages
```bash
python scripts/run_test_case.py                      # whole chain on the test case, FREE (mock CFD), ~1 min

python scripts/01_setup.py    --backend mock --check-api   # 1. verify env/credentials/config, build baseline STLs
python scripts/02_run.py      --backend mock --parallel 2  # 2. baseline + DOE through geometry -> mesh -> CFD
python scripts/04_optimise.py --backend mock --batch 2 --parallel 2   # 4. Bayesian optimisation (PyTorch)
python scripts/03_review.py   --compare-objectives         # 3. review (any time; no CFD is run)
```
Real CFD (costs SimScale compute units): replace `--backend mock` by `--backend simscale`
(default). Do `01_setup.py --check-api --create-project`, then `02_run.py --only baseline` ONCE and look at the result
before launching the DOE and the optimisation. `run_test_case.py --backend simscale --yes` chains everything.

* **Objective per run:** `--objective max_ld|min_cd|max_ld_cm|multi` (weights/targets in `objective:`). Re-running a
  stage with another objective re-scores the stored CFD results; nothing is recomputed.
* **Analysis per run:** `--analysis incompressible|compressible` (test case: incompressible, 100 m/s, Mach 0.29).
* **PyTorch optimiser:** exact GP (Matern-5/2 ARD, MAP fit with L-BFGS, float64) + log-EI / LCB / log-PI maximised with
  autograd (multi-start Adam), `--device auto|cpu|cuda`, `--batch q` Kriging-believer batches so q CFD runs can overlap
  (`--parallel`). Failed CFD runs are imputed above the worst valid cost so they steer the search without wrecking the fit.
* Any value: `--set flow.speed=80 --set mesh.processors=32`. Everything is logged to `runs/<name>/evaluations.jsonl`;
  stages are resumable (re-run the same command).

## Geometry config (config.yaml)
One half-wing, 5 sections / 6 airfoil stations. Per section: `span` (y-projected), `tip_chord`, `sweep_deg`
(at `sweep_location`, 0 = leading edge), `twist_deg` (of the OUTBOARD airfoil, nose-up +), `dihedral_deg`.
The root chord is `geometry.root_chord`; each section starts at the previous tip chord (continuity by construction).
`symmetry_mode: half` simulates the half-model with a symmetry plane at y = 0 (coefficients use the half area, so they
equal the full-aircraft values); `full` mirrors the wing.

## Mesh requirements (regular SST, SimScale hex-dominant)
Derived per design in `cfd/mesh_requirements.py` from flat-plate relations (Cf = 0.026 Re^-1/7, delta = 0.37 x Re^-1/5):
* `wall_treatment: resolved` -> y+ ~ 1 (first layer ~4 um for the test case), layer stack sized to cover the root-chord
  boundary layer (35 layers at ratio 1.25); `wall_function` -> y+ ~ 30, a handful of layers (far cheaper; use for exploration).
* Surface refinement 0.4 %-2 % MAC, edge/feature refinement, near-body and wake refinement boxes, far-field max edge 0.6 MAC.
* Warnings if layers are capped, edges are too small for the layers, or the rough cell estimate exceeds `mesh.max_cells`.
* A cost guard (`simscale.max_compute_units`) aborts a run when SimScale's own estimate is too high.

## What is verified and what is not
Verified here: config/validation, analytic planform, the wing + enclosure (watertight, volume = box - wing), mesh
plan, force-CSV -> coefficient round trip, objective modes, the BO loop (mock end-to-end and resume), and that all
SimScale models (mesh + incompressible + compressible) instantiate and serialise against the real
`simscale-python-sdk` source. **Not run here (needs your accounts/machine):** live OpenVSP calls and live SimScale calls.
Check these on your first run:
1. **OpenVSP**: the `twist` of `XSec_i` is assumed to act on the section's outboard airfoil; open
   `runs/.../geometry/*.vsp3` once and confirm twist/sweep look right. A wrong parameter name raises immediately.
2. **Named STL patches**: the enclosure STL has solids `inlet, outlet, farfield, symmetry, wing`. `entity_resolution: auto`
   looks them up by name. If SimScale names them differently, run `scripts/inspect_geometry.py`, then use `manual`.
3. **Force CSV columns**: detected heuristically. If a run completes but fails to parse, run `scripts/inspect_run.py`
   and set `simscale.results_column_map`. If forces look kinematic (tiny), set `simscale.force_scale` (e.g. to rho).
4. **Layer sizes** are passed to SimScale as absolute metres (my reading of the API); confirm in the first mesh's layer report.
5. SDK model names can change between releases: all of them live in `cfd/simscale_specs.py`.
