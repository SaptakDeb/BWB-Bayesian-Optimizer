import json

import pytest

pytest.importorskip("torch")

from bwbopt.optimiser import BayesianOptimiser, Database
from bwbopt.parametrisation import DesignSpace
from bwbopt.stages import optimise_stage, review_stage, run_stage, setup_stage
from conftest import ROOT


def args(tmp_path, *extra):
    return ["--backend", "mock", "--config", str(ROOT / "config" / "config.yaml"),
            "--set", f"project.workdir={tmp_path / 'run'}", "--set", "optimisation.n_initial=4",
            "--set", "optimisation.n_iterations=3", *extra]


def test_four_stages_end_to_end(tmp_path):
    assert setup_stage.main(args(tmp_path)) == 0
    setup = tmp_path / "run" / "setup"
    assert (setup / "setup_report.json").is_file() and (setup / "domain.stl").is_file()
    assert run_stage.main(args(tmp_path, "--parallel", "2")) == 0
    rows = Database(tmp_path / "run" / "evaluations.jsonl").load()
    assert len(rows) == 4 and rows[0].label == "baseline"
    assert run_stage.main(args(tmp_path)) == 0                       # resumable: nothing new
    assert len(Database(tmp_path / "run" / "evaluations.jsonl").load()) == 4
    assert optimise_stage.main(args(tmp_path, "--batch", "2", "--parallel", "2")) == 0
    best = json.loads((tmp_path / "run" / "best.json").read_text())
    assert best["n_evaluations"] == 7
    assert review_stage.main(args(tmp_path, "--compare-objectives")) == 0
    rev = tmp_path / "run" / "review"
    for f in ("summary.md", "evaluations.csv", "best_design.yaml", "review.png"):
        assert (rev / f).is_file()


def test_setup_fails_cleanly_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("SIMSCALE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)                                       # no .env here
    argv = ["--backend", "simscale", "--no-geometry", "--config", str(ROOT / "config" / "config.yaml"),
            "--set", "geometry.backend=analytic", "--set", f"project.workdir={tmp_path / 'run'}"]
    assert setup_stage.main(argv) == 1


def test_batch_proposals_are_distinct_and_feasible(cfg):
    cfg.optimisation.n_initial = 5
    sp = DesignSpace(cfg)
    bo = BayesianOptimiser(sp, cfg.optimisation)
    bo.plan_initial()
    for x in bo.ask_batch(5):
        bo.tell(x, sum(x.values()) * 1e-2)
    batch = bo.ask_batch(3)
    pts = [tuple(sp.to_unit(x)) for x in batch]
    assert len(set(pts)) == 3
    assert all(sp.lower[i] - 1e-9 <= batch[0][n] <= sp.upper[i] + 1e-9 for i, n in enumerate(sp.names))


def test_failed_runs_are_imputed_not_poisonous(cfg):
    cfg.optimisation.n_initial = 4
    sp = DesignSpace(cfg)
    bo = BayesianOptimiser(sp, cfg.optimisation)
    bo.plan_initial()
    for i, x in enumerate(bo.ask_batch(4)):
        bo.tell(x, 10.0 if i == 1 else -0.3 - 0.01 * i, failed=(i == 1))
    y = bo.training_targets()
    assert y.max() < 1.0 and y[1] > max(y[0], y[2], y[3])
    assert bo.best()[1] == min(v for v, f in zip(bo.y, bo.failed) if not f)
