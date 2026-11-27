import numpy as np

from bwbopt.cfd import make_backend
from bwbopt.optimiser import BayesianOptimiser, Database
from bwbopt.parametrisation import DesignSpace
from bwbopt.pipeline import Evaluator, run_optimisation


def test_bo_beats_random_on_quadratic(cfg):
    cfg.optimisation.n_initial = 6
    sp = DesignSpace(cfg)
    target = sp.from_unit(np.full(sp.dim, 0.7))

    def f(x):
        return float(sum(((x[n] - target[n]) / (hi - lo)) ** 2 for n, lo, hi in zip(sp.names, sp.lower, sp.upper)))

    bo = BayesianOptimiser(sp, cfg.optimisation)
    bo.plan_initial()
    for _ in range(20):
        x = bo.ask()
        bo.tell(x, f(x))
    rng = np.random.default_rng(1)
    random_best = min(f(sp.from_unit(rng.random(sp.dim))) for _ in range(20))
    assert bo.best()[1] < random_best


def test_full_mock_run_and_resume(cfg):
    cfg.optimisation.n_initial = 4
    cfg.optimisation.n_iterations = 3
    sp = DesignSpace(cfg)
    ev = Evaluator(cfg, sp, make_backend("mock", cfg), cfg.project.workdir)
    out = run_optimisation(cfg, ev)
    assert out["n_evaluations"] == 7
    rows = Database(f"{cfg.project.workdir}/evaluations.jsonl").load()
    assert len(rows) == 7 and rows[0].label == "baseline"
    # resume with a different objective: nothing new is evaluated, results are rescored
    cfg.objective.mode = "min_cd"
    ev2 = Evaluator(cfg, sp, make_backend("mock", cfg), cfg.project.workdir)
    out2 = run_optimisation(cfg, ev2)
    assert out2["objective_mode"] == "min_cd"
    assert len(Database(f"{cfg.project.workdir}/evaluations.jsonl").load()) == len(rows)


def test_infeasible_designs_skip_cfd(cfg):
    sp = DesignSpace(cfg)
    calls = []

    class Spy:
        name = "spy"

        def run(self, *a, **k):
            calls.append(1)
            raise AssertionError("CFD must not be called for infeasible geometry")

    ev = Evaluator(cfg, sp, Spy(), cfg.project.workdir)
    rec = ev.evaluate({**sp.nominal(), "chord_tip": 0.05}, index=1)
    assert rec.status == "infeasible" and not calls
