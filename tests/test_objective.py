import pytest

from bwbopt.cfd.base import CFDResult
from bwbopt.config import ObjectiveCfg
from bwbopt.objective import Baseline, score

BASE = Baseline(cl=0.45, cd=0.02, cm=0.0)


def res(cl, cd, cm=0.0, ok=True):
    return CFDResult(ok=ok, cl=cl, cd=cd, cm=cm)


def test_failed_result_gets_failure_value():
    cfg = ObjectiveCfg()
    s = score(cfg, res(float("nan"), 0.02, ok=False), BASE)
    assert not s.valid and s.total == cfg.penalties.failure_value


def test_higher_ld_is_better_in_every_mode():
    for mode in ("max_ld", "multi", "max_ld_cm"):
        cfg = ObjectiveCfg(mode=mode)
        assert score(cfg, res(0.45, 0.015), BASE).total < score(cfg, res(0.45, 0.025), BASE).total


def test_min_cd_ignores_ld():
    cfg = ObjectiveCfg(mode="min_cd")
    assert score(cfg, res(0.45, 0.015), BASE).total < score(cfg, res(0.45, 0.02), BASE).total


def test_cl_penalty_pulls_towards_target():
    cfg = ObjectiveCfg(mode="max_ld", cl_target=0.45)
    on = score(cfg, res(0.45, 0.02), BASE).total
    off = score(cfg, res(0.20, 0.0089), BASE).total          # same L/D, wrong CL
    assert on < off


def test_cm_bounds_only_bite_outside():
    cfg = ObjectiveCfg(mode="max_ld_cm", cm_bounds=[-0.04, 0.04])
    inside = score(cfg, res(0.45, 0.02, 0.03), BASE)
    outside = score(cfg, res(0.45, 0.02, 0.10), BASE)
    assert inside.terms["cm_pen"] == 0.0 and outside.terms["cm_pen"] > 0.0


def test_baseline_scores_consistently():
    cfg = ObjectiveCfg(mode="multi")
    s = score(cfg, res(BASE.cl, BASE.cd, BASE.cm), BASE)
    w = cfg.weights
    assert s.total == pytest.approx(-w.ld + w.cd)
