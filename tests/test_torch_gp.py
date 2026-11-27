import pytest

torch = pytest.importorskip("torch")

from bwbopt.optimiser.torch_gp import TorchGP, _log_h, log_expected_improvement, lower_confidence_bound


def _data(n=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g, dtype=torch.float64)
    f = lambda x: torch.sin(6 * x[:, 0]) + 3 * (x[:, 1] - 0.5) ** 2
    return X, f(X), f


def test_gp_interpolates_smooth_function():
    X, y, f = _data()
    gp = TorchGP(2).fit(X, y, n_restarts=2)
    Xt = torch.rand(200, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    m, s = gp.predict(Xt)
    assert float((m - f(Xt)).pow(2).mean().sqrt()) < 0.1
    assert float(s.min()) > 0


def test_loo_matches_brute_force():
    X, y, _ = _data(12)
    gp = TorchGP(2).fit(X, y, n_restarts=1)
    mu, _ = gp.loo()
    i = 3
    keep = [k for k in range(12) if k != i]
    g2 = TorchGP(2)
    g2.log_ls, g2.log_sf2, g2.log_sn2 = gp.log_ls, gp.log_sf2, gp.log_sn2
    g2.y_mean, g2.y_std = gp.y_mean, gp.y_std
    g2._set_data(X[keep], ((y - gp.y_mean) / gp.y_std)[keep])
    brute, _ = g2.predict(X[i:i + 1])
    assert float(mu[i]) == pytest.approx(float(brute[0]), abs=1e-6)


def test_condition_on_shrinks_uncertainty():
    X, y, _ = _data()
    gp = TorchGP(2).fit(X, y, n_restarts=1)
    x = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    m, s0 = gp.predict_std(x)
    s1 = gp.condition_on(x, m).predict_std(x)[1]
    assert float(s1) < float(s0)


def test_log_h_is_continuous_and_finite():
    z = torch.tensor([-30.0, -20.0001, -19.9999, -10.0, 0.0, 3.0], dtype=torch.float64)
    h = _log_h(z)
    assert torch.isfinite(h).all()
    assert float(h[2] - h[1]) == pytest.approx(20.0 * 2e-4, rel=0.05)   # just the local slope: no seam
    assert (h[1:] > h[:-1]).all()                      # monotone increasing in z


def test_acquisitions_are_differentiable():
    X, y, _ = _data()
    gp = TorchGP(2).fit(X, y, n_restarts=1)
    u = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
    for acq in (log_expected_improvement, lower_confidence_bound):
        u.grad = None
        acq(gp, u).sum().backward()
        assert torch.isfinite(u.grad).all() and float(u.grad.abs().sum()) > 0
