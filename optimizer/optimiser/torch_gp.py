"""Exact Gaussian process in PyTorch (float64, CPU or CUDA).

* Matern-5/2 ARD kernel, learnable log-lengthscales / output-scale / noise
* hyper-parameters by MAP (marginal likelihood + weak log-normal priors) with L-BFGS and random restarts
* differentiable predictive mean/variance, so acquisition functions can be optimised with autograd
* closed-form leave-one-out residuals (used by the review stage as a surrogate quality check)
* `condition_on` adds fantasy observations (Kriging believer) for batch proposals
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

SQRT5 = math.sqrt(5.0)
LOG_2PI = math.log(2.0 * math.pi)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def matern52(x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor, outputscale: torch.Tensor) -> torch.Tensor:
    a = x1 / lengthscale
    b = x2 / lengthscale
    d2 = (a.pow(2).sum(-1, keepdim=True) - 2.0 * a @ b.T + b.pow(2).sum(-1).unsqueeze(0)).clamp_min(0.0)
    r = torch.sqrt(d2 + 1e-12)
    return outputscale * (1.0 + SQRT5 * r + (5.0 / 3.0) * d2) * torch.exp(-SQRT5 * r)


class TorchGP:
    """GP on inputs in [0, 1]^d and (internally standardised) scalar targets."""

    def __init__(self, dim: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        self.dim = dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.log_ls = torch.full((dim,), math.log(0.4), device=self.device, dtype=dtype)
        self.log_sf2 = torch.zeros((), device=self.device, dtype=dtype)
        self.log_sn2 = torch.tensor(math.log(1e-3), device=self.device, dtype=dtype)
        self.noise_floor = 1e-6
        self.X: Optional[torch.Tensor] = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._L: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._y_std_train: Optional[torch.Tensor] = None
        self.fit_loss = float("nan")

    # ---- hyper-parameters ------------------------------------------------ #
    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_ls.exp()

    @property
    def outputscale(self) -> torch.Tensor:
        return self.log_sf2.exp()

    @property
    def noise(self) -> torch.Tensor:
        return self.log_sn2.exp() + self.noise_floor

    def _prior_penalty(self, log_ls: torch.Tensor, log_sf2: torch.Tensor, log_sn2: torch.Tensor) -> torch.Tensor:
        return (0.5 * ((log_ls - math.log(0.4)) / 1.0).pow(2).sum()
                + 0.5 * ((log_sf2 - 0.0) / 1.5) ** 2
                + 0.5 * ((log_sn2 - math.log(1e-3)) / 3.0) ** 2)

    def _nll(self, X: torch.Tensor, y: torch.Tensor, log_ls, log_sf2, log_sn2) -> torch.Tensor:
        n = X.shape[0]
        K = matern52(X, X, log_ls.exp(), log_sf2.exp())
        K = K + (log_sn2.exp() + self.noise_floor) * torch.eye(n, device=X.device, dtype=X.dtype)
        L, info = torch.linalg.cholesky_ex(K)
        if int(info) > 0:
            return (log_ls.sum() + log_sf2 + log_sn2) * 0.0 + 1e10
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        nll = 0.5 * (y @ alpha) + torch.log(torch.diagonal(L)).sum() + 0.5 * n * LOG_2PI
        return nll + self._prior_penalty(log_ls, log_sf2, log_sn2)

    # ---- fitting --------------------------------------------------------- #
    def fit(self, X: torch.Tensor, y: torch.Tensor, n_restarts: int = 4, max_iter: int = 60, seed: int = 0) -> "TorchGP":
        X = X.to(self.device, self.dtype)
        y = y.to(self.device, self.dtype).reshape(-1)
        if X.shape[0] != y.shape[0] or X.shape[0] < 2:
            raise ValueError("need at least 2 observations")
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        ys = (y - self.y_mean) / self.y_std
        gen = torch.Generator(device="cpu").manual_seed(seed)

        starts = [(self.log_ls.clone(), self.log_sf2.clone(), self.log_sn2.clone())]
        for _ in range(max(0, n_restarts)):
            starts.append((
                (math.log(0.4) + 0.7 * torch.randn(self.dim, generator=gen, dtype=self.dtype)).to(self.device),
                (0.5 * torch.randn((), generator=gen, dtype=self.dtype)).to(self.device),
                (math.log(1e-3) + 1.5 * torch.randn((), generator=gen, dtype=self.dtype)).to(self.device),
            ))
        best = None
        for ls0, sf0, sn0 in starts:
            ls = ls0.clone().requires_grad_(True)
            sf = sf0.clone().requires_grad_(True)
            sn = sn0.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([ls, sf, sn], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = self._nll(X, ys, ls.clamp(-5.0, 3.0), sf.clamp(-6.0, 6.0), sn.clamp(-14.0, 1.0))
                loss.backward()
                return loss

            try:
                opt.step(closure)
                with torch.no_grad():
                    val = float(self._nll(X, ys, ls.clamp(-5.0, 3.0), sf.clamp(-6.0, 6.0), sn.clamp(-14.0, 1.0)))
            except Exception:                              # noqa: BLE001 - a failed restart is simply skipped
                continue
            if math.isfinite(val) and (best is None or val < best[0]):
                best = (val, ls.detach().clamp(-5.0, 3.0), sf.detach().clamp(-6.0, 6.0), sn.detach().clamp(-14.0, 1.0))
        if best is None:
            raise RuntimeError("GP hyper-parameter fit failed for every restart")
        self.fit_loss = best[0]
        self.log_ls, self.log_sf2, self.log_sn2 = best[1], best[2], best[3]
        self._set_data(X, ys)
        return self

    def _set_data(self, X: torch.Tensor, ys: torch.Tensor) -> None:
        n = X.shape[0]
        K = matern52(X, X, self.lengthscale, self.outputscale) + self.noise * torch.eye(n, device=X.device, dtype=X.dtype)
        jitter = 0.0
        for _ in range(6):
            L, info = torch.linalg.cholesky_ex(K + jitter * torch.eye(n, device=X.device, dtype=X.dtype))
            if int(info) == 0:
                break
            jitter = max(1e-10, jitter * 10.0)
        else:
            raise RuntimeError("Cholesky failed even with jitter")
        self.X, self._L, self._y_std_train = X, L, ys
        self._alpha = torch.cholesky_solve(ys.unsqueeze(-1), L).squeeze(-1)

    # ---- prediction (standardised scale; differentiable) ----------------- #
    def predict_std(self, Xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and std of the latent function on the STANDARDISED target scale."""
        assert self.X is not None and self._L is not None and self._alpha is not None
        Xs = Xs.to(self.device, self.dtype)
        ks = matern52(Xs, self.X, self.lengthscale, self.outputscale)         # (m, n)
        mean = ks @ self._alpha
        v = torch.linalg.solve_triangular(self._L, ks.T, upper=False)         # (n, m)
        var = (self.outputscale - v.pow(2).sum(0)).clamp_min(1e-12)
        return mean, var.sqrt()

    def predict(self, Xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and std on the ORIGINAL target scale."""
        m, s = self.predict_std(Xs)
        return m * self.y_std + self.y_mean, s * self.y_std

    @property
    def best_std(self) -> float:
        assert self._y_std_train is not None
        return float(self._y_std_train.min())

    # ---- batch proposals ------------------------------------------------- #
    def condition_on(self, x_new: torch.Tensor, y_new_std: torch.Tensor) -> "TorchGP":
        """Copy of this GP (same hyper-parameters) with extra observations appended (Kriging believer)."""
        assert self.X is not None and self._y_std_train is not None
        g = TorchGP(self.dim, self.device, self.dtype)
        g.log_ls, g.log_sf2, g.log_sn2 = self.log_ls.clone(), self.log_sf2.clone(), self.log_sn2.clone()
        g.y_mean, g.y_std = self.y_mean, self.y_std
        X = torch.cat([self.X, x_new.reshape(-1, self.dim).to(self.device, self.dtype)])
        ys = torch.cat([self._y_std_train, y_new_std.reshape(-1).to(self.device, self.dtype)])
        g._set_data(X, ys)
        return g

    # ---- diagnostics ------------------------------------------------------ #
    def loo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form leave-one-out predictive mean and std (ORIGINAL scale)."""
        assert self._L is not None and self._alpha is not None and self._y_std_train is not None
        Kinv = torch.cholesky_inverse(self._L)
        d = torch.diagonal(Kinv)
        mu = self._y_std_train - self._alpha / d
        sd = (1.0 / d).sqrt()
        return mu * self.y_std + self.y_mean, sd * self.y_std

    def relevance(self) -> torch.Tensor:
        """ARD relevance: normalised inverse lengthscales (sums to 1)."""
        inv = 1.0 / self.lengthscale
        return inv / inv.sum()


# --------------------------------------------------------------------------- #
# acquisition functions (all return a quantity to MAXIMISE; minimisation problem)
# --------------------------------------------------------------------------- #
_LOG_SQRT_2PI = 0.5 * LOG_2PI


def _log_h(z: torch.Tensor) -> torch.Tensor:
    """log( phi(z) + z * Phi(z) ), stable for very negative z.

    Direct evaluation is accurate in float64 down to z ~ -20 (erfc keeps relative precision); beyond that the
    asymptotic series  phi(z)/z^2 * (1 - 3/z^2 + 15/z^4 - 105/z^6 + 945/z^8)  is used (error < 1e-9 there).
    """
    cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
    pdf = torch.exp(-0.5 * z * z - _LOG_SQRT_2PI)
    direct = torch.log((pdf + z * cdf).clamp_min(1e-300))
    zz = torch.where(z < -20.0, z, torch.full_like(z, -20.0))
    zi = 1.0 / zz ** 2
    series = torch.log1p(-3.0 * zi + 15.0 * zi ** 2 - 105.0 * zi ** 3 + 945.0 * zi ** 4)
    asym = -0.5 * zz * zz - _LOG_SQRT_2PI - 2.0 * torch.log(-zz) + series
    return torch.where(z < -20.0, asym, direct)


def log_expected_improvement(gp: TorchGP, x: torch.Tensor, xi: float = 0.01) -> torch.Tensor:
    mu, sd = gp.predict_std(x)
    z = (gp.best_std - mu - xi) / sd
    return _log_h(z) + torch.log(sd)


def lower_confidence_bound(gp: TorchGP, x: torch.Tensor, kappa: float = 2.0) -> torch.Tensor:
    mu, sd = gp.predict_std(x)
    return -(mu - kappa * sd)


def log_probability_of_improvement(gp: TorchGP, x: torch.Tensor, xi: float = 0.01) -> torch.Tensor:
    mu, sd = gp.predict_std(x)
    z = (gp.best_std - mu - xi) / sd
    return torch.special.log_ndtr(z)
