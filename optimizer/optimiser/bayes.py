"""PyTorch Bayesian optimisation (minimisation) on a box-bounded design space.

* surrogate  : exact GP in PyTorch (`torch_gp.TorchGP`), refitted from scratch every iteration
* acquisition: log-EI | LCB (ucb) | log-PI, maximised with autograd (multi-start Adam in a sigmoid
               re-parametrisation of the unit cube, seeded from the best of a large quasi-random pool)
* batches    : `ask_batch(q)` uses Kriging believer fantasies, so q CFD runs can be launched in parallel
* constraints: cheap geometric feasibility is enforced on every proposal (checked lazily, best first)
* failures   : failed CFD runs are imputed just above the worst valid cost so they steer the search
               away without wrecking the GP's target scale
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import qmc

from ..config import OptCfg
from ..parametrisation import DesignSpace
from .torch_gp import (TorchGP, log_expected_improvement, log_probability_of_improvement, lower_confidence_bound,
                       resolve_device)

Feasible = Callable[[Dict[str, float]], bool]


def initial_design(space: DesignSpace, n: int, seed: int, feasible: Optional[Feasible] = None,
                   baseline_point: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
    """Scrambled-Sobol design of experiments (feasible points only); optionally starts with the baseline."""
    feasible = feasible or (lambda x: True)
    pts: List[Dict[str, float]] = []
    if baseline_point is not None:
        pts.append(space.clip(baseline_point))
    sob = qmc.Sobol(d=space.dim, scramble=True, seed=seed)
    guard = 0
    while len(pts) < n and guard < 400:
        guard += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = sob.random(max(8, n))
        for u in batch:
            if len(pts) >= n:
                break
            x = space.from_unit(u)
            if feasible(x):
                pts.append(x)
    if len(pts) < n:
        raise RuntimeError("could not generate enough feasible initial designs - relax constraints/bounds")
    return pts


class BayesianOptimiser:
    def __init__(self, space: DesignSpace, opt: OptCfg, feasible: Optional[Feasible] = None):
        self.space = space
        self.opt = opt
        self.feasible: Feasible = feasible or (lambda x: True)
        self.device = resolve_device(opt.device)
        self.dtype = torch.float64
        self.gen = torch.Generator(device="cpu").manual_seed(opt.seed)
        self.rng = np.random.default_rng(opt.seed)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.failed: List[bool] = []
        self._initial: List[Dict[str, float]] = []
        self.gp: Optional[TorchGP] = None
        self._fit_counter = 0

    # ---- data ----------------------------------------------------------- #
    def tell(self, x: Dict[str, float], y: float, failed: bool = False) -> None:
        self.X.append(np.clip(self.space.to_unit(x), 0.0, 1.0))
        self.y.append(float(y))
        self.failed.append(bool(failed))

    @property
    def n_told(self) -> int:
        return len(self.y)

    def best(self) -> Tuple[Dict[str, float], float]:
        ok = [i for i, f in enumerate(self.failed) if not f] or list(range(len(self.y)))
        i = min(ok, key=lambda k: self.y[k])
        return self.space.from_unit(self.X[i]), self.y[i]

    def training_targets(self) -> np.ndarray:
        """Costs with failed runs imputed just above the worst valid one."""
        y = np.asarray(self.y, dtype=float)
        bad = np.asarray(self.failed, dtype=bool)
        if bad.any() and (~bad).any():
            good = y[~bad]
            y = y.copy()
            y[bad] = good.max() + 0.25 * max(good.max() - good.min(), 1e-6)
        return y

    # ---- initial design ------------------------------------------------- #
    def plan_initial(self, baseline_point: Optional[Dict[str, float]] = None) -> None:
        self._initial = initial_design(self.space, self.opt.n_initial, self.opt.seed, self.feasible, baseline_point)

    # ---- surrogate -------------------------------------------------------- #
    def fit(self) -> TorchGP:
        if self.n_told < 2:
            raise RuntimeError("need at least 2 observations before the GP can be fitted")
        self._fit_counter += 1
        gp = TorchGP(self.space.dim, self.device, self.dtype)
        X = torch.as_tensor(np.vstack(self.X), dtype=self.dtype)
        y = torch.as_tensor(self.training_targets(), dtype=self.dtype)
        gp.fit(X, y, n_restarts=self.opt.n_restarts, seed=self.opt.seed + self._fit_counter)
        self.gp = gp
        return gp

    def _acq(self, gp: TorchGP, u: torch.Tensor) -> torch.Tensor:
        kind = self.opt.acquisition
        if kind == "ucb":
            return lower_confidence_bound(gp, u, self.opt.kappa)
        if kind == "pi":
            return log_probability_of_improvement(gp, u, self.opt.xi)
        return log_expected_improvement(gp, u, self.opt.xi)

    # ---- proposals -------------------------------------------------------- #
    def _novel(self, u: np.ndarray, known: np.ndarray) -> bool:
        return known.size == 0 or float(np.min(np.linalg.norm(known - u, axis=1))) > 1e-3

    def _maximise_acquisition(self, gp: TorchGP, known: np.ndarray) -> np.ndarray:
        d = self.space.dim
        sob = qmc.Sobol(d=d, scramble=True, seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = sob.random(max(256, int(self.opt.n_candidates)))
        good = np.argsort(self.training_targets())[:5]
        local = np.vstack([np.clip(self.X[i] + 0.06 * self.rng.standard_normal((100, d)), 0.0, 1.0) for i in good])
        cand = np.vstack([raw, local])
        with torch.no_grad():
            a = self._acq(gp, torch.as_tensor(cand, dtype=self.dtype)).cpu().numpy()
        order = np.argsort(-a)

        starts: List[np.ndarray] = []
        for i in order:                                     # lazy feasibility: best candidates first
            u = cand[i]
            if self._novel(u, known) and self.feasible(self.space.from_unit(u)):
                starts.append(u)
                if len(starts) >= self.opt.acq_starts:
                    break
        if not starts:
            raise RuntimeError("no feasible, novel candidate found - constraints may be too tight")

        z = torch.logit(torch.as_tensor(np.clip(np.vstack(starts), 1e-3, 1 - 1e-3), dtype=self.dtype, device=self.device))
        z.requires_grad_(True)
        optim = torch.optim.Adam([z], lr=0.1)
        for _ in range(self.opt.acq_steps):
            optim.zero_grad()
            loss = -self._acq(gp, torch.sigmoid(z)).sum()
            loss.backward()
            optim.step()
        with torch.no_grad():
            polished = torch.sigmoid(z)
            a_pol = self._acq(gp, polished).cpu().numpy()
            a_raw = self._acq(gp, torch.as_tensor(np.vstack(starts), dtype=self.dtype)).cpu().numpy()
        pol = polished.cpu().numpy()
        best_u, best_a = starts[int(np.argmax(a_raw))], float(a_raw.max())
        for u, av in zip(pol, a_pol):
            if (av > best_a and self._novel(u, known) and self.feasible(self.space.from_unit(u))
                    and np.all(np.isfinite(u))):
                best_u, best_a = u, float(av)
        return np.clip(best_u, 0.0, 1.0)

    def ask_batch(self, q: int = 1) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        # initial design points not yet told
        start = self.n_told
        while len(out) < q and start + len(out) < len(self._initial):
            out.append(self._initial[start + len(out)])
        if len(out) == q:
            return out
        gp = self.fit()
        known = np.vstack(self.X) if self.X else np.empty((0, self.space.dim))
        batch_u: List[np.ndarray] = []
        while len(out) < q:
            u = self._maximise_acquisition(gp, known)
            out.append(self.space.from_unit(u))
            batch_u.append(u)
            known = np.vstack([known, u[None, :]])
            if len(out) < q:                                 # Kriging believer: pretend the mean was observed
                with torch.no_grad():
                    m, _ = gp.predict_std(torch.as_tensor(u[None, :], dtype=self.dtype))
                gp = gp.condition_on(torch.as_tensor(u[None, :], dtype=self.dtype), m)
        return out

    def ask(self) -> Dict[str, float]:
        return self.ask_batch(1)[0]

    # ---- diagnostics ------------------------------------------------------ #
    def predict(self, x: Dict[str, float]) -> Tuple[float, float]:
        gp = self.gp or self.fit()
        with torch.no_grad():
            m, s = gp.predict(torch.as_tensor(self.space.to_unit(x)[None, :], dtype=self.dtype))
        return float(m[0]), float(s[0])
