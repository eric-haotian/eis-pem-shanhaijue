# learning AI at www.haotianblog.com
"""The estimator ensemble (ShanHaiJue) — CPU backend.

Every arm minimises the SAME box-constrained weighted least-squares objective

    s_hat(lambda) = argmin_s  sum_k |Z_model(w_k; s) - Z_k|^2 / |Z_k|^2
                              + lambda^2 || W_p (s - s_0) ||^2

and differs ONLY in the prior scale lambda and in initialization. The physical
mechanism is the bias-variance geometry of a sloppy model: at large lambda the prior
stabilises the flat directions of the cost valley at the price of bias on the
data-determined directions; as lambda -> 0 that bias vanishes but the flat directions
destabilise. The regimes therefore fail on DIFFERENT coordinates, which is what the
router in `router.py` exploits.

Backend: scipy trust-region-reflective least squares (CPU, multiprocessing-friendly).
The GPU distribution ships an identical interface backed by a batched JAX
Levenberg-Marquardt solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import panel as P

# Prior-strength ladder. The three-member subset {1.0, multistart, data-only} is the
# original configuration; the full eight-member ladder is the strongest tested system.
LADDER_8 = (3.0, 1.0, 0.6, 0.3, 0.1, 0.03)
LADDER_6 = (3.0, 1.0, 0.3, 0.1)
LADDER_3 = (1.0,)

N_STARTS = 3            # starts per lambda arm (coordinate-wise median is the estimate)
JITTER = 0.15           # start dispersion for the lambda arms
WIDE_JITTER = 0.35      # start dispersion for the multistart arm
N_WIDE = 3              # extra wide starts for the multistart arm
MAX_NFEV = 400


@dataclass
class ArmResult:
    """One estimator arm's output for one cell."""
    name: str
    estimate_search: np.ndarray                 # coordinate-wise median, search coords
    starts_search: list = field(default_factory=list)
    cost: float = float("nan")


def _fit(z_obs, s0, prior_scale=1.0, mask=None, max_nfev=None):
    """Single bounded fit. `prior_scale` multiplies the base prior weight (lambda);
    `prior_scale=0` gives the data-only objective."""
    max_nfev = MAX_NFEV if max_nfev is None else max_nfev
    zmag = np.maximum(np.abs(z_obs), 1e-30)
    m = np.ones(P.NFREQ, bool) if mask is None else mask
    zo, zm = z_obs[m], zmag[m]
    w = prior_scale * P.PRIOR_W

    def residual(sv):
        z = P.MODEL.simulate(P.FSTACK, P.to_physical(sv))[m]
        r = (z - zo) / zm
        rows = [r.real, r.imag]
        if prior_scale > 0:
            rows.append(w * (sv - P.SPMEAN))
        return np.concatenate(rows)

    sol = least_squares(residual, s0, bounds=(P.SLO, P.SHI), method="trf",
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    return sol.x, sol.jac, float(sol.cost)


def _starts(seed, n, scale):
    rng = np.random.default_rng(seed)
    return [np.clip(P.SPMEAN + rng.normal(0, scale, P.NDIM), P.SLO, P.SHI)
            for _ in range(n)]


def fit_lambda_arm(z_obs, lam: float, seed: int) -> ArmResult:
    """One prior-strength arm: N_STARTS fits, coordinate-wise median in search space."""
    starts = [P.SPMEAN.copy()] + _starts(seed + 11, N_STARTS - 1, JITTER)
    sols, costs = [], []
    for s0 in starts:
        s, _, c = _fit(z_obs, s0, prior_scale=lam)
        sols.append(s)
        costs.append(c)
    est = np.median(np.vstack(sols), axis=0)
    return ArmResult(f"lam{lam:g}".replace(".", ""), est, sols, float(min(costs)))


def fit_multistart_arm(z_obs, seed: int, reference: np.ndarray = None) -> ArmResult:
    """Wide multistart at lambda=1; the lowest-cost solution wins."""
    cands = []
    if reference is not None:
        _, _, c0 = _fit(z_obs, reference, prior_scale=1.0, max_nfev=1)
        cands.append((c0, reference))
    for s0 in _starts(seed + 11, N_WIDE, WIDE_JITTER):
        s, _, c = _fit(z_obs, s0, prior_scale=1.0)
        cands.append((c, s))
    c, s = min(cands, key=lambda t: t[0])
    return ArmResult("M1", s, [s], float(c))


def fit_dataonly_arm(z_obs, warm_start: np.ndarray) -> ArmResult:
    """lambda -> 0 refinement warm-started from the MAP solution."""
    s, _, c = _fit(z_obs, warm_start, prior_scale=0.0)
    return ArmResult("M4", s, [s], float(c))


def fit_ensemble(z_obs, seed: int, ladder=LADDER_8):
    """Fit the full ensemble for one cell.

    Returns (arms, aux) where `aux` carries the shared per-cell quantities the feature
    extractor needs: the Jacobian at the lambda=1 first start and the held-out-frequency
    refit.
    """
    arms = {}
    jac_ref, s_ref = None, None
    for lam in ladder:
        arm = fit_lambda_arm(z_obs, lam, seed)
        arms[arm.name] = arm
        if abs(lam - 1.0) < 1e-12:
            s_ref = arm.starts_search[0]
            _, jac_ref, _ = _fit(z_obs, P.SPMEAN.copy(), prior_scale=1.0, max_nfev=1)
    if s_ref is None:                                  # ladder without lambda=1
        s_ref, jac_ref, _ = _fit(z_obs, P.SPMEAN.copy(), prior_scale=1.0)

    m1 = fit_multistart_arm(z_obs, seed, reference=s_ref)
    arms[m1.name] = m1
    m4 = fit_dataonly_arm(z_obs, s_ref)
    arms[m4.name] = m4

    s_holdout, _, _ = _fit(z_obs, P.SPMEAN.copy(), prior_scale=1.0, mask=P.KEEP)
    aux = dict(jacobian=jac_ref, reference_start=s_ref, holdout_estimate=s_holdout)
    return arms, aux
