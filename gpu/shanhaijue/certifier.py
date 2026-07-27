# learning AI at www.haotianblog.com
"""Risk-controlled certification (ShanHaiJian).

Three strictly separated stages, so that a failure in one cannot silently corrupt
another:

  ranker      -> the routed claim probability from `router.py`
  calibrator  -> a global Platt map fitted on cross-fitted out-of-fold routed scores,
                 so the winner's-curse optimism of the arg-max is absorbed into the
                 calibration it sees
  controller  -> the claim threshold is the SMALLEST t whose false-claim burden
                 respects a preregistered budget B at a 95% upper confidence bound:

                     t* = min { t : UCB95[ F(t) ] <= B },
                     F(t) = (1/R) sum_r sum_j 1{p_jr >= t} 1{e_jr > tol}

                 The upper bound is a percentile bootstrap over GENERATING TRUTHS (the
                 cluster unit), never over rows. Because F(t) is nonincreasing in t, the
                 first admissible threshold is the least conservative one.

Abstention is the default state: a coordinate below t* is simply not claimed.

Scope: the control is empirical, not a finite-sample guarantee. A dedicated simulation
of this construction at ~30 clusters measures one-sided coverage at 0.88-0.93 against
the nominal 0.95, so budgets certified at this panel size should be read with a 2-7
percentage-point coverage haircut.
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattMap:
    """A one-dimensional logistic map, stored as its two coefficients.

    Both calibration stages (per-arm in the router, global here) are single-feature
    logistic regressions, i.e. p = sigma(a x + b). Freezing the fitted estimator object
    itself would tie a frozen system to the exact library version that produced it; the
    two numbers are the whole model, so they are what gets stored. The evaluation is
    identical to the estimator's `predict_proba` to machine precision.
    """

    __slots__ = ("a", "b")

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def from_estimator(cls, est):
        return cls(np.ravel(est.coef_)[0], np.ravel(est.intercept_)[0])

    def probability(self, x) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(x, float) + self.b)))

    def predict_proba(self, X) -> np.ndarray:
        p = self.probability(np.asarray(X, float).ravel())
        return np.column_stack([1.0 - p, p])

    def __repr__(self):
        return f"PlattMap(a={self.a:.6g}, b={self.b:.6g})"


def as_platt(obj):
    """Normalise a calibrator to `PlattMap`, accepting an already-converted one."""
    return obj if isinstance(obj, PlattMap) else PlattMap.from_estimator(obj)


class Certifier:
    """Platt calibration of routed probabilities + cluster-UCB budget thresholds."""

    def __init__(self, tolerance: float = 0.10):
        self.tolerance = tolerance
        self.platt = None
        self.thresholds = {}          # budget -> t*

    # -- calibration ---------------------------------------------------------------
    def fit_calibrator(self, routed_records):
        """Fit the Platt map on CROSS-FITTED routed probabilities (see router)."""
        from sklearn.linear_model import LogisticRegression
        p = np.array([r["p_route"] for r in routed_records])
        y = np.array([r["y"] for r in routed_records])
        lr = LogisticRegression(max_iter=1000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        self.platt = PlattMap.from_estimator(lr)
        return self

    def calibrated(self, routed_records) -> np.ndarray:
        p = np.array([r["p_route"] for r in routed_records])
        return as_platt(self.platt).probability(_logit(p))

    # -- risk control --------------------------------------------------------------
    def fit_threshold(self, routed_records, budget: float, group_of,
                      n_boot: int = 2000, n_grid: int = 300, seed: int = 0):
        """Smallest threshold whose cluster-bootstrap UCB95 on false claims/cell <= B."""
        phat = self.calibrated(routed_records)
        cells = sorted({r["cell"] for r in routed_records})
        groups = sorted({group_of(c) for c in cells})
        rng = np.random.default_rng(seed)
        grid = np.quantile(phat, np.linspace(0.5, 0.999, n_grid))

        for t in grid:
            false_per_cell = {c: 0 for c in cells}
            for r, p in zip(routed_records, phat):
                if p >= t and r["y"] == 0:
                    false_per_cell[r["cell"]] += 1
            by_group = {}
            for c, v in false_per_cell.items():
                by_group.setdefault(group_of(c), []).append(v)
            vals = np.array([np.mean(by_group[g]) for g in groups])
            boots = [rng.choice(vals, size=vals.size, replace=True).mean()
                     for _ in range(n_boot)]
            if np.percentile(boots, 95) <= budget:
                self.thresholds[float(budget)] = float(t)
                return float(t)
        self.thresholds[float(budget)] = 1.1          # nothing admissible: abstain fully
        return 1.1

    # -- deployment ----------------------------------------------------------------
    def certify(self, routed_records, budget: float) -> list:
        """Apply a frozen threshold. Returns the certified claims with full records."""
        t = self.thresholds[float(budget)]
        phat = self.calibrated(routed_records)
        out = []
        for r, p in zip(routed_records, phat):
            rec = dict(r)
            rec["p_certified"] = float(p)
            rec["threshold"] = t
            rec["certified"] = bool(p >= t)
            out.append(rec)
        return out


def certification_report(certified_records) -> dict:
    """Correct/false claims per cell and realized precision (requires offline labels)."""
    correct, false = {}, {}
    for r in certified_records:
        c = r["cell"]
        correct.setdefault(c, 0)
        false.setdefault(c, 0)
        if r["certified"]:
            if r.get("y", 0) == 1:
                correct[c] += 1
            else:
                false[c] += 1
    cm = float(np.mean(list(correct.values()))) if correct else 0.0
    fm = float(np.mean(list(false.values()))) if false else 0.0
    return dict(correct_per_cell=cm, false_per_cell=fm,
                realized_precision=cm / max(cm + fm, 1e-9),
                n_cells=len(correct))


def cluster_bootstrap_ci(values_per_group, n_boot: int = 4000, seed: int = 0):
    """Percentile bootstrap CI over generating truths (the cluster unit)."""
    v = np.asarray(values_per_group, float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
