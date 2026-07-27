# learning AI at www.haotianblog.com
"""Per-coordinate routing across the estimator ensemble (ShanHaiJue claim layer).

For coordinate j under arm b, a gradient-boosted ranker f maps the ground-truth-free
diagnostic vector x_jb (plus coordinate and arm identity) to a score s_jb. Per-arm
logistic calibration g_b renders scores comparable across arms, and the routed estimate
takes coordinate j from

    b*_j = argmax_b  g_b( s_jb ).

The ranker is an additive model of shallow trees fitted to the functional gradient of
the logistic loss; with depth <= 3 every tree is at most eight axis-aligned threshold
rules on named physical diagnostics, so the routing decision is rule-readable and every
routed coordinate carries a complete decision record.

Truth labels are used only to fit the ranker offline; deployment consumes none.
"""
from __future__ import annotations

import numpy as np

from .certifier import PlattMap, as_platt
from .features import FEATURES, design_matrix

# manuscript hyperparameters
HGB_KWARGS = dict(max_depth=3, max_iter=200, learning_rate=0.08,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=0)
CATEGORICAL = [len(FEATURES), len(FEATURES) + 1]      # coordinate identity, arm identity


class Router:
    """Gradient-boosted ranker + per-arm calibrators + arg-max routing."""

    def __init__(self, arm_names):
        self.arm_names = sorted(arm_names)
        self.arm_ids = {a: i for i, a in enumerate(self.arm_names)}
        self.ranker = None
        self.calibrators = {}

    # -- training ------------------------------------------------------------------
    def fit(self, rows, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        rows = [r for r in rows if r["arm"] in self.arm_ids]
        y = np.array([r["y"] for r in rows])
        kw = dict(HGB_KWARGS)
        kw["random_state"] = seed
        self.ranker = HistGradientBoostingClassifier(categorical_features=CATEGORICAL, **kw)
        X = design_matrix(rows, self.arm_ids)
        self.ranker.fit(X, y)
        raw = self.ranker.predict_proba(X)[:, 1]
        for a in self.arm_names:
            idx = [i for i, r in enumerate(rows) if r["arm"] == a]
            lr = LogisticRegression(max_iter=1000)
            lr.fit(raw[idx].reshape(-1, 1), y[idx])
            self.calibrators[a] = PlattMap.from_estimator(lr)
        return self

    # -- inference -----------------------------------------------------------------
    def score(self, rows) -> np.ndarray:
        """Calibrated per-(arm, coordinate) claim probabilities."""
        rows = [r for r in rows if r["arm"] in self.arm_ids]
        raw = self.ranker.predict_proba(design_matrix(rows, self.arm_ids))[:, 1]
        return np.array([as_platt(self.calibrators[r["arm"]]).probability(s)
                         for r, s in zip(rows, raw)])

    def route(self, rows) -> list:
        """Route each coordinate to its highest-scoring arm.

        Returns one record per coordinate with the winning arm, its calibrated
        probability, the estimate, and (when present) the offline label.
        """
        rows = [r for r in rows if r["arm"] in self.arm_ids]
        probs = self.score(rows)
        best = {}
        for r, p in zip(rows, probs):
            key = (r.get("cell"), r["coord_idx"])
            if key not in best or p > best[key]["p_route"]:
                rec = dict(r)
                rec["p_route"] = float(p)
                best[key] = rec
        return list(best.values())


def routed_recovery(routed_records) -> dict:
    """Correct routed coordinates per cell (requires offline labels)."""
    per = {}
    for r in routed_records:
        per.setdefault(r.get("cell"), 0)
        per[r.get("cell")] += int(r.get("y", 0))
    return per


def cross_fitted_route(rows, arm_names, group_of, n_folds: int = 5, seed: int = 0):
    """Out-of-fold routing with folds grouped by generating truth.

    `group_of(cell) -> group id`; every cell sharing a group (e.g. the noise replicates
    of one truth) stays on the same side of the train/test split, so no coordinate is
    ever routed by a model that saw its own truth.
    """
    groups = sorted({group_of(r.get("cell")) for r in rows})
    fold_of = {g: i % n_folds for i, g in enumerate(groups)}
    out = []
    for fold in range(n_folds):
        train = [r for r in rows if fold_of[group_of(r.get("cell"))] != fold]
        test = [r for r in rows if fold_of[group_of(r.get("cell"))] == fold]
        if not train or not test:
            continue
        out.extend(Router(arm_names).fit(train, seed=seed).route(test))
    return out
