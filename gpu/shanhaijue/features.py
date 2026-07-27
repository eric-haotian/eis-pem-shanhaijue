# learning AI at www.haotianblog.com
"""Ground-truth-free per-coordinate diagnostics.

These ten quantities are the ONLY evidence the router and the certifier ever see. None
of them requires knowledge of the true parameters, which is what makes the system
deployable on measured cells where no answer key exists.

    local_ci95          local curvature half-width from the exact Jacobian
    ens_std             dispersion of the coordinate across restarts
    warmstart_range     range of the coordinate across restarts
    nominal_disp        displacement from the nominal reference (decades)
    holdout_shift       shift when a quarter of the frequencies is withheld (decades)
    holdout_err_run     prediction error on the withheld frequencies (per cell)
    cross_arm_disagree  disagreement against the lambda=1 arm (decades)
    jac_colnorm         Jacobian column norm for the coordinate
    resid_autocorr_run  lag-1 residual autocorrelation (per cell)
    boundary_prox       distance to the nearest parameter bound (search coords)

Truth labels, when available (synthetic panels), are attached as `y` for offline
training and evaluation only.
"""
from __future__ import annotations

import math

import numpy as np

from . import panel as P

FEATURES = ["local_ci95", "ens_std", "warmstart_range", "nominal_disp", "holdout_shift",
            "holdout_err_run", "cross_arm_disagree", "jac_colnorm", "resid_autocorr_run",
            "boundary_prox"]
LOG_FEATURES = {"local_ci95", "jac_colnorm"}      # entered as log10 magnitudes
TOLERANCE = 0.10                                   # relative-error tolerance for `y`


def local_ci95(jac: np.ndarray) -> np.ndarray:
    """Wald half-widths from the Gauss-Newton information (prior rows included)."""
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj + 1e-12 * np.eye(jtj.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj)
    return 1.96 * np.sqrt(np.clip(np.diag(cov), 0, None))


def residual_autocorrelation(s_search: np.ndarray, z_obs: np.ndarray) -> float:
    """Mean lag-1 autocorrelation of the modulus-scaled residual, per condition block."""
    z = P.simulate_clean(P.to_physical(s_search))
    r = (z - z_obs) / np.maximum(np.abs(z_obs), 1e-30)
    blocks = np.concatenate([r.real, r.imag]).reshape(-1, P.FREQ.size)
    acs = []
    for b in blocks:
        b0 = b - b.mean()
        d = float(np.sum(b0 * b0))
        acs.append(float(np.sum(b0[:-1] * b0[1:]) / d) if d > 0 else 0.0)
    return float(np.mean(acs))


def holdout_error(s_holdout: np.ndarray, z_obs: np.ndarray) -> float:
    """Relative prediction error on the withheld frequencies."""
    zp = P.simulate_clean(P.to_physical(s_holdout))
    h = P.HOLDOUT
    return float(np.mean(np.abs((zp - z_obs)[h]) / np.maximum(np.abs(z_obs[h]), 1e-30)))


def _decades(a: float, b: float) -> float:
    return float(abs(math.log10(max(a, 1e-300)) - math.log10(max(b, 1e-300))))


def extract(arms: dict, aux: dict, z_obs: np.ndarray, basis: str = "phys",
            truth: np.ndarray = None, cell_id=None) -> list:
    """Build one feature row per (arm, coordinate) for a single cell.

    `arms` maps arm name -> ArmResult; `aux` is the auxiliary dict from
    `arms.fit_ensemble`. If `truth` is given (synthetic panel), the label `y` and the
    realized relative error are attached.
    """
    ci = local_ci95(aux["jacobian"])
    colnorm = np.linalg.norm(aux["jacobian"][:P.NDATA, :], axis=0)
    ac = residual_autocorrelation(aux["reference_start"], z_obs)
    ho_err = holdout_error(aux["holdout_estimate"], z_obs)

    nominal = P.coord_vector(basis, P.SNOM)
    ho_vec = P.coord_vector(basis, aux["holdout_estimate"])
    truth_vec = P.coord_vector(basis, P.to_search(truth)) if truth is not None else None

    reference_name = "lam1" if "lam1" in arms else sorted(arms)[0]
    reference_vec = P.coord_vector(basis, arms[reference_name].estimate_search)

    rows = []
    for name, arm in arms.items():
        est = P.coord_vector(basis, arm.estimate_search)
        if len(arm.starts_search) > 1:
            logs = np.log10(np.maximum(
                np.vstack([P.coord_vector(basis, s) for s in arm.starts_search]), 1e-300))
            ens = logs.std(axis=0)
            rng_ = logs.max(axis=0) - logs.min(axis=0)
        else:
            ens = np.full(P.NDIM, np.nan)
            rng_ = np.full(P.NDIM, np.nan)
        est_log = P.to_search(est)
        for j in range(P.NDIM):
            row = dict(
                cell=cell_id, basis=basis, arm=name, coord_idx=j,
                coord=P.COORD_NAME[basis][j], family=P.family_of(P.COORD_NAME[basis][j]),
                est_log10=float(math.log10(max(est[j], 1e-300))),
                local_ci95=float(ci[j]),
                ens_std=float(ens[j]) if np.isfinite(ens[j]) else None,
                warmstart_range=float(rng_[j]) if np.isfinite(rng_[j]) else None,
                nominal_disp=_decades(est[j], nominal[j]),
                holdout_shift=_decades(est[j], ho_vec[j]),
                holdout_err_run=ho_err,
                cross_arm_disagree=_decades(est[j], reference_vec[j]),
                jac_colnorm=float(colnorm[j]),
                resid_autocorr_run=ac,
                boundary_prox=float(min(abs(est_log[j] - P.SLO[j]),
                                        abs(P.SHI[j] - est_log[j])))
                if np.isfinite(est_log[j]) else None,
            )
            if truth_vec is not None:
                rel = abs(est[j] - truth_vec[j]) / max(abs(truth_vec[j]), 1e-300)
                row["rel_err"] = float(rel)
                row["y"] = int(rel <= TOLERANCE)
            rows.append(row)
    return rows


def design_matrix(rows, arm_ids: dict) -> np.ndarray:
    """Feature rows -> numeric matrix [features | coordinate identity | arm identity]."""
    X = np.zeros((len(rows), len(FEATURES) + 2))
    for i, r in enumerate(rows):
        for k, f in enumerate(FEATURES):
            v = r.get(f)
            v = 0.0 if v is None else float(v)
            X[i, k] = math.log10(max(v, 1e-12)) if f in LOG_FEATURES else v
        X[i, len(FEATURES)] = r["coord_idx"]
        X[i, len(FEATURES) + 1] = arm_ids[r["arm"]]
    return X
