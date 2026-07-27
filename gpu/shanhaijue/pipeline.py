# learning AI at www.haotianblog.com
"""End-to-end orchestration: panel -> ensemble -> features -> router -> certifier.

Two workflows are supported and they must not be mixed:

  DEVELOPMENT  build_panel + train_and_freeze
               Everything is fitted, cross-fitted, and selected on the development
               panel. Numbers read off this panel are development quantities.

  VALIDATION   build_panel(seed bases reserved AT FREEZE TIME) + apply_frozen
               A frozen system is applied unchanged to a never-before-generated panel
               and the panel is analysed EXACTLY ONCE. No re-training, no re-thresholding.

The discipline that makes the reported numbers meaningful is procedural, not
algorithmic: freeze first (artifacts hashed, validation seeds reserved), generate the
validation panel second, analyse once, never pool panels.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import certifier as C
from . import panel as P
from . import router as R

# `arms` (the fitting backend) and `joblib` are imported lazily so that the routing and
# certification stages remain usable in environments without the fitting dependencies
# (e.g. an analysis host with scikit-learn but no CUDA-enabled JAX).


def _arms():
    from . import arms
    return arms


def _features():
    from . import features
    return features


def _joblib():
    import joblib
    return joblib


# ----------------------------------------------------------------------------------
# panel construction
# ----------------------------------------------------------------------------------
def fit_one_cell(task):
    """Worker for one cell. Module-level so it survives multiprocessing pickling.

    task = (truth_index, noise_index, draw_seed_base, noise_seed_base, ladder, basis)
    """
    ti, ni, draw_seed_base, noise_seed_base, ladder, basis = task
    A, F = _arms(), _features()
    truth = P.draw_truth(ti, draw_seed_base)
    seed = noise_seed_base + ti * 3 + ni
    z = P.simulate_observation(truth, seed)
    fitted, aux = A.fit_ensemble(z, seed, ladder=tuple(ladder))
    return F.extract(fitted, aux, z, basis=basis, truth=truth, cell_id=(ti, ni))


def build_panel(n_truths: int, n_noise: int, draw_seed_base: int, noise_seed_base: int,
                ladder=None, basis: str = "phys", progress=True, pool=None):
    """Fit the ensemble on a synthetic panel and return feature rows.

    Cell identity is the pair (truth index, noise index); the generating truth is the
    cluster unit for every downstream confidence interval.
    """
    if ladder is None:
        ladder = _arms().LADDER_8
    tasks = [(ti, ni, draw_seed_base, noise_seed_base, tuple(ladder), basis)
             for ti in range(n_truths) for ni in range(n_noise)]
    t0 = time.time()
    rows = []
    stream = map(fit_one_cell, tasks) if pool is None else pool.imap_unordered(fit_one_cell, tasks)
    for k, part in enumerate(stream):
        rows.extend(part)
        if progress:
            print(f"  [{k + 1}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    return rows


def group_of_cell(cell):
    """Cluster unit: the generating truth index."""
    return cell[0]


# ----------------------------------------------------------------------------------
# development: train, cross-fit, freeze
# ----------------------------------------------------------------------------------
def evaluate_routing(rows, arm_names, n_folds: int = 5, seed: int = 0):
    """Out-of-fold routed recovery, reported as a truth-cluster mean with a CI."""
    routed = R.cross_fitted_route(rows, arm_names, group_of_cell, n_folds, seed)
    per_cell = R.routed_recovery(routed)
    by_group = {}
    for cell, v in per_cell.items():
        by_group.setdefault(group_of_cell(cell), []).append(v)
    means = np.array([np.mean(by_group[g]) for g in sorted(by_group)])
    lo, hi = C.cluster_bootstrap_ci(means)
    return dict(routed_recovery=float(means.mean()), ci=[lo, hi],
                n_cells=len(per_cell), n_clusters=len(by_group)), routed


def train_and_freeze(rows, arm_names, budgets, out_dir, val_draw_seed, val_noise_seed,
                     n_folds: int = 5, seed: int = 0, note: str = ""):
    """Fit the deployable system on the FULL development panel and freeze it.

    The calibrator and thresholds are fitted on CROSS-FITTED routed scores (so they
    never see in-sample optimism), while the shipped router is trained on all
    development data. Validation seed bases are recorded in the manifest here, BEFORE
    any validation truth exists.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = R.cross_fitted_route(rows, arm_names, group_of_cell, n_folds, seed)
    cert = C.Certifier().fit_calibrator(oof)
    for b in budgets:
        cert.fit_threshold(oof, b, group_of_cell, seed=seed)

    router = R.Router(arm_names).fit(rows, seed=seed)

    # the grid travels with the system: its calibrated probabilities and thresholds were
    # fitted at this information content and are only valid where it is reproduced
    grid = dict(conditions=[[float(t), float(s)] for t, s in P.CONDITIONS],
                frequencies=[float(f) for f in P.FREQ])

    artifact = out_dir / "frozen_system.joblib"
    _joblib().dump(dict(router=router, certifier=cert, arm_names=sorted(arm_names),
                     budgets=list(budgets), val_draw_seed=val_draw_seed,
                     val_noise_seed=val_noise_seed, grid=grid), artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = dict(
        artifact="frozen_system.joblib", artifact_sha256=digest,
        arm_names=sorted(arm_names), budgets=list(budgets),
        thresholds={str(k): v for k, v in cert.thresholds.items()},
        grid=dict(n_conditions=len(P.CONDITIONS), n_frequencies=int(P.FREQ.size),
                  conditions=grid["conditions"]),
        calibrator="platt_global_on_cross_fitted_routed_scores",
        validation_seed_bases=dict(draw=val_draw_seed, noise=val_noise_seed),
        n_development_cells=len({r["cell"] for r in rows}),
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"), note=note,
    )
    (out_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ----------------------------------------------------------------------------------
# validation / deployment: apply a frozen system
# ----------------------------------------------------------------------------------
# The ranker is a fitted HistGradientBoostingClassifier, and unlike the calibrators it
# cannot be reduced to a few numbers, so the artifact carries scikit-learn's own pickle of
# it and inherits that pickle's version sensitivity. Measured against the shipped system,
# frozen under 1.7.2:
#
#     1.5.2   loads, then predict_proba raises on a renamed check_array keyword
#     1.6.1   loads; scores bit-identical to 1.7.2
#     1.7.2   the freezing version
#     1.8.0   loads; scores bit-identical to 1.7.2
#     1.9.0   fails to unpickle: sklearn._loss was restructured
#
# requirements.txt pins that verified window. This message exists for the reader who
# installed something else anyway.
SKLEARN_VERIFIED = ">=1.6,<1.9"


def load_frozen(path):
    """Load a frozen system, normalising anything version-fragile inside it.

    Calibrators frozen as fitted library estimators are converted to their two-coefficient
    form, so a system frozen under one scikit-learn release stays loadable under another.
    The conversion is exact and touches nothing else in the artifact.

    The ranker cannot be normalised that way; see `SKLEARN_VERIFIED` above.
    """
    try:
        frozen = _joblib().load(path)
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        try:
            import sklearn
            installed = sklearn.__version__
        except ImportError:                                   # pragma: no cover
            installed = "not installed"
        raise RuntimeError(
            f"Could not unpickle the frozen system at {path}: {type(e).__name__}: {e}\n"
            f"The ranker inside it is a scikit-learn estimator, and scikit-learn "
            f"{installed} is installed. Systems shipped with this distribution load and "
            f"score identically on scikit-learn {SKLEARN_VERIFIED}; outside that window "
            f"the pickle format has moved. Either install a version in range\n"
            f"    pip install 'scikit-learn{SKLEARN_VERIFIED}'\n"
            f"or re-freeze the system on your own scikit-learn (README section 3)."
        ) from e
    from .certifier import as_platt
    router = frozen.get("router")
    if router is not None and getattr(router, "calibrators", None):
        router.calibrators = {a: as_platt(c) for a, c in router.calibrators.items()}
    cert = frozen.get("certifier")
    if cert is not None and getattr(cert, "platt", None) is not None:
        cert.platt = as_platt(cert.platt)
    return frozen


def frozen_grid(frozen):
    """The acquisition grid a frozen system was calibrated on.

    Systems frozen before the grid was recorded fall back to the module defaults, which
    is what they were in fact calibrated on.
    """
    grid = frozen.get("grid")
    if grid:
        return [(float(t), float(s)) for t, s in grid["conditions"]], \
            np.asarray(grid["frequencies"], float)
    return list(P.CONDITIONS), np.asarray(P.FREQ, float)


def apply_frozen(frozen, rows, budget: float):
    """Route and certify with a frozen system. No fitting of any kind happens here."""
    routed = frozen["router"].route(rows)
    return frozen["certifier"].certify(routed, budget), routed


def validation_report(frozen, rows, budgets=None) -> dict:
    """Single-pass validation summary: routed recovery plus per-budget certification."""
    routed = frozen["router"].route(rows)
    per_cell = R.routed_recovery(routed)
    by_group = {}
    for cell, v in per_cell.items():
        by_group.setdefault(group_of_cell(cell), []).append(v)
    means = np.array([np.mean(by_group[g]) for g in sorted(by_group)])
    lo, hi = C.cluster_bootstrap_ci(means)
    out = dict(routed_recovery=float(means.mean()), recovery_ci=[lo, hi], rungs={})
    for b in (budgets or frozen["budgets"]):
        certified = frozen["certifier"].certify(routed, b)
        out["rungs"][str(b)] = C.certification_report(certified)
    return out
