# learning AI at www.haotianblog.com
"""Identifiability-aware model comparison — Generation 2.

Generation 1 charges every candidate the full ``2k`` AIC penalty even
when the gate has already frozen (or the priors have effectively pinned)
most of its parameters, so complex models are doubly penalized: once by
the gate and again by the information criterion.  Generation 2 scores
each candidate with the **effective number of parameters**

    k_eff = tr[ F_data (F_data + Sigma_p^{-1})^{-1} ]

returned by the tiered gate — the generalized degrees of freedom of the
MAP problem (each whitened data eigenvalue contributes
``lambda / (lambda + 1)``).  A prior-dominated parameter costs ~0 degrees
of freedom, a strictly identified one costs ~1, so complex models compete
on the complexity the data actually pays for.

The plain-``k`` AICc is always reported alongside for auditability, and
all residual diagnostics (Durbin-Watson, Ljung-Box) reuse the untouched
Generation-1 implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .model_selection import (
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_log_likelihood,
    durbin_watson,
    f_test_nested,
    ljung_box_p_value,
)
from .results import IdentificationResult

from .fitting import HierarchicalMAPFitter, SubsetModel
from .gating import ParameterTier, TieredIdentifiabilityGate, TieredSelection
from .priors import DRTPriorBuilder, PriorSet

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFitResultV2:
    """Fit + identifiability record for one Generation-2 candidate."""

    model_name: str
    circuit_string: str
    n_params_full: int
    n_params_free: int
    k_eff: float
    n_strict: int
    n_prior_informed: int
    n_prior_dominated: int
    n_fixed: int
    n_observations: int
    rss: float
    log_likelihood: float
    aic_eff: float
    aicc_eff: float
    bic_eff: float
    aicc_full: float
    durbin_watson: float
    ljung_box_p: float
    identification_result: IdentificationResult
    tiered_selection: TieredSelection
    prior_set: PriorSet

    @property
    def n_identifiable(self) -> int:
        return self.n_strict + self.n_prior_informed


@dataclass(frozen=True)
class ModelComparisonResultV2:
    """Ranking of Generation-2 candidates by effective-dof AICc.

    ``best_parsimonious`` applies the Burnham-Anderson rule: among
    candidates within ``parsimony_window`` AICc_eff units of the
    minimum (statistically indistinguishable support), the structurally
    simplest one is preferred.  ``best_by_aicc_eff`` remains the raw
    minimum for auditability.
    """

    candidates: tuple[ModelFitResultV2, ...]
    best_by_aicc_eff: str
    best_by_bic_eff: str
    best_parsimonious: str
    parsimony_window: float
    delta_aicc_eff: dict[str, float]
    akaike_weights: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.candidates:
            rows.append(
                {
                    "model": c.model_name,
                    "circuit": c.circuit_string,
                    "k_full": c.n_params_full,
                    "k_free": c.n_params_free,
                    "k_eff": c.k_eff,
                    "n_strict": c.n_strict,
                    "n_prior_informed": c.n_prior_informed,
                    "n_prior_dominated": c.n_prior_dominated,
                    "n_fixed": c.n_fixed,
                    "RSS": c.rss,
                    "AICc_eff": c.aicc_eff,
                    "AICc_full": c.aicc_full,
                    "BIC_eff": c.bic_eff,
                    "delta_AICc_eff": self.delta_aicc_eff[c.model_name],
                    "akaike_weight": self.akaike_weights[c.model_name],
                    "durbin_watson": c.durbin_watson,
                    "ljung_box_p": c.ljung_box_p,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "Generation-2 Model Comparison",
            "=" * 78,
            f"  Best by AICc_eff: {self.best_by_aicc_eff}",
            f"  Best by BIC_eff:  {self.best_by_bic_eff}",
            f"  Selected (parsimony window {self.parsimony_window:.1f}): "
            f"{self.best_parsimonious}",
            "",
            f"  {'Model':<14} {'k':>3} {'k_eff':>6} {'strict':>6} {'+prior':>6} "
            f"{'AICc_eff':>11} {'dAICc':>8} {'w':>6}",
            "  " + "-" * 68,
        ]
        for c in self.candidates:
            lines.append(
                f"  {c.model_name:<14} {c.n_params_full:3d} {c.k_eff:6.2f} "
                f"{c.n_strict:6d} {c.n_prior_informed:6d} "
                f"{c.aicc_eff:11.2f} {self.delta_aicc_eff[c.model_name]:8.2f} "
                f"{self.akaike_weights[c.model_name]:6.3f}"
            )
        return "\n".join(lines)


def compare_models_v2(
    dataset: EISDataset,
    candidates: Sequence[Any],
    drt_result: Any,
    noise_level: float,
    weights: FloatArray | None = None,
    gate: TieredIdentifiabilityGate | None = None,
    fitter: HierarchicalMAPFitter | None = None,
    prior_builder: DRTPriorBuilder | None = None,
    parsimony_window: float = 2.0,
    f_test_alpha: float = 0.05,
    screen_candidates: bool = True,
    max_complexity_gap: int = 4,
) -> ModelComparisonResultV2:
    """Fit, gate, and rank Generation-2 candidate models.

    Per candidate: build DRT priors → hierarchical MAP fit → tiered gate
    at the fitted point → refit with FIXED-tier parameters frozen →
    effective-dof information criteria.  With ``screen_candidates`` the
    pool is first pruned by :func:`screen_candidates` using the DRT
    evidence, skipping fits the diagnostics already rule out.
    """

    if not candidates:
        raise ValueError("At least one candidate model is required")
    if gate is None:
        gate = TieredIdentifiabilityGate(assumed_noise_level=noise_level)
    elif not np.isclose(gate.assumed_noise_level, noise_level, rtol=0.25):
        logger.warning(
            "gate.assumed_noise_level (%.4g) disagrees with noise_level (%.4g); "
            "the MAP fit and the gate will use inconsistent noise floors",
            gate.assumed_noise_level,
            noise_level,
        )
    fitter = fitter or HierarchicalMAPFitter()
    prior_builder = prior_builder or DRTPriorBuilder()

    pool = list(candidates)
    if screen_candidates:
        pool = list(
            _screen_candidate_pool(
                drt_result, pool, max_complexity_gap=max_complexity_gap
            )
        )

    n_obs = 2 * dataset.freq_hz.size
    results: list[ModelFitResultV2] = []

    for model in sorted(pool, key=lambda m: m.n_params):
        try:
            fit = _fit_and_gate_one(
                dataset, model, drt_result, noise_level, weights, gate, fitter,
                prior_builder, n_obs,
            )
        except Exception:
            logger.warning("candidate %s failed", model.name, exc_info=True)
            continue
        results.append(fit)

    if not results:
        raise RuntimeError("All candidate models failed to fit")

    results.sort(key=lambda r: r.aicc_eff)
    best_aicc = results[0].model_name
    best_bic = min(results, key=lambda r: r.bic_eff).model_name

    min_aicc = min(r.aicc_eff for r in results)
    delta = {r.model_name: float(r.aicc_eff - min_aicc) for r in results}
    exp_delta = {name: np.exp(-0.5 * d) for name, d in delta.items()}
    total = sum(exp_delta.values())
    weights_akaike = {
        name: float(v / total) if total > 0 else 0.0 for name, v in exp_delta.items()
    }

    best_parsimonious = _select_parsimonious(
        results, delta, n_obs, parsimony_window, f_test_alpha, drt_result
    )

    return ModelComparisonResultV2(
        candidates=tuple(results),
        best_by_aicc_eff=best_aicc,
        best_by_bic_eff=best_bic,
        best_parsimonious=best_parsimonious,
        parsimony_window=float(parsimony_window),
        delta_aicc_eff=delta,
        akaike_weights=weights_akaike,
    )


# ---------------------------------------------------------------------------
# Candidate screening (Improvement 3)
# ---------------------------------------------------------------------------

#: Relaxation-arc count per registry model (diffusion / inductive elements
#: are not arcs).  Unknown models fall back to counting indexed alphas.
_ARC_COUNTS = {
    "1RC": 1, "1RCPE": 1, "2RCPE": 2, "Randles_W": 1, "2RCPE_W": 2, "3RCPE": 3,
    "ZARC1": 1, "ZARC2": 2, "ZARC3": 3, "ZARC1_G": 1,
    "ZARC2_FLW": 2, "ZARC2_FSW": 2, "ZARC2_fFLW": 2, "ZARC3_FLW": 3,
    "TLMB": 1, "TLM_CT": 1, "ZARC_SEI_TLM": 2,
}

_DIFFUSION_PARAMS = frozenset({"Rd", "tau_d", "Aw", "Rg", "tau_g"})


def _model_arc_count(model: Any) -> int:
    count = _ARC_COUNTS.get(getattr(model, "name", ""))
    if count is not None:
        return count
    indices = [
        int(name[5:])
        for name in getattr(model, "parameter_names", ())
        if name.startswith("alpha") and name[5:].isdigit()
    ]
    return max(indices, default=1)


def _model_has_diffusion(model: Any) -> bool:
    return bool(_DIFFUSION_PARAMS.intersection(getattr(model, "parameter_names", ())))


def _diffusion_evidence(
    drt_result: Any,
    significance_fraction: float = 0.08,
    slow_peak_safety_fraction: float = 1.0 / 3.0,
) -> bool:
    """True when the DRT carries evidence of a diffusion-like slow process.

    Either an explicit low-frequency tail, or a *significant* peak near
    the slow edge of the measured band — a finite-length Warburg with a
    large τ_d images as a slow peak, not a tail, so the binary flag alone
    misses bounded diffusion.  (The DRT τ grid extends one decade past
    1/(2π f_min), hence ``τ_limit ≈ tau.max() / 10``.)
    """

    if drt_result is None:
        return True
    if bool(getattr(drt_result, "has_diffusion_tail", True)):
        return True
    peak_resistances = np.asarray(
        getattr(drt_result, "peak_resistances", ()), dtype=float
    )
    peak_taus = np.asarray(getattr(drt_result, "peak_taus", ()), dtype=float)
    if not peak_taus.size:
        return False
    total_r = float(getattr(drt_result, "total_resistance", lambda: 0.0)())
    denominator = max(total_r, float(np.sum(peak_resistances)), 1e-30)
    significant = peak_resistances >= significance_fraction * denominator
    tau_grid = np.asarray(getattr(drt_result, "tau", ()), dtype=float)
    tau_limit = float(tau_grid.max()) / 10.0 if tau_grid.size else np.inf
    return bool(
        np.any(
            peak_taus[significant[: peak_taus.size]]
            >= slow_peak_safety_fraction * tau_limit
        )
    )


def screen_candidates(
    drt_result: Any,
    all_models: Sequence[Any],
    max_complexity_gap: int = 4,
    significance_fraction: float = 0.08,
    slow_peak_safety_fraction: float = 1.0 / 3.0,
) -> tuple[Any, ...]:
    """Discard candidates the DRT evidence already rules out.

    Applied rules, each deliberately erring toward *keeping* models
    because DRT diagnostics are noisy in known ways:

    * **Arc cap** — models needing more than ``n_significant_peaks + 1``
      relaxation arcs are dropped.  Peaks are counted only when they
      carry at least ``significance_fraction`` of the total polarisation
      resistance: broad CPE arcs routinely split into shoulder peaks,
      and tiny artifacts must not inflate the allowed complexity.
    * **Diffusion screen** — diffusion-bearing models are dropped when
      there is no diffusion tail *and* no significant peak near the
      slow edge of the measured band.  The second condition matters: a
      finite-length Warburg with a large τ_d shows up as a slow *peak*,
      not a tail, so ``has_diffusion_tail`` alone hides true models.
    * **Inductance screen** — an L-bearing model is dropped for
      inductance-free data only when an otherwise identical L-free
      model remains in the pool.  In this library L is a nuisance rider
      on diffusion models (bounded near zero when the data show no
      inductive kick), so excluding those models outright would discard
      the diffusion physics along with the inductor.
    * **Complexity gap** — with ``k`` the parameter count of the
      simplest surviving model able to represent every significant
      process, models with more than ``k + max_complexity_gap``
      parameters are dropped.

    Falls back to the unscreened pool if fewer than two models survive.
    """

    models = list(all_models)
    if drt_result is None or not models:
        return tuple(models)

    peak_resistances = np.asarray(
        getattr(drt_result, "peak_resistances", ()), dtype=float
    )
    total_r = float(getattr(drt_result, "total_resistance", lambda: 0.0)())
    denominator = max(total_r, float(np.sum(peak_resistances)), 1e-30)
    significant = peak_resistances >= significance_fraction * denominator
    n_significant = int(np.count_nonzero(significant))
    max_arcs = max(n_significant, 1) + 1

    diffusion_plausible = _diffusion_evidence(
        drt_result, significance_fraction, slow_peak_safety_fraction
    )
    has_inductive = bool(getattr(drt_result, "has_inductive", True))

    survivors = [m for m in models if _model_arc_count(m) <= max_arcs]
    if not diffusion_plausible:
        survivors = [m for m in survivors if not _model_has_diffusion(m)]
    if not has_inductive:
        remaining_signatures = {
            frozenset(getattr(m, "parameter_names", ())) for m in survivors
        }
        survivors = [
            m
            for m in survivors
            if "L" not in getattr(m, "parameter_names", ())
            or frozenset(set(m.parameter_names) - {"L"}) not in remaining_signatures
        ]

    covering = [m for m in survivors if _model_arc_count(m) >= n_significant]
    if covering:
        k_reference = min(m.n_params for m in covering)
        survivors = [
            m for m in survivors if m.n_params <= k_reference + max_complexity_gap
        ]

    if len(survivors) < 2:
        logger.warning(
            "candidate screening left %d model(s); falling back to the full pool",
            len(survivors),
        )
        return tuple(models)
    if len(survivors) < len(models):
        logger.info(
            "candidate screening: %d -> %d models (dropped %s)",
            len(models),
            len(survivors),
            [m.name for m in models if m not in survivors],
        )
    return tuple(survivors)


# Preserve access to the screener inside compare_models_v2, whose
# `screen_candidates` boolean parameter shadows the function name.
_screen_candidate_pool = screen_candidates


def _select_parsimonious(
    results: Sequence[ModelFitResultV2],
    delta: dict[str, float],
    n_obs: int,
    parsimony_window: float,
    f_test_alpha: float,
    drt_result: Any | None = None,
) -> str:
    """Simplest model the raw AICc_eff winner does not beat significantly.

    Walking from the structurally simplest candidate upward, a simpler
    model is adequate when either it fits at least as well as the raw
    winner, or the winner's RSS improvement fails a nested F-test at
    ``f_test_alpha`` (i.e. the extra parameters are absorbing noise).
    Falls back to the Burnham-Anderson window rule among equally sized
    models when no strictly simpler candidate is adequate.

    Candidates of equal size are visited in DRT-evidence order: when the
    nonparametric DRT shows a diffusion signature, a diffusion-bearing
    structure is tried before a same-size arc-only one (and vice versa).
    The F-test alone cannot separate statistically equivalent structures
    (ΔAICc within a couple of units), so the independent DRT evidence is
    the deciding vote — never overriding a statistically better model.
    """

    evidence = _diffusion_evidence(drt_result)

    def matches_evidence(r: ModelFitResultV2) -> int:
        has_diffusion = bool(
            _DIFFUSION_PARAMS.intersection(r.tiered_selection.parameter_names)
        )
        return 0 if has_diffusion == evidence else 1

    order = lambda r: (r.n_params_full, matches_evidence(r), r.aicc_eff)  # noqa: E731

    best = results[0]
    for cand in sorted(results, key=order):
        if cand.model_name == best.model_name:
            break
        if cand.n_params_full >= best.n_params_full:
            continue
        if cand.rss <= best.rss:
            return cand.model_name
        try:
            p = f_test_nested(
                cand.rss, best.rss, cand.n_params_full, best.n_params_full, n_obs
            )
        except ValueError:  # pragma: no cover - defensive
            continue
        if p > f_test_alpha:
            return cand.model_name

    supported = [r for r in results if delta[r.model_name] <= parsimony_window]
    return min(supported, key=order).model_name


def _reduced_map_refit(
    dataset: EISDataset,
    model: Any,
    specs: tuple,
    names: tuple[str, ...],
    fixed_values: dict[str, float],
    prior_set: PriorSet,
    noise_level: float,
    weights: FloatArray | None,
    fitter: HierarchicalMAPFitter,
    previous_result: IdentificationResult,
) -> tuple[IdentificationResult, FloatArray]:
    """MAP-refit the model with the given parameters frozen.

    Returns a full-model :class:`IdentificationResult` (frozen values
    merged back in) and the full parameter vector in spec order.
    """

    subset = SubsetModel(
        full_model=model,
        full_specs=specs,
        fixed_values=dict(fixed_values),
    )
    reduced = fitter.fit(
        dataset=dataset,
        model=subset,
        parameter_specs=subset.parameter_specs,
        prior_set=prior_set,
        noise_level=noise_level,
        weights=weights,
        initial_guess=dict(previous_result.theta_best),
    )
    full_theta_best = dict(fixed_values)
    full_theta_best.update(reduced.theta_best)
    theta = np.asarray([full_theta_best[name] for name in names], dtype=float)
    z_fit = model.simulate(dataset.freq_hz, theta)
    metadata = dict(reduced.metadata)
    metadata["gate_fixed_parameters"] = sorted(fixed_values)
    result = IdentificationResult(
        theta_best=full_theta_best,
        final_cost=float(reduced.final_cost),
        z_fit=z_fit,
        residuals=dataset.z_obs - z_fit,
        dataset=dataset,
        metadata=metadata,
    )
    return result, theta


def _fit_and_gate_one(
    dataset: EISDataset,
    model: Any,
    drt_result: Any,
    noise_level: float,
    weights: FloatArray | None,
    gate: TieredIdentifiabilityGate,
    fitter: HierarchicalMAPFitter,
    prior_builder: DRTPriorBuilder,
    n_obs: int,
) -> ModelFitResultV2:
    specs = tuple(model.parameter_specs)
    names = tuple(spec.name for spec in specs)

    prior_set = prior_builder.build(model, drt_result, specs)

    # Stage 1-2: hierarchical MAP fit of the complete model.
    result = fitter.fit(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
        prior_set=prior_set,
        noise_level=noise_level,
        weights=weights,
    )
    theta = np.asarray([result.theta_best[name] for name in names], dtype=float)

    # Empirical-Bayes prior width calibration against the data-only fit:
    # a DRT prior contradicted by the data is widened before it can shape
    # either the MAP estimate or the posterior identifiability analysis.
    theta_data = result.metadata.get("stage_c_theta") or result.theta_best
    theta_data_vec = np.asarray([theta_data[name] for name in names], dtype=float)
    calibrated = prior_builder.calibrated(prior_set, specs, theta_data_vec)
    # Refinement refits start from an already-converged point, so a single
    # warm start suffices — multi-start exploration would only burn time.
    refit_fitter = replace(fitter, n_starts=1)
    if calibrated is not prior_set:
        prior_set = calibrated
        result = refit_fitter.fit(
            dataset=dataset,
            model=model,
            parameter_specs=specs,
            prior_set=prior_set,
            noise_level=noise_level,
            weights=weights,
            initial_guess={name: float(v) for name, v in theta_data.items()},
        )
        theta = np.asarray([result.theta_best[name] for name in names], dtype=float)

    # Tiered gate with iterative refinement: the Fisher information of the
    # reduced problem differs from the full one, so after freezing FIXED
    # parameters and refitting, the gate is re-evaluated at the refined
    # point; borderline parameters may be legitimately promoted (e.g.
    # PRIOR_INFORMED -> STRICT once a collinear direction is frozen).
    # Stops immediately when nothing is frozen or tiers are stable — the
    # common case costs exactly one gate evaluation, as before.
    selection = gate.evaluate(dataset, model, specs, theta, prior_set, weights)
    refitted_for: dict[str, float] | None = None
    gate_evaluations = 1
    max_iters = int(gate.max_gate_refinement_iters)
    for iteration in range(max_iters):
        fixed = selection.fixed_values
        if not fixed or len(fixed) >= len(specs) or fixed == refitted_for:
            break  # estimate already consistent with the tier assignment
        result, theta = _reduced_map_refit(
            dataset, model, specs, names, fixed, prior_set, noise_level,
            weights, refit_fitter, result,
        )
        refitted_for = fixed
        if iteration + 1 >= max_iters:
            break  # budget exhausted (max_iters=1 == one-shot Gen-2.0 gating)
        reevaluated = gate.evaluate(dataset, model, specs, theta, prior_set, weights)
        gate_evaluations += 1
        if reevaluated.tiers == selection.tiers:
            # Same taxonomy — adopt the sharper diagnostics computed at the
            # refined point and stop.
            selection = reevaluated
            break
        changes = {
            name: (selection.tiers[name].value, reevaluated.tiers[name].value)
            for name in names
            if selection.tiers[name] is not reevaluated.tiers[name]
        }
        logger.info(
            "gate refinement (%s, iteration %d): tier changes %s",
            model.name,
            iteration + 1,
            changes,
        )
        selection = reevaluated
    result.metadata["gate_evaluations"] = gate_evaluations

    # Raw (unweighted) residuals for information criteria, as in Gen 1.
    raw_residuals = dataset.z_obs - result.z_fit
    rss = float(np.sum(raw_residuals.real**2 + raw_residuals.imag**2))
    flat = np.concatenate([raw_residuals.real, raw_residuals.imag])

    k_eff = float(selection.k_eff)
    return ModelFitResultV2(
        model_name=model.name,
        circuit_string=getattr(model, "circuit_string", model.name),
        n_params_full=len(specs),
        n_params_free=len(selection.free_names),
        k_eff=k_eff,
        n_strict=selection.n_strict,
        n_prior_informed=selection.n_prior_informed,
        n_prior_dominated=selection.n_prior_dominated,
        n_fixed=selection.n_fixed,
        n_observations=n_obs,
        rss=rss,
        log_likelihood=compute_log_likelihood(n_obs, rss),
        aic_eff=compute_aic(n_obs, k_eff, rss),
        aicc_eff=compute_aicc(n_obs, k_eff, rss),
        bic_eff=compute_bic(n_obs, k_eff, rss),
        aicc_full=compute_aicc(n_obs, len(specs), rss),
        durbin_watson=durbin_watson(flat),
        ljung_box_p=ljung_box_p_value(flat),
        identification_result=result,
        tiered_selection=selection,
        prior_set=prior_set,
    )
