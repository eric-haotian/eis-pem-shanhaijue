# learning AI at www.haotianblog.com
"""End-to-end Generation-2 identification pipeline.

Drop-in upgrade of :func:`eis_pem.identify_with_model_selection`: the
request keys are a superset of the Generation-1 contract, and the
response mirrors the Generation-1 shape while adding the tiered
identifiability taxonomy, effective degrees of freedom, and the prior
audit table.

Pipeline: data quality → frequency filtering → DRT → candidate models
(expanded Gen-2 library) → DRT priors → hierarchical MAP fit → tiered
gate → effective-dof AICc ranking → report.

Everything upstream of the Generation-2 additions (quality gate,
frequency filter, DRT, Nyquist features, physics priors) is reused from
the untouched Generation-1 package.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from .data_quality import assess_data_quality
from .dataset import EISDataset
from .drt import DRTAnalyzer
from .frequency_filter import FrequencyBandAnalyzer, FrequencyBandConfig, weighted_dataset
from .nyquist_init import extract_nyquist_features
from .physics_priors import check_prior_violations, lithium_ion_priors, pemfc_priors

from .fitting import HierarchicalMAPFitter
from .gating import TieredIdentifiabilityGate
from .models import ecm_model_by_name_v2, suggest_models_v2
from .priors import DRTPriorBuilder
from .selection import ModelFitResultV2, compare_models_v2

logger = logging.getLogger(__name__)

#: Bounds used when adopting the estimated noise level for gating.
_NOISE_FLOOR = 0.002
_NOISE_CEIL = 0.05


def estimate_noise_level(dataset: EISDataset, quality: Any | None = None) -> float:
    """Estimate the relative measurement noise of an EIS spectrum.

    Primary source is the Generation-1 data-quality assessment
    (``estimated_noise_level``: KK-residual / smoothness based, accurate
    to ~10% on synthetic benchmarks).  When that is unavailable or
    degenerate, falls back to a second-difference estimator restricted
    to the highest frequency decade, where the spectrum is locally flat
    so curvature bias is minimal: for i.i.d. relative noise of standard
    deviation sigma per component, the second difference has variance
    ``6 sigma^2``.
    """

    if quality is not None:
        value = float(getattr(quality, "estimated_noise_level", np.nan))
        if np.isfinite(value) and value > 0:
            return float(np.clip(value, _NOISE_FLOOR, _NOISE_CEIL))

    freq = np.asarray(dataset.freq_hz, dtype=float)
    z = np.asarray(dataset.z_obs, dtype=complex)
    order = np.argsort(freq)
    freq, z = freq[order], z[order]
    high = freq >= freq.max() / 10.0
    if np.count_nonzero(high) < 5:
        high = np.ones_like(freq, dtype=bool)
    z_high = z[high]
    if z_high.size < 3:
        return 0.005
    d2 = np.diff(z_high, n=2) / np.maximum(np.abs(z_high[1:-1]), 1e-30)
    # Robust per-component scale (median absolute deviation -> sigma).
    sigma = (
        1.4826
        * np.hypot(np.median(np.abs(d2.real)), np.median(np.abs(d2.imag)))
        / np.sqrt(6.0)
    )
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.005
    return float(np.clip(sigma, _NOISE_FLOOR, _NOISE_CEIL))


def identify_with_model_selection_v2(
    request: Mapping[str, Any],
    noise_level: float | None = None,
) -> dict[str, Any]:
    """Full Generation-2 pipeline; request contract is a superset of Gen 1.

    Additional optional request keys (all with safe defaults):

    - ``include_gen1_baselines`` (bool, default True): add Gen-1 models
      to the auto-suggested candidate pool for cross-generation audits.
    - ``max_posterior_ci95`` (float, default 0.20)
    - ``min_information_gain`` (float, default 0.50)
    - ``assumed_noise_level`` (float): overrides the estimated noise.
    - ``polish_data_only`` (bool, default True)

    Noise handling: the ``noise_level`` argument (highest precedence) or
    the ``assumed_noise_level`` request key overrides auto-estimation;
    otherwise :func:`estimate_noise_level` derives it from the data.
    Whatever the source, the *same* value is threaded into both the
    :class:`HierarchicalMAPFitter` (MAP prior-row scaling) and the
    :class:`TieredIdentifiabilityGate` (Fisher whitening), so the fit
    and the gate can never silently disagree about the noise floor.
    """

    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")

    freq_hz = np.asarray(request["freq_hz"], dtype=float)
    z_obs = (
        np.asarray(request["z_obs_real"], dtype=float)
        + 1j * np.asarray(request["z_obs_imag"], dtype=float)
    )
    dataset = EISDataset(freq_hz=freq_hz, z_obs=z_obs)

    # --- Step 1: data quality (Gen 1) ---
    quality = assess_data_quality(dataset)

    # --- Step 2: frequency filtering (Gen 1) ---
    filter_overrides = request.get("filter_config", {})
    filter_config = (
        FrequencyBandConfig(**filter_overrides) if filter_overrides
        else FrequencyBandConfig()
    )
    filter_result = FrequencyBandAnalyzer(config=filter_config).analyze(dataset)
    filtered_dataset, weights = weighted_dataset(dataset, filter_result)

    # --- Step 3: DRT (Gen 1) ---
    drt_result = DRTAnalyzer().analyze(filtered_dataset)

    # --- Step 4: candidate models (Gen 2 library) ---
    explicit_models = request.get("candidate_models")
    if explicit_models:
        candidates = [ecm_model_by_name_v2(name) for name in explicit_models]
    else:
        candidates = list(
            suggest_models_v2(
                n_peaks=drt_result.n_peaks,
                has_diffusion_tail=drt_result.has_diffusion_tail,
                has_inductive=drt_result.has_inductive,
                include_gen1_baselines=bool(
                    request.get("include_gen1_baselines", True)
                ),
            )
        )

    # --- Step 5: noise level for MAP weighting and gating ---
    if noise_level is not None:
        noise_source = "argument"
        noise_level = float(noise_level)
    elif "assumed_noise_level" in request:
        noise_source = "request"
        noise_level = float(request["assumed_noise_level"])
    else:
        noise_source = "estimated"
        noise_level = estimate_noise_level(filtered_dataset, quality)
    if not np.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("noise_level must be finite and positive")

    # --- Step 6: configure Generation-2 machinery ---
    gate = TieredIdentifiabilityGate(
        assumed_noise_level=noise_level,
        max_posterior_ci95=float(request.get("max_posterior_ci95", 0.20)),
        min_information_gain=float(request.get("min_information_gain", 0.50)),
    )
    fitter = HierarchicalMAPFitter(
        n_starts=int(request.get("n_starts", 6)),
        max_nfev=int(request.get("max_nfev", 8000)),
        polish_data_only=bool(request.get("polish_data_only", True)),
    )
    prior_builder = DRTPriorBuilder()

    # --- Step 7: fit, gate, and rank ---
    comparison = compare_models_v2(
        dataset=filtered_dataset,
        candidates=candidates,
        drt_result=drt_result,
        noise_level=noise_level,
        weights=weights,
        gate=gate,
        fitter=fitter,
        prior_builder=prior_builder,
        parsimony_window=float(request.get("parsimony_window", 2.0)),
        # An explicit candidate list expresses user intent; only
        # auto-suggested pools are screened against the DRT evidence.
        screen_candidates=(
            bool(request.get("screen_candidates", True)) and not explicit_models
        ),
    )
    best = next(
        c for c in comparison.candidates
        if c.model_name == comparison.best_parsimonious
    )

    # --- Step 8: physics prior check (Gen 1, name-tolerant) ---
    prior_set_name = str(request.get("priors", "lithium_ion"))
    violations = _physics_prior_violations(best, prior_set_name)

    # --- Step 9: Nyquist features for the report (Gen 1) ---
    nyquist = extract_nyquist_features(filtered_dataset.freq_hz, filtered_dataset.z_obs)

    return {
        "generation": 2,
        "best_model": best.model_name,
        "best_circuit": best.circuit_string,
        "theta_best": {
            k: float(v) for k, v in best.identification_result.theta_best.items()
        },
        "final_cost": float(best.identification_result.final_cost),
        "quality_score": float(quality.quality_score),
        "quality_warnings": list(quality.warnings),
        "estimated_noise_level": float(quality.estimated_noise_level),
        "assumed_noise_level": float(noise_level),
        "noise_level_source": noise_source,
        "n_points_used": int(filtered_dataset.freq_hz.size),
        "n_points_total": int(dataset.freq_hz.size),
        "drt": {
            "n_peaks": drt_result.n_peaks,
            "peak_frequencies_hz": [
                float(1.0 / (2 * np.pi * t)) for t in drt_result.peak_taus
            ]
            if drt_result.n_peaks > 0
            else [],
            "peak_resistances": [float(r) for r in drt_result.peak_resistances],
            "has_diffusion_tail": drt_result.has_diffusion_tail,
            "has_inductive": drt_result.has_inductive,
            "r_inf": float(drt_result.r_inf),
            "total_polarisation_R": float(drt_result.total_resistance()),
        },
        "nyquist_features": {
            "n_arcs": nyquist.n_arcs,
            "rs_estimate": float(nyquist.rs_estimate),
            "total_resistance": float(nyquist.total_resistance),
        },
        "comparison": comparison.to_frame().to_dict(orient="records"),
        "selection_criterion": "aicc_eff_parsimonious",
        "best_by_aicc_eff": comparison.best_by_aicc_eff,
        "prior_violations": violations,
        "identifiability": {
            "tiers": {
                name: best.tiered_selection.tiers[name].value
                for name in best.tiered_selection.parameter_names
            },
            "n_strict": best.n_strict,
            "n_prior_informed": best.n_prior_informed,
            "n_prior_dominated": best.n_prior_dominated,
            "n_fixed": best.n_fixed,
            "n_identifiable": best.n_identifiable,
            "k_eff": float(best.k_eff),
            "data_condition_number": float(
                best.tiered_selection.data_condition_number
            ),
            "posterior_condition_number": float(
                best.tiered_selection.posterior_condition_number
            ),
            "information_gain": {
                k: float(v)
                for k, v in best.tiered_selection.information_gain.items()
            },
            "posterior_ci95_relative": {
                k: _finite_or_none(v)
                for k, v in best.tiered_selection.posterior_ci95.items()
            },
        },
        "parameters": _parameter_rows(best),
        "prior_table": best.prior_set.to_rows(tuple(
            ecm_model_by_name_v2(best.model_name).parameter_specs
        )),
        "metadata": best.identification_result.metadata,
    }


def _parameter_rows(best: ModelFitResultV2) -> list[dict[str, Any]]:
    selection = best.tiered_selection
    rows: list[dict[str, Any]] = []
    for name in selection.parameter_names:
        tier = selection.tiers[name]
        rows.append(
            {
                "parameter": name,
                "value": float(best.identification_result.theta_best[name]),
                "tier": tier.value,
                "status": selection.statuses[name],
                "posterior_ci95_relative": _finite_or_none(
                    selection.posterior_ci95[name]
                ),
                "information_gain": float(selection.information_gain[name]),
                "is_identifiable_claim": tier.value in ("strict", "prior_informed"),
                "strict_status": selection.strict_selection.statuses[name],
            }
        )
    return rows


def _physics_prior_violations(best: ModelFitResultV2, prior_set_name: str) -> list[str]:
    if prior_set_name == "lithium_ion":
        priors = lithium_ion_priors()
    elif prior_set_name == "pemfc":
        priors = pemfc_priors()
    else:
        return []
    params = best.identification_result.theta_best
    names = list(params.keys())
    theta = np.asarray(list(params.values()), dtype=float)
    try:
        return list(check_prior_violations(priors, names, theta))
    except Exception:
        # Gen-2 parameter names (tau*, gamma, Rion, ...) are not covered by
        # every Gen-1 prior definition; treat unmatched priors as satisfied.
        logger.debug("physics prior check skipped", exc_info=True)
        return []


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
