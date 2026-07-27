# learning AI at www.haotianblog.com
"""Model selection for EIS equivalent circuit models.

Compares candidate ECM models using information criteria (AIC, BIC, AICc),
residual diagnostics, and F-tests.  The core function :func:`compare_models`
fits each candidate and returns a :class:`ModelComparisonResult` that
identifies the best model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as _stats

from .dataset import EISDataset
from .results import IdentificationResult

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Per-model fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFitResult:
    """Fit statistics for a single candidate model.

    Attributes
    ----------
    model_name : str
        Short name of the ECM model (e.g. ``"2RCPE"``).
    circuit_string : str
        Human-readable circuit description.
    n_params : int
        Number of free parameters.
    n_observations : int
        Number of real-valued observations (2 × n_freq for complex data).
    rss : float
        Residual sum of squares (real + imaginary).
    log_likelihood : float
        Gaussian log-likelihood assuming i.i.d. errors.
    aic : float
        Akaike Information Criterion.
    aicc : float
        Corrected AIC for finite sample sizes.
    bic : float
        Bayesian Information Criterion.
    durbin_watson : float
        Durbin-Watson statistic for residual autocorrelation (ideal ≈ 2).
    ljung_box_p : float
        Ljung-Box test p-value for residual whiteness (p > 0.05 = white).
    identification_result : IdentificationResult
        Full optimizer result.
    """

    model_name: str
    circuit_string: str
    n_params: int
    n_observations: int
    rss: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    durbin_watson: float
    ljung_box_p: float
    identification_result: IdentificationResult


# ---------------------------------------------------------------------------
# Model comparison result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparison of multiple candidate models.

    Attributes
    ----------
    candidates : tuple[ModelFitResult, ...]
        Per-model fit results, ordered by AICc.
    best_by_aic : str
        Model name with lowest AIC.
    best_by_bic : str
        Model name with lowest BIC.
    best_by_aicc : str
        Model name with lowest AICc (recommended).
    delta_aic : dict[str, float]
        ΔAIC relative to the best AIC model.
    akaike_weights : dict[str, float]
        Model probabilities from Akaike weights.
    f_test_results : dict[tuple[str, str], float]
        Pairwise F-test p-values for nested model pairs.
    """

    candidates: tuple[ModelFitResult, ...]
    best_by_aic: str
    best_by_bic: str
    best_by_aicc: str
    delta_aic: dict[str, float]
    akaike_weights: dict[str, float]
    f_test_results: dict[tuple[str, str], float]

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [
            "Model Comparison Summary",
            "=" * 70,
            f"  Best by AICc: {self.best_by_aicc}",
            f"  Best by AIC:  {self.best_by_aic}",
            f"  Best by BIC:  {self.best_by_bic}",
            "",
            f"  {'Model':<16} {'k':>4} {'AICc':>12} {'ΔAIC':>10} "
            f"{'Weight':>8} {'BIC':>12} {'D-W':>6} {'L-B p':>7}",
            "  " + "-" * 75,
        ]
        for c in self.candidates:
            lines.append(
                f"  {c.model_name:<16} {c.n_params:4d} "
                f"{c.aicc:12.2f} {self.delta_aic[c.model_name]:10.2f} "
                f"{self.akaike_weights[c.model_name]:8.3f} "
                f"{c.bic:12.2f} {c.durbin_watson:6.2f} "
                f"{c.ljung_box_p:7.3f}"
            )
        if self.f_test_results:
            lines.append("")
            lines.append("  F-test results (nested pairs):")
            for (m1, m2), p in self.f_test_results.items():
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                lines.append(f"    {m1} → {m2}: p = {p:.4f} {sig}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame with per-model statistics."""
        rows = []
        for c in self.candidates:
            rows.append(
                {
                    "model": c.model_name,
                    "circuit": c.circuit_string,
                    "n_params": c.n_params,
                    "n_obs": c.n_observations,
                    "RSS": c.rss,
                    "log_likelihood": c.log_likelihood,
                    "AIC": c.aic,
                    "AICc": c.aicc,
                    "BIC": c.bic,
                    "delta_AIC": self.delta_aic[c.model_name],
                    "akaike_weight": self.akaike_weights[c.model_name],
                    "durbin_watson": c.durbin_watson,
                    "ljung_box_p": c.ljung_box_p,
                    "final_cost": c.identification_result.final_cost,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------


def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion: AIC = n·ln(RSS/n) + 2k."""
    if n <= 0 or rss <= 0:
        return np.inf
    return float(n * np.log(rss / n) + 2 * k)


def compute_aicc(n: int, k: int, rss: float) -> float:
    """Corrected AIC for small samples: AICc = AIC + 2k(k+1)/(n-k-1)."""
    aic = compute_aic(n, k, rss)
    if n - k - 1 <= 0:
        return np.inf
    return float(aic + 2 * k * (k + 1) / (n - k - 1))


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion: BIC = n·ln(RSS/n) + k·ln(n)."""
    if n <= 0 or rss <= 0:
        return np.inf
    return float(n * np.log(rss / n) + k * np.log(n))


def compute_log_likelihood(n: int, rss: float) -> float:
    """Gaussian log-likelihood: LL = -n/2 · (1 + ln(2π) + ln(RSS/n))."""
    if n <= 0 or rss <= 0:
        return -np.inf
    return float(-n / 2 * (1 + np.log(2 * np.pi) + np.log(rss / n)))


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def durbin_watson(residuals: FloatArray) -> float:
    """Durbin-Watson statistic for autocorrelation (ideal ≈ 2.0).

    D-W < 1.5 suggests positive autocorrelation (systematic model misfit).
    D-W > 2.5 suggests negative autocorrelation.
    """
    if residuals.size < 2:
        return 2.0  # neutral
    d = np.diff(residuals)
    ss_res = np.sum(residuals**2)
    if ss_res < 1e-30:
        return 2.0
    return float(np.sum(d**2) / ss_res)


def ljung_box_p_value(residuals: FloatArray, n_lags: int = 10) -> float:
    """Ljung-Box test p-value for residual whiteness.

    p > 0.05 means residuals appear white (good model fit).
    p < 0.05 means significant autocorrelation (model misfit).
    """
    n = residuals.size
    if n < n_lags + 2:
        n_lags = max(1, n - 2)
    if n < 3:
        return 1.0  # not enough data

    # Compute autocorrelation function
    mean = np.mean(residuals)
    centered = residuals - mean
    var = np.sum(centered**2) / n
    if var < 1e-30:
        return 1.0

    acf = np.zeros(n_lags)
    for lag in range(1, n_lags + 1):
        acf[lag - 1] = np.sum(centered[:-lag] * centered[lag:]) / (n * var)

    # Ljung-Box Q statistic
    q = n * (n + 2) * np.sum(acf**2 / np.arange(n - 1, n - n_lags - 1, -1))
    # p-value from chi-squared distribution
    p = 1.0 - _stats.chi2.cdf(q, df=n_lags)
    return float(p)


# ---------------------------------------------------------------------------
# F-test for nested models
# ---------------------------------------------------------------------------


def f_test_nested(
    rss_simple: float,
    rss_complex: float,
    k_simple: int,
    k_complex: int,
    n: int,
) -> float:
    """F-test p-value for comparing nested models.

    Tests whether the complex model provides a statistically significant
    improvement over the simple model.

    Parameters
    ----------
    rss_simple : float
        RSS of the simpler model (fewer parameters).
    rss_complex : float
        RSS of the more complex model (more parameters).
    k_simple : int
        Number of parameters in the simpler model.
    k_complex : int
        Number of parameters in the more complex model.
    n : int
        Number of observations.

    Returns
    -------
    float
        p-value. Small values (< 0.05) indicate the complex model is
        significantly better.
    """
    if k_complex <= k_simple:
        raise ValueError("Complex model must have more parameters")
    if n <= k_complex:
        return 1.0  # not enough data

    df1 = k_complex - k_simple
    df2 = n - k_complex

    if rss_complex <= 0 or df2 <= 0:
        return 1.0

    f_stat = ((rss_simple - rss_complex) / df1) / (rss_complex / df2)
    if f_stat < 0:
        return 1.0  # complex model is worse (shouldn't happen but be safe)

    p = 1.0 - _stats.f.cdf(f_stat, df1, df2)
    return float(p)


# ---------------------------------------------------------------------------
# Post-fit arc sorting
# ---------------------------------------------------------------------------


def _sort_rcpe_arcs(
    result: Any,
    model_name: str,
) -> Any:
    """Sort RCPE arcs by time constant (τ ascending = high freq first).

    Multi-RCPE models have a label permutation ambiguity: the optimizer
    may assign Arc 1 parameters to the slow arc and vice versa.  This
    function resolves the ambiguity by sorting arcs so that
    τ₁ < τ₂ (< τ₃), i.e. Arc 1 is always the fastest.

    The time constant for an R‖CPE element is τ = (R·Q)^(1/α).
    """
    params = dict(result.theta_best)

    if model_name in ("2RCPE", "2RCPE_W"):
        r1 = params.get("R1", 0)
        q1 = params.get("Q1", 0)
        a1 = params.get("alpha1", 0.85)
        r2 = params.get("R2", 0)
        q2 = params.get("Q2", 0)
        a2 = params.get("alpha2", 0.85)

        if r1 > 0 and q1 > 0 and r2 > 0 and q2 > 0:
            tau1 = (r1 * q1) ** (1.0 / max(a1, 0.01))
            tau2 = (r2 * q2) ** (1.0 / max(a2, 0.01))

            if tau1 > tau2:
                # Swap arcs so τ₁ < τ₂
                params["R1"], params["R2"] = r2, r1
                params["Q1"], params["Q2"] = q2, q1
                params["alpha1"], params["alpha2"] = a2, a1

                # Rebuild the result with swapped params
                from .results import IdentificationResult
                return IdentificationResult(
                    theta_best=params,
                    final_cost=result.final_cost,
                    z_fit=result.z_fit,
                    residuals=result.residuals,
                    dataset=result.dataset,
                    metadata=result.metadata,
                )

    elif model_name == "3RCPE":
        r_vals = [params.get(f"R{i}", 0) for i in range(1, 4)]
        q_vals = [params.get(f"Q{i}", 0) for i in range(1, 4)]
        a_vals = [params.get(f"alpha{i}", 0.85) for i in range(1, 4)]

        if all(r > 0 for r in r_vals) and all(q > 0 for q in q_vals):
            taus = [
                (r * q) ** (1.0 / max(a, 0.01))
                for r, q, a in zip(r_vals, q_vals, a_vals)
            ]
            order = np.argsort(taus)

            if not np.array_equal(order, [0, 1, 2]):
                for new_idx, old_idx in enumerate(order):
                    params[f"R{new_idx+1}"] = r_vals[old_idx]
                    params[f"Q{new_idx+1}"] = q_vals[old_idx]
                    params[f"alpha{new_idx+1}"] = a_vals[old_idx]

                from .results import IdentificationResult
                return IdentificationResult(
                    theta_best=params,
                    final_cost=result.final_cost,
                    z_fit=result.z_fit,
                    residuals=result.residuals,
                    dataset=result.dataset,
                    metadata=result.metadata,
                )

    return result


# ---------------------------------------------------------------------------
# Profile refinement for Q-α collinearity
# ---------------------------------------------------------------------------


def _profile_refine_rcpe(
    result: Any,
    model_name: str,
    model: Any,
    dataset: Any,
    specs: tuple,
    weights: Any | None = None,
    n_alpha_points: int = 15,
) -> Any:
    """Break Q-α collinearity via profile likelihood over α.

    For each RCPE arc in the model, profile over α ∈ [0.5, 1.0]:
    fix α, re-optimize (R, Q) via local least squares, pick the
    combination that minimises the total cost.  This resolves the
    well-known Q-α degeneracy in CPE fitting.
    """
    from scipy.optimize import least_squares as _lsq

    from .costs import EISPredictionErrorCost
    from .parameters import decode_parameter_vector, optimization_bounds_array
    from .results import IdentificationResult

    # Detect which arcs exist
    arc_indices: list[tuple[int, int, int]] = []  # (R_idx, Q_idx, alpha_idx)
    param_names = [s.name for s in specs]
    for arc_num in range(1, 4):
        r_name = f"R{arc_num}"
        q_name = f"Q{arc_num}"
        a_name = f"alpha{arc_num}"
        if r_name in param_names and q_name in param_names and a_name in param_names:
            arc_indices.append((
                param_names.index(r_name),
                param_names.index(q_name),
                param_names.index(a_name),
            ))

    if not arc_indices:
        return result

    cost = EISPredictionErrorCost(
        dataset=dataset, model=model, weights=weights,
        relative=True, modulus_weighting=False, eps=1e-12,
    )

    best_params = dict(result.theta_best)
    best_cost = result.final_cost
    improved = False

    for r_idx, q_idx, a_idx in arc_indices:
        alpha_lo = max(specs[a_idx].bounds[0], 0.50)
        alpha_hi = min(specs[a_idx].bounds[1], 1.00)
        alpha_grid = np.linspace(alpha_lo, alpha_hi, n_alpha_points)

        best_arc_cost = best_cost
        best_arc_params = dict(best_params)

        for alpha_trial in alpha_grid:
            trial_params = dict(best_params)
            trial_params[specs[a_idx].name] = alpha_trial

            # Build search vector with alpha fixed
            free_indices = [i for i in range(len(specs)) if i != a_idx]
            free_specs = [specs[i] for i in free_indices]
            free_values = np.array([trial_params[s.name] for s in free_specs])

            # Encode free params
            sv_free = np.array([
                np.log10(v) if s.log_transform else v
                for s, v in zip(free_specs, free_values)
            ])
            bounds_free = np.array([s.optimization_bounds for s in free_specs])
            sv_free = np.clip(sv_free, bounds_free[:, 0], bounds_free[:, 1])

            def _make_residual_fn(
                _free_indices: list,
                _a_idx: int,
                _alpha: float,
                _specs: tuple,
                _cost: Any,
            ):
                def _fn(sv: np.ndarray) -> np.ndarray:
                    theta = np.empty(len(_specs))
                    for j, idx in enumerate(_free_indices):
                        val = sv[j]
                        if _specs[idx].log_transform:
                            val = 10.0 ** val
                        theta[idx] = val
                    theta[_a_idx] = _alpha
                    res = _cost.residuals(theta)
                    return np.concatenate((res.real, res.imag))
                return _fn

            residual_fn = _make_residual_fn(
                free_indices, a_idx, alpha_trial, specs, cost
            )

            try:
                ls_result = _lsq(
                    residual_fn, sv_free,
                    bounds=(bounds_free[:, 0], bounds_free[:, 1]),
                    max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12,
                )
                trial_cost = 2.0 * float(ls_result.cost)

                if trial_cost < best_arc_cost:
                    best_arc_cost = trial_cost
                    best_arc_params = dict(trial_params)
                    # Update from optimized free params
                    for j, idx in enumerate(free_indices):
                        val = ls_result.x[j]
                        if specs[idx].log_transform:
                            val = 10.0 ** val
                        best_arc_params[specs[idx].name] = float(val)
                    best_arc_params[specs[a_idx].name] = float(alpha_trial)
            except Exception:
                continue

        if best_arc_cost < best_cost:
            best_cost = best_arc_cost
            best_params = best_arc_params
            improved = True

    if improved:
        theta_vec = np.array([best_params[s.name] for s in specs])
        z_fit = model.simulate(dataset.freq_hz, theta_vec)
        residuals = dataset.z_obs - z_fit
        metadata = dict(result.metadata)
        metadata["profile_refined"] = True
        return IdentificationResult(
            theta_best=best_params,
            final_cost=best_cost,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )

    return result


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------


def compare_models(
    dataset: EISDataset,
    candidates: Sequence[Any],
    optimizer: Any | None = None,
    weights: FloatArray | None = None,
    relative: bool = True,
    max_nfev: int = 5000,
    initial_guesses: dict[str, dict[str, float]] | None = None,
    nyquist_guesses: dict[str, dict[str, float]] | None = None,
) -> ModelComparisonResult:
    """Fit multiple candidate ECM models and compare with AIC/BIC.

    Parameters
    ----------
    dataset : EISDataset
        Impedance dataset (filtered/weighted).
    candidates : Sequence[ECMModel]
        Candidate models to compare.  Each must implement ``ForwardModel``
        and have ``name``, ``circuit_string``, ``parameter_specs``, and
        ``n_params`` attributes.
    optimizer
        Optimizer instance (default: ``LeastSquaresOptimizer``).
    weights : FloatArray | None
        Optional per-point weights.
    relative : bool
        Whether to use relative residuals.
    max_nfev : int
        Maximum function evaluations per model.

    Returns
    -------
    ModelComparisonResult
        Full comparison with best model selection.
    """
    from .optimizers import LeastSquaresOptimizer

    if optimizer is None:
        optimizer = LeastSquaresOptimizer(relative=relative, max_nfev=max_nfev)

    if not candidates:
        raise ValueError("At least one candidate model is required")

    n_freq = dataset.freq_hz.size
    n_obs = 2 * n_freq  # real + imaginary parts

    fit_results: list[ModelFitResult] = []

    # Track best results per model for cascading warm-start
    _best_per_model: dict[str, dict[str, float]] = {}

    # Sort candidates by complexity (fewer params first) for cascading
    sorted_candidates = sorted(candidates, key=lambda m: m.n_params)

    for model in sorted_candidates:
        specs = list(model.parameter_specs)
        k = model.n_params

        # --- Collect multi-strategy initial guesses ---
        all_guesses: list[dict[str, float]] = []

        # Strategy 1: DRT-informed guess
        if initial_guesses and model.name in initial_guesses:
            all_guesses.append(initial_guesses[model.name])

        # Strategy 2: Nyquist-based guess
        if nyquist_guesses and model.name in nyquist_guesses:
            all_guesses.append(nyquist_guesses[model.name])

        # Strategy 3: Warm-start from simpler model
        for simpler_name, simpler_params in _best_per_model.items():
            warm = {}
            for s in specs:
                if s.name in simpler_params:
                    warm[s.name] = simpler_params[s.name]
            if warm:
                all_guesses.append(warm)

        # Deduplicate and ensure at least one (None = default)
        if not all_guesses:
            all_guesses.append({})

        # --- Try each initial guess strategy ---
        best_result = None
        best_cost_value = float("inf")

        for guess in all_guesses:
            try:
                result = optimizer.fit(
                    dataset=dataset,
                    model=model,
                    parameter_specs=specs,
                    weights=weights,
                    initial_guess=guess if guess else None,
                )
                if result.final_cost < best_cost_value:
                    best_cost_value = result.final_cost
                    best_result = result
            except Exception:
                continue

        # Also try with no initial guess (default bounds center)
        if len(all_guesses) > 0 and all(bool(g) for g in all_guesses):
            try:
                result = optimizer.fit(
                    dataset=dataset,
                    model=model,
                    parameter_specs=specs,
                    weights=weights,
                    initial_guess=None,
                )
                if result.final_cost < best_cost_value:
                    best_cost_value = result.final_cost
                    best_result = result
            except Exception:
                pass

        if best_result is None:
            continue

        # --- Refinement stage: re-fit from best solution with tightened tolerances ---
        try:
            from .optimizers import LeastSquaresOptimizer
            refine_optimizer = LeastSquaresOptimizer(
                relative=getattr(optimizer, 'relative', True),
                max_nfev=20000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                n_starts=1,
            )
            refined = refine_optimizer.fit(
                dataset=dataset,
                model=model,
                parameter_specs=specs,
                weights=weights,
                initial_guess=best_result.theta_best,
            )
            if refined.final_cost < best_cost_value:
                best_result = refined
                best_cost_value = refined.final_cost
        except Exception:
            pass

        # --- Post-fit: sort arcs by time constant (τ ascending) ---
        # Resolves label permutation ambiguity in multi-RCPE models
        best_result = _sort_rcpe_arcs(best_result, model.name)

        # --- Post-fit: profile refinement for Q-α collinearity ---
        # Profiles over α values for each RCPE arc, re-optimising R,Q
        best_result = _profile_refine_rcpe(
            result=best_result,
            model_name=model.name,
            model=model,
            dataset=dataset,
            specs=specs,
            weights=weights,
        )

        # Store best params for cascading to more complex models
        _best_per_model[model.name] = dict(best_result.theta_best)

        # Compute RSS from raw (unweighted) residuals for information criteria
        z_fit = model.simulate(
            dataset.freq_hz,
            np.array([best_result.theta_best[s.name] for s in specs]),
        )
        raw_residuals = dataset.z_obs - z_fit
        rss = float(np.sum(raw_residuals.real**2 + raw_residuals.imag**2))

        # Flatten residuals for autocorrelation tests (interleave real/imag)
        flat_residuals = np.concatenate([raw_residuals.real, raw_residuals.imag])

        # Compute information criteria
        aic = compute_aic(n_obs, k, rss)
        aicc = compute_aicc(n_obs, k, rss)
        bic = compute_bic(n_obs, k, rss)
        ll = compute_log_likelihood(n_obs, rss)

        # Residual diagnostics
        dw = durbin_watson(flat_residuals)
        lb_p = ljung_box_p_value(flat_residuals)

        fit_results.append(
            ModelFitResult(
                model_name=model.name,
                circuit_string=model.circuit_string,
                n_params=k,
                n_observations=n_obs,
                rss=rss,
                log_likelihood=ll,
                aic=aic,
                aicc=aicc,
                bic=bic,
                durbin_watson=dw,
                ljung_box_p=lb_p,
                identification_result=best_result,
            )
        )

    if not fit_results:
        raise RuntimeError("All candidate models failed to fit")

    # Sort by AICc
    fit_results.sort(key=lambda r: r.aicc)

    # Find best by each criterion
    best_aic = min(fit_results, key=lambda r: r.aic).model_name
    best_bic = min(fit_results, key=lambda r: r.bic).model_name
    best_aicc = fit_results[0].model_name

    # Delta AIC and Akaike weights
    min_aic = min(r.aic for r in fit_results)
    delta_aic = {r.model_name: r.aic - min_aic for r in fit_results}
    exp_delta = {name: np.exp(-0.5 * d) for name, d in delta_aic.items()}
    sum_exp = sum(exp_delta.values())
    akaike_weights = {
        name: float(w / sum_exp) if sum_exp > 0 else 0.0
        for name, w in exp_delta.items()
    }

    # F-tests for nested model pairs
    # Models are considered nested if the simpler one's circuit elements
    # are a subset. We test all pairs where k_i < k_j.
    f_test_results: dict[tuple[str, str], float] = {}
    for i, r1 in enumerate(fit_results):
        for r2 in fit_results[i + 1 :]:
            if r1.n_params < r2.n_params:
                simple, complex_ = r1, r2
            elif r2.n_params < r1.n_params:
                simple, complex_ = r2, r1
            else:
                continue
            try:
                p = f_test_nested(
                    simple.rss, complex_.rss,
                    simple.n_params, complex_.n_params,
                    n_obs,
                )
                f_test_results[(simple.model_name, complex_.model_name)] = p
            except ValueError:
                pass

    return ModelComparisonResult(
        candidates=tuple(fit_results),
        best_by_aic=best_aic,
        best_by_bic=best_bic,
        best_by_aicc=best_aicc,
        delta_aic=delta_aic,
        akaike_weights=akaike_weights,
        f_test_results=f_test_results,
    )
