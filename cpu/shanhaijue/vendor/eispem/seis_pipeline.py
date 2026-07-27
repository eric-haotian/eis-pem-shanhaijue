# learning AI at www.haotianblog.com
"""Generation-2 SEIS physics-based parameter identification.

Extends the Gen-2 machinery (posterior gate, MAP priors) to the 48-parameter
DFN/SEIS forward model from the archived Generation-1 package, which is
imported unchanged.  The central objective is to push the identifiable
count beyond Generation 1's strict-gate baseline (~8-12 on the 3x3
temperature/SOC grid) without ever overclaiming.

Design levers, in order of importance:

1. **Multi-condition stacking.**  The stacked Fisher information is the
   sum of condition-specific blocks ``F = sum_c J_c^T J_c / sigma_n^2``;
   temperature variation separates Arrhenius activation energies from
   their base coefficients, SOC variation separates stoichiometry-window
   parameters.  :func:`select_informative_conditions` greedily selects
   conditions by their trace contribution.

2. **Candidate-subset gating.**  A joint 48-parameter posterior is so
   collinear that marginal confidence intervals explode even with
   priors (2 claims out of 48 in the design probe).  Gating a candidate
   *subset* — the remaining parameters explicitly FIXED at literature
   values and disclosed as such — is Generation 1's own semantics, and
   the subset posterior is well conditioned.  The identifiable set is
   found by a deterministic annealing search over subsets that maximises
   the number of parameters passing the claim test
   (information gain >= ``min_information_gain`` and posterior CI95 <=
   ``max_posterior_ci95``), which plain greedy provably under-finds
   (greedy stalls at 8-9 claims where annealing reaches 15-16).

3. **Physics priors.**  Literature values from the SEIS default
   parameter table are the prior means (:class:`SEISPhysicsPriorBuilder`);
   widths are tight for narrow linear parameters (span/12, e.g. 0.05 for
   charge-transfer symmetry factors) and capped at one decade for
   log-transformed parameters.

4. **Tier bookkeeping.**  :class:`SEISParameterGrouper` assigns each of
   the 48 parameters to tiers A-D from the joint-posterior information
   gain, providing the unlocking order for the subset search.

The final identifiability report always comes from one full
:class:`~eis_pem_v2.gating.TieredIdentifiabilityGate` evaluation on the
attempted subset at the fitted point, so every claim carries either the
unchanged Gen-1 strict guarantee or the posterior information-gain audit.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .optimizers import AdaptiveLeastSquaresOptimizer
from .parameters import ParameterSpec, encode_parameter_vector
from .seis_model import (
    SEISModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    stacked_seis_condition_context,
)

from .fitting import SubsetModel
from .gating import (
    ParameterTier,
    TieredIdentifiabilityGate,
    TieredSelection,
    relative_search_jacobian,
)
from .nuisance import NuisancePolicy, parse_nuisance_policy
from .physics_reparameterization import PhysicsProductReparameterizer
from .prior_policy import resolve_legacy_prior_tightening
from .priors import GaussianParameterPrior, PriorSet
from .seis_reparameterization import (
    ArrheniusReparameterizer,
    optimal_reference_temperatures,
)

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

#: Loop seed (the task-specified Tier-A intuition): parameters that appear
#: directly in the interfacial impedance formula.  The empirical grouper
#: refines the *order* of everything else.
DEFAULT_TIER_A_SEED = (
    "Ds_neg_0",
    "rs_neg",
    "k_neg_0",
    "rou_sei_neg_0",
    "Cdl_neg",
    "alpha_dl_neg",
    "R_contact",
    "L_ind",
)


# ---------------------------------------------------------------------------
# Physics priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEISPhysicsPriorBuilder:
    """Literature-value Gaussian priors for the 48 SEIS parameters.

    Means come from the SEIS default parameter table (the model's
    ``ParameterSpec.initial_value``); ``mean_overrides`` supports
    prior-sensitivity studies.  Widths: linear parameters get
    ``span * linear_span_fraction`` (0.05 for a 0.6-wide symmetry
    factor), log-transformed parameters ``span_decades *
    log_span_fraction`` capped at ``max_log_sigma_decades``.
    """

    linear_span_fraction: float = 1.0 / 12.0
    log_span_fraction: float = 0.25
    max_log_sigma_decades: float = 1.0
    mean_overrides: Mapping[str, float] = field(default_factory=dict)

    def sigma_search(self, spec: ParameterSpec) -> float:
        lower, upper = spec.optimization_bounds
        span = upper - lower
        if spec.log_transform:
            return float(min(self.max_log_sigma_decades, span * self.log_span_fraction))
        return float(span * self.linear_span_fraction)

    def build(self, specs: Sequence[ParameterSpec] | None = None) -> PriorSet:
        spec_list = tuple(specs) if specs is not None else tuple(all_seis_parameter_specs())
        priors = []
        for spec in spec_list:
            mean = float(self.mean_overrides.get(spec.name, spec.initial_value))
            mean = min(max(mean, spec.bounds[0]), spec.bounds[1])
            priors.append(
                GaussianParameterPrior(
                    name=spec.name,
                    mean_physical=mean,
                    sigma_search=self.sigma_search(spec),
                    source="literature",
                )
            )
        return PriorSet(priors=tuple(priors))


# ---------------------------------------------------------------------------
# Condition selection
# ---------------------------------------------------------------------------


def condition_information_traces(
    model_class: type,
    candidate_conditions: Sequence[tuple[float, float]],
    theta_reference: NDArray[np.floating],
    freq_hz: NDArray[np.floating],
    parameter_names: Sequence[str] | None = None,
) -> dict[tuple[float, float], float]:
    """tr(J_c^T J_c) of each candidate condition at the reference point.

    The stacked-Fisher trace is additive over conditions, so these
    traces fully determine the greedy trace-gain selection.
    """

    specs = _specs_for(parameter_names)
    names = tuple(spec.name for spec in specs)
    theta = np.asarray(theta_reference, dtype=float)
    if theta.shape != (len(specs),):
        raise ValueError("theta_reference must match the parameter set")
    traces: dict[tuple[float, float], float] = {}
    for temperature_k, soc in candidate_conditions:
        model = model_class(
            temperature_k=temperature_k, soc=soc, parameter_names=names
        )
        z_ref = model.simulate(np.asarray(freq_hz, dtype=float), theta)
        dataset = EISDataset(freq_hz=np.asarray(freq_hz, dtype=float), z_obs=z_ref)
        jacobian = relative_search_jacobian(dataset, model, specs, theta)
        traces[(float(temperature_k), float(soc))] = float(
            np.sum(jacobian * jacobian)
        )
    return traces


def select_informative_conditions(
    model_class: type,
    candidate_conditions: Sequence[tuple[float, float]],
    theta_reference: NDArray[np.floating],
    freq_hz: NDArray[np.floating],
    min_info_gain_ratio: float = 0.10,
    max_conditions: int = 9,
    parameter_names: Sequence[str] | None = None,
) -> list[tuple[float, float]]:
    """Greedy trace-gain condition selection.

    Conditions are added in decreasing order of their Fisher-trace
    contribution; the walk stops once the next condition would add less
    than ``min_info_gain_ratio`` of the accumulated trace, or at
    ``max_conditions``.
    """

    if not candidate_conditions:
        raise ValueError("candidate_conditions cannot be empty")
    if max_conditions < 1:
        raise ValueError("max_conditions must be >= 1")
    traces = condition_information_traces(
        model_class, candidate_conditions, theta_reference, freq_hz, parameter_names
    )
    ordered = sorted(traces.items(), key=lambda item: -item[1])
    selected = [ordered[0][0]]
    accumulated = ordered[0][1]
    for condition, trace in ordered[1:]:
        if len(selected) >= max_conditions:
            break
        if trace < min_info_gain_ratio * accumulated:
            break
        selected.append(condition)
        accumulated += trace
    return selected


# ---------------------------------------------------------------------------
# Posterior claim arithmetic (shared by grouper, subset search, and loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PosteriorWorkspace:
    """Cached whitened data FIM and prior geometry for fast subset math."""

    names: tuple[str, ...]
    data_fim: FloatArray          # J^T J / sigma_n^2 over all parameters
    prior_sigma: FloatArray       # search-space prior widths
    ci_scale: FloatArray          # sd_search -> relative CI95 factor
    prior_covariance: FloatArray | None = None

    def subset_metrics(self, indices: Sequence[int]) -> tuple[FloatArray, FloatArray]:
        idx = list(indices)
        covariance = (
            np.diag(self.prior_sigma[idx] ** 2)
            if self.prior_covariance is None
            else self.prior_covariance[np.ix_(idx, idx)]
        )
        fim = self.data_fim[np.ix_(idx, idx)] + np.linalg.inv(covariance)
        covariance = np.linalg.inv(fim)
        variance = np.maximum(np.diag(covariance), 0.0)
        prior_variance = (
            self.prior_sigma[idx] ** 2
            if self.prior_covariance is None
            else np.diag(self.prior_covariance)[idx]
        )
        gain = np.clip(1.0 - variance / prior_variance, 0.0, 1.0)
        ci95 = self.ci_scale[idx] * np.sqrt(variance)
        return gain, ci95

    def claim_mask(
        self, indices: Sequence[int], max_ci95: float, min_gain: float
    ) -> NDArray[np.bool_]:
        gain, ci95 = self.subset_metrics(indices)
        return (gain >= min_gain) & (ci95 <= max_ci95)

    def n_claims(self, indices: Sequence[int], max_ci95: float, min_gain: float) -> int:
        return int(np.count_nonzero(self.claim_mask(indices, max_ci95, min_gain)))


def _build_workspace(
    dataset: EISDataset,
    model: Any,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
    prior_set: PriorSet,
    noise_level: float,
    use_frequency_weighting: bool = False,
) -> _PosteriorWorkspace:
    jacobian = _data_only_relative_jacobian(
        dataset,
        model,
        specs,
        theta_reference,
        use_frequency_weighting=use_frequency_weighting,
    )
    sigma = prior_set.sigma_search_vector(specs)
    ci_scale = np.asarray(
        [
            1.96 * np.log(10.0)
            if spec.log_transform
            else 1.96 / max(abs(value), 1e-30)
            for spec, value in zip(specs, theta_reference, strict=True)
        ],
        dtype=float,
    )
    return _PosteriorWorkspace(
        names=tuple(spec.name for spec in specs),
        data_fim=jacobian.T @ jacobian / noise_level**2,
        prior_sigma=sigma,
        ci_scale=ci_scale,
        prior_covariance=prior_set.covariance_search_matrix(specs),
    )


def _data_only_relative_jacobian(
    dataset: EISDataset,
    model: Any,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
    *,
    use_frequency_weighting: bool,
) -> FloatArray:
    """Return the prior-free Jacobian used for candidate diagnostics."""

    jacobian = relative_search_jacobian(dataset, model, specs, theta_reference)
    if use_frequency_weighting:
        # Weight rows by 1/|Z_ref| so low-frequency (large |Z|) and
        # high-frequency (small |Z|) points contribute equally to the FIM
        # on a relative basis.  The Jacobian is already in relative
        # (search-space) coordinates; the weighting adjusts inter-frequency
        # balance only.
        z_ref = model.simulate(dataset.freq_hz, theta_reference)
        z_mag = np.maximum(np.abs(z_ref), 1e-30)
        # jacobian has 2N rows (N real + N imag); z_mag has N entries
        weights = np.concatenate([1.0 / z_mag, 1.0 / z_mag])
        weights *= len(weights) / np.sum(weights)  # normalise
        jacobian = jacobian * np.sqrt(weights)[:, None]
    return np.asarray(jacobian, dtype=float)


def _fit_objective_and_geometry_audit(
    *,
    dataset: EISDataset,
    model: Any,
    specs: Sequence[ParameterSpec],
    theta_fit: FloatArray,
    prior_set: PriorSet,
    noise_level: float,
    use_frequency_weighting: bool,
) -> dict[str, Any]:
    """Separate data fit, prior penalty, and prior-free local geometry."""

    spec_tuple = tuple(specs)
    theta = np.asarray(theta_fit, dtype=float)
    prediction = model.simulate(dataset.freq_hz, theta)
    relative_residual = (dataset.z_obs - prediction) / np.maximum(
        np.abs(dataset.z_obs), 1e-12
    )
    data_objective = float(
        np.sum(relative_residual.real**2 + relative_residual.imag**2)
    )

    search = encode_parameter_vector(spec_tuple, theta)
    means_physical = prior_set.mean_physical_vector(spec_tuple)
    means_search = encode_parameter_vector(spec_tuple, means_physical)
    sigma = prior_set.sigma_search_vector(spec_tuple)
    prior_precision = prior_set.precision_search_matrix(spec_tuple)
    prior_shift = search - means_search
    gaussian_prior_penalty = float(prior_shift @ prior_precision @ prior_shift)
    optimizer_prior_penalty = float(noise_level**2 * gaussian_prior_penalty)

    jacobian = _data_only_relative_jacobian(
        dataset,
        model,
        spec_tuple,
        theta,
        use_frequency_weighting=use_frequency_weighting,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else float("inf")
    )
    fim_data = jacobian.T @ jacobian / noise_level**2
    prior_curvature = prior_precision
    return {
        "data_objective": data_objective,
        "prior_penalty": {
            "gaussian_standardized": gaussian_prior_penalty,
            "optimizer_scale": optimizer_prior_penalty,
        },
        "total_objective": {
            "optimizer_scale": data_objective + optimizer_prior_penalty,
        },
        "prior_curvature": {
            "coordinate_space": "active_fit_search_coordinates",
            "matrix": prior_curvature.tolist(),
            "parameter_order": [spec.name for spec in spec_tuple],
        },
        "data_only_jacobian": {
            "coordinate_space": "active_fit_search_coordinates",
            "prior_rows_included": False,
            "frequency_weighting": bool(use_frequency_weighting),
            "shape": [int(v) for v in jacobian.shape],
            "sha256_float64_c_order": hashlib.sha256(
                np.ascontiguousarray(jacobian, dtype=np.float64).tobytes()
            ).hexdigest(),
            "frobenius_norm": float(np.linalg.norm(jacobian)),
            "rank": rank,
            "condition_number": condition,
            "data_fim_diagonal": {
                spec.name: float(value)
                for spec, value in zip(spec_tuple, np.diag(fim_data), strict=True)
            },
        },
    }


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------


#: Tier thresholds on the joint-posterior information gain.
TIER_GAIN_THRESHOLDS = {"A": 0.75, "B": 0.35, "C": 0.05}


class SEISParameterGrouper:
    """Empirical tier assignment for the 48 SEIS parameters.

    Runs the joint posterior over the full parameter set once and
    assigns tiers from the information gain: A (>= 0.75, directly
    EIS-visible), B (>= 0.35, transport-mediated), C (>= 0.05, weakly
    informative — typically only across a temperature range), D (< 0.05,
    not identifiable from EIS; never offered to the search).
    """

    def __init__(
        self,
        specs: Sequence[ParameterSpec],
        gate: TieredIdentifiabilityGate,
    ) -> None:
        self.specs = tuple(specs)
        self.gate = gate
        self.tiers: dict[str, str] = {}
        self.gains: dict[str, float] = {}

    def assign_tiers(
        self,
        dataset: EISDataset,
        model: Any,
        theta_reference: NDArray[np.floating],
        prior_set: PriorSet,
        use_frequency_weighting: bool = False,
    ) -> dict[str, str]:
        workspace = _build_workspace(
            dataset, model, self.specs,
            np.asarray(theta_reference, dtype=float), prior_set,
            self.gate.assumed_noise_level,
            use_frequency_weighting=use_frequency_weighting,
        )
        gain, _ = workspace.subset_metrics(range(len(self.specs)))
        self.gains = {
            spec.name: float(g) for spec, g in zip(self.specs, gain, strict=True)
        }
        self.tiers = {}
        for name, value in self.gains.items():
            if value >= TIER_GAIN_THRESHOLDS["A"]:
                self.tiers[name] = "A"
            elif value >= TIER_GAIN_THRESHOLDS["B"]:
                self.tiers[name] = "B"
            elif value >= TIER_GAIN_THRESHOLDS["C"]:
                self.tiers[name] = "C"
            else:
                self.tiers[name] = "D"
        return dict(self.tiers)

    def ordered_candidates(self) -> list[str]:
        """Unlocking order: tier A -> B -> C, by descending gain; D excluded."""

        if not self.tiers:
            raise RuntimeError("assign_tiers must be called first")
        order = []
        for tier in ("A", "B", "C"):
            members = [n for n, t in self.tiers.items() if t == tier]
            order.extend(sorted(members, key=lambda n: -self.gains[n]))
        return order


# ---------------------------------------------------------------------------
# Subset search (claim maximisation)
# ---------------------------------------------------------------------------


def _max_claim_subset(
    workspace: _PosteriorWorkspace,
    candidate_indices: Sequence[int],
    seed_indices: Sequence[int],
    required_indices: Sequence[int],
    max_ci95: float,
    min_gain: float,
    annealing_steps: int = 60000,
    rng_seed: int = 7,
    max_subset: int = 30,
    n_restarts: int = 5,
    score_cache_size: int = 8192,
) -> list[int]:
    """Deterministic annealing search for the maximum-claim subset.

    Objective: number of subset members passing the claim test, with a
    small continuous penalty on total CI to guide plateaus.  Sweeps in
    tier order, then anneals with ``n_restarts`` independent seeds,
    then prunes members whose removal does not reduce the claim count
    (fixing an uninformative parameter can only sharpen the rest —
    conditional variance is never larger than marginal), and finally
    greedily expands.  ``required_indices`` are always kept.
    """

    candidates = list(dict.fromkeys(candidate_indices))
    required = set(required_indices)

    def calculate_score(key: tuple[int, ...]) -> tuple[int, float]:
        gain, ci = workspace.subset_metrics(key)
        claims = int(np.count_nonzero((gain >= min_gain) & (ci <= max_ci95)))
        penalty_value = float(np.sum(np.minimum(ci, 1.0)))
        return claims, penalty_value

    if score_cache_size > 0:
        cached_score = lru_cache(maxsize=int(score_cache_size))(calculate_score)
    else:
        cached_score = calculate_score

    def score(subset: Sequence[int]) -> tuple[int, float]:
        return cached_score(tuple(int(index) for index in subset))

    def n_claims(subset: Sequence[int]) -> int:
        return score(subset)[0]

    # Greedy sweep in tier order.
    subset = list(dict.fromkeys(list(seed_indices)))
    best = n_claims(subset) if subset else 0
    for cand in candidates:
        if cand in subset:
            continue
        trial = subset + [cand]
        c = n_claims(trial)
        if c > best:
            subset, best = trial, c

    # Multi-restart deterministic annealing polish.
    pool = candidates
    global_best_c, global_best_set = best, list(subset)
    for restart in range(n_restarts):
        rng = np.random.default_rng(rng_seed + restart * 1337)
        current = set(subset)
        cur_c, cur_pen = score(sorted(current))
        best_c, best_set = cur_c, sorted(current)
        for step in range(annealing_steps):
            temperature = max(0.02, float(np.exp(-step / max(annealing_steps / 4, 1))))
            trial = set(current)
            move = rng.random()
            if move < 0.45 and len(trial) < max_subset:
                trial.add(int(pool[rng.integers(len(pool))]))
            elif move < 0.9 and len(trial) > max(1, len(required)):
                removable = [i for i in trial if i not in required]
                if removable:
                    trial.discard(int(rng.choice(removable)))
            elif len(trial) > max(1, len(required)):
                removable = [i for i in trial if i not in required]
                if removable:
                    trial.discard(int(rng.choice(removable)))
                    trial.add(int(pool[rng.integers(len(pool))]))
            if not trial or trial == current:
                continue
            c, pen = score(sorted(trial))
            delta = (c - cur_c) - 0.001 * (pen - cur_pen)
            if delta > 0 or rng.random() < np.exp(delta / temperature):
                current, cur_c, cur_pen = trial, c, pen
                if c > best_c:
                    best_c, best_set = c, sorted(trial)
        if best_c > global_best_c:
            global_best_c, global_best_set = best_c, best_set

    # Prune: drop non-required members whose removal keeps the claim count.
    pruned = list(global_best_set)
    changed = True
    while changed:
        changed = False
        for member in list(pruned):
            if member in required or len(pruned) <= 1:
                continue
            trial = [i for i in pruned if i != member]
            if n_claims(trial) >= n_claims(pruned):
                pruned = trial
                changed = True

    # Greedy expansion: try adding any candidate that increases claims.
    expanded = True
    while expanded:
        expanded = False
        base_c = n_claims(pruned)
        for cand in candidates:
            if cand in pruned or len(pruned) >= max_subset:
                continue
            trial = sorted(set(pruned) | {cand})
            if n_claims(trial) > base_c:
                pruned = trial
                expanded = True
                break

    return sorted(set(pruned) | required)


# ---------------------------------------------------------------------------
# Staged (block-coordinate) MAP fitting — Strategy 4
# ---------------------------------------------------------------------------


def _map_optimizer(n_starts: int, max_nfev: int) -> AdaptiveLeastSquaresOptimizer:
    return AdaptiveLeastSquaresOptimizer(
        relative=True,
        n_starts=n_starts,
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )


def _bcd_map_fit(
    dataset: EISDataset,
    sub_model: Any,
    sub_specs: Sequence[ParameterSpec],
    prior_set: PriorSet,
    noise_level: float,
    initial_guess: dict[str, float],
    tier_of: Mapping[str, str],
    use_bcd: bool,
    n_starts: int,
    max_nfev: int,
):
    """MAP fit with optional tiered block-coordinate initialization.

    Phase 1 optimizes the tier-A (interfacial) block with everything
    else frozen at the starting point; phase 2 optimizes the remaining
    block with tier A frozen at the phase-1 result; phase 3 is the full
    joint fit warm-started from the staged point.  For small subsets the
    staging is skipped — the joint problem is well behaved.
    """

    names = [spec.name for spec in sub_specs]
    guess = dict(initial_guess)
    optimizer = _map_optimizer(n_starts, max_nfev)

    tier_a = [n for n in names if tier_of.get(n) == "A"]
    if use_bcd and len(names) > 6 and 0 < len(tier_a) < len(names):
        stage_optimizer = _map_optimizer(1, max_nfev)
        for block in (tier_a, [n for n in names if n not in tier_a]):
            frozen = {n: guess[n] for n in names if n not in block}
            staged = SubsetModel(
                full_model=sub_model,
                full_specs=tuple(sub_specs),
                fixed_values=frozen,
            )
            block_specs = list(staged.parameter_specs)
            reg_matrix, reg_reference = prior_set.regularization_geometry_for_map(
                block_specs, noise_level
            )
            stage_fit = stage_optimizer.fit(
                dataset=dataset,
                model=staged,
                parameter_specs=block_specs,
                initial_guess={n: guess[n] for n in staged.parameter_names},
                regularization_matrix=reg_matrix,
                regularization_matrix_reference=reg_reference,
            )
            guess.update({k: float(v) for k, v in stage_fit.theta_best.items()})

    reg_matrix, reg_reference = prior_set.regularization_geometry_for_map(
        sub_specs, noise_level
    )
    return optimizer.fit(
        dataset=dataset,
        model=sub_model,
        parameter_specs=sub_specs,
        initial_guess=guess,
        regularization_matrix=reg_matrix,
        regularization_matrix_reference=reg_reference,
    )


# ---------------------------------------------------------------------------
# Profile likelihood rescue — Strategy 3
# ---------------------------------------------------------------------------


def _profile_rescue(
    dataset: EISDataset,
    sub_model: Any,
    sub_specs: Sequence[ParameterSpec],
    prior_set: PriorSet,
    noise_level: float,
    map_fit,
    selection: TieredSelection,
    quantile: float,
    max_rescue_params: int,
    max_nfev: int,
    min_gain: float = 0.35,
    ci_window: tuple[float, float] = (0.20, 0.80),
) -> tuple[TieredSelection, dict[str, Any]]:
    """Upgrade bubble parameters whose profile likelihood rejects deviations.

    The Wald CI is a linearization; for a nonlinear model it can
    overestimate the true uncertainty.  For every attempted parameter
    with information gain >= ``min_gain`` and Wald CI in ``ci_window``,
    the parameter is fixed at shifted values, the remaining subset
    is re-optimized, and the likelihood-ratio statistic
    ``dCost / noise^2`` compared to ``chi2(quantile, df=1)``.  Uses
    multi-level shifts: first ±10%, then ±20% for more robust rescue.
    Only when the data reject *both* deviations is the parameter upgraded
    to PRIOR_INFORMED (the strict tier is never touched).  A confirmation
    that the Wald width was accurate (no upgrade) is recorded too.

    Two safeguards keep the rescue honest:

    * the LR statistic subtracts the *fixed parameter's own prior*
      contribution, so a tight prior can never reject the deviation by
      itself (a pure-prior rejection is exactly the false
      identifiability this gate exists to prevent);
    * an upgraded parameter gets the profile-implied posterior CI
      (the rejected shift level) and the corresponding information
      gain, so the reported taxonomy stays internally consistent.
    """

    from dataclasses import replace as dc_replace

    from scipy.stats import chi2

    names = [spec.name for spec in sub_specs]
    candidates = [
        name
        for name in names
        if selection.tiers[name]
        in (ParameterTier.PRIOR_DOMINATED, ParameterTier.FIXED)
        and selection.information_gain[name] >= min_gain
        and ci_window[0] < selection.posterior_ci95[name] <= ci_window[1]
    ]
    candidates.sort(key=lambda n: selection.posterior_ci95[n])
    candidates = candidates[:max_rescue_params]
    log: dict[str, Any] = {"threshold_lr": float(chi2.ppf(quantile, df=1))}
    if not candidates:
        log["candidates"] = []
        return selection, log

    threshold = float(chi2.ppf(quantile, df=1)) * noise_level**2
    optimizer = _map_optimizer(1, max_nfev)
    spec_by_name = {spec.name: spec for spec in sub_specs}
    new_tiers = dict(selection.tiers)
    new_ci = dict(selection.posterior_ci95)
    new_gain = dict(selection.information_gain)
    entries = []

    def own_prior_penalty(name: str, physical_value: float) -> float:
        """This parameter's own prior residual^2 at a physical value."""

        prior = prior_set.get(name)
        spec = spec_by_name[name]
        if prior is None:
            return 0.0
        s = spec.to_optimization(float(physical_value))
        mu = spec.to_optimization(
            min(max(prior.mean_physical, spec.bounds[0]), spec.bounds[1])
        )
        return float((noise_level / prior.sigma_search) ** 2 * (s - mu) ** 2)

    for name in candidates:
        map_value = float(map_fit.theta_best[name])
        rescued = False
        rescued_shift = None
        best_lr_stats = [0.0, 0.0]
        # Multi-level shifts: try ±10% first (gentler), then ±20%.
        for shift_pct in (0.10, 0.20):
            deltas = []
            for factor in (1.0 + shift_pct, 1.0 / (1.0 + shift_pct)):
                fixed_value = map_value * factor
                profiled = SubsetModel(
                    full_model=sub_model,
                    full_specs=tuple(sub_specs),
                    fixed_values={name: fixed_value},
                )
                profile_specs = list(profiled.parameter_specs)
                reg_matrix, reg_reference = prior_set.regularization_geometry_for_map(
                    profile_specs, noise_level
                )
                profile_fit = optimizer.fit(
                    dataset=dataset,
                    model=profiled,
                    parameter_specs=profile_specs,
                    initial_guess={
                        k: float(v) for k, v in map_fit.theta_best.items() if k != name
                    },
                    regularization_matrix=reg_matrix,
                    regularization_matrix_reference=reg_reference,
                )
                # Subtract the fixed parameter's own prior contribution:
                # the deviation must be rejected by the DATA (plus the
                # other parameters' priors), never by its own prior.
                prior_shift = own_prior_penalty(name, fixed_value) - own_prior_penalty(
                    name, map_value
                )
                deltas.append(
                    float(profile_fit.final_cost - map_fit.final_cost) - prior_shift
                )
            best_lr_stats = [max(d, 0.0) / noise_level**2 for d in deltas]
            if min(deltas) > threshold:
                rescued = True
                rescued_shift = shift_pct
                break  # Gentler shift already rejects — no need for ±20%
        implied_gain = None
        if rescued:
            # Profile-implied posterior: the 95% interval is inside the
            # rejected +-shift, so CI95 <= shift; the implied gain follows
            # from the corresponding search-space standard deviation.
            spec = spec_by_name[name]
            sd_scale = (
                1.96 * np.log(10.0)
                if spec.log_transform
                else 1.96 / max(abs(map_value), 1e-30)
            )
            implied_sd = rescued_shift / sd_scale
            prior = prior_set.get(name)
            implied_gain = (
                float(np.clip(1.0 - (implied_sd / prior.sigma_search) ** 2, 0.0, 1.0))
                if prior is not None
                else 1.0
            )
            # The claim standard still applies: the deviation must be
            # rejected by information the DATA contributed, not because
            # the profile interval barely undercuts a similar prior.
            if implied_gain < 0.5:
                logger.info(
                    "profile rescue %s: deviation rejected but implied gain "
                    "%.2f < 0.5 — not upgraded", name, implied_gain,
                )
                rescued = False
        if rescued:
            new_tiers[name] = ParameterTier.PRIOR_INFORMED
            new_ci[name] = float(rescued_shift)
            new_gain[name] = implied_gain
        entries.append(
            {
                "parameter": name,
                "wald_ci95": float(selection.posterior_ci95[name]),
                "lr_stat_up": best_lr_stats[0],
                "lr_stat_down": best_lr_stats[1],
                "rescued": bool(rescued),
                "rescued_shift": rescued_shift if rescued else None,
                "implied_gain": implied_gain,
            }
        )
        logger.info(
            "profile rescue %s: LR(+)=%.2f LR(-)=%.2f -> %s",
            name, best_lr_stats[0], best_lr_stats[1],
            "UPGRADED" if rescued else "Wald width confirmed",
        )
    log["candidates"] = entries
    if any(e["rescued"] for e in entries):
        selection = dc_replace(
            selection,
            tiers=new_tiers,
            posterior_ci95=new_ci,
            information_gain=new_gain,
        )
    return selection, log


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEISIdentificationResult:
    """Outcome of the Generation-2 SEIS identification loop."""

    theta_best: dict[str, float]
    tiered_selection: TieredSelection
    identifiable_params: list[str]
    n_identifiable: int
    prior_informed_params: list[str]
    conditions_used: list[tuple[float, float]]
    condition_info_gains: dict[str, float]
    parameter_sensitivity_ranking: list[str]
    attempted_params: list[str]
    tier_assignment: dict[str, str]
    noise_level: float
    final_cost: float
    #: Prior-robust safe claims from an independent marginal-CI and
    #: nuisance-bias audit.  This list is reported separately from the
    #: publication headline ``identifiable_params`` (STRICT only), because
    #: the two gates answer different statistical questions.
    safe_claim_params: list[str] = field(default_factory=list)
    n_safe_claims: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """One audit row per parameter in the full 48-parameter space."""

        selection = self.tiered_selection
        rows = []
        for name in selection.parameter_names:
            tier = selection.tiers[name]
            rows.append(
                {
                    "parameter": name,
                    "sensitivity_tier": self.tier_assignment.get(name, "-"),
                    "attempted": name in self.attempted_params,
                    "gate_tier": tier.value,
                    "identifiable": name in self.identifiable_params,
                    "data_supported": name in self.identifiable_params,
                    "prior_informed": name in self.prior_informed_params,
                    "fitted_value": self.theta_best.get(name, np.nan),
                    "information_gain": selection.information_gain[name],
                    "posterior_ci95_relative": selection.posterior_ci95[name],
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _required_subset_indices(
    parameter_names: Sequence[str],
    seed_names: Sequence[str],
    *,
    explicit_candidates: bool,
    protected_names: Sequence[str],
) -> list[int]:
    """Resolve parameters that the V2 subset search is not allowed to drop."""

    names = tuple(parameter_names)
    name_to_index = {name: i for i, name in enumerate(names)}
    unknown_seed = set(seed_names) - set(names)
    if unknown_seed:
        raise ValueError(f"unknown SEIS parameters: {sorted(unknown_seed)}")
    unknown_protected = set(protected_names) - set(names)
    if unknown_protected:
        raise ValueError(
            f"unknown protected SEIS parameters: {sorted(unknown_protected)}"
        )
    required_names = list(seed_names) if explicit_candidates else []
    required_names.extend(protected_names)
    return [name_to_index[name] for name in dict.fromkeys(required_names)]


def identify_seis_parameters(
    datasets: list[EISDataset],
    conditions: list[tuple[float, float]],
    freq_hz: NDArray[np.floating],
    noise_level: float | None = None,
    candidate_params: list[str] | None = None,
    max_conditions: int = 25,
    min_info_gain_ratio: float = 0.02,
    max_iter: int = 5,
    gate: TieredIdentifiabilityGate | None = None,
    prior_builder: SEISPhysicsPriorBuilder | None = None,
    n_starts: int = 1,
    max_nfev: int = 4000,
    annealing_steps: int = 60000,
    annealing_seed: int = 7,
    annealing_restarts: int = 5,
    use_arrhenius_reparameterization: bool = True,
    use_physics_reparameterization: bool = True,
    use_frequency_weighting: bool = True,
    use_prior_tightening: bool = False,
    use_bcd: bool = True,
    profile_rescue: bool = False,
    max_rescue_params: int = 10,
    profile_rescue_quantile: float = 0.95,
    certify_profile: bool = False,
    profile_certification_mode: str = "posterior",
    demote_profile_overclaims: bool = True,
    nuisance_policy: NuisancePolicy | str = NuisancePolicy.LEGACY_HARD_FIXED,
    certify_safe_claims: bool = False,
    safe_claim_max_ci95: float = 0.20,
    safe_claim_max_bias_fraction: float = 1.0,
    safe_claim_nuisance_prior: bool = True,
    safe_claim_refit_union: bool = False,
    export_safe_fim: bool = False,
    compute_backend: str = "numpy",
) -> SEISIdentificationResult:
    """Identify as many SEIS physical parameters as the data support.

    Implements the iterative unlocking loop: stack the most informative
    conditions, MAP-fit the attempted subset with literature priors,
    re-derive the posterior workspace at the fitted point, grow the
    attempted subset by claim-maximising search over the tier-ordered
    candidates, and add conditions while their trace gain exceeds
    ``min_info_gain_ratio``.  The final tiers come from one full
    :class:`TieredIdentifiabilityGate` evaluation on the attempted
    subset (fixed parameters disclosed as FIXED at literature values).

    Coordinate and optimization controls:

    * ``use_arrhenius_reparameterization`` — re-centre every Arrhenius
      pair at its information-centroid temperature, zeroing the pairwise
      Jacobian correlation (identifiability claims for the base
      coefficients then refer to the rate at T_c; see
      ``metadata['reparameterized_coordinates']``, and ``theta_best`` is
      always reported back in original k_0 coordinates).
    * ``use_bcd`` — tiered block-coordinate initialization of every MAP
      fit (interfacial block first, transport second, joint third).
    * ``profile_rescue`` — disabled by publication default. When explicitly
      enabled for a legacy control, borderline parameters (gain >= 0.45,
      Wald CI in 20-35 %), test +-20 % deviations by profile likelihood
      and upgrade to PRIOR_INFORMED only when the data reject both at
      the ``profile_rescue_quantile`` level.  When the profile confirms
      the Wald width instead, that confirmation is recorded in
      ``metadata['profile_rescue']``.
    * ``use_prior_tightening`` — deprecated compatibility input. A true value
      warns and is ignored; the removed empirical-Bayes stage cannot run.
    """

    freq = np.asarray(freq_hz, dtype=float)
    if compute_backend not in {"numpy", "jax", "jax-cpu", "jax-gpu"}:
        raise ValueError(
            "compute_backend must be one of 'numpy', 'jax', 'jax-cpu', "
            "or 'jax-gpu'"
        )
    if len(datasets) != len(conditions):
        raise ValueError("datasets and conditions must have equal length")
    if not datasets:
        raise ValueError("at least one dataset is required")
    for dataset in datasets:
        if dataset.freq_hz.shape != freq.shape or not np.allclose(
            dataset.freq_hz, freq
        ):
            raise ValueError("every dataset must share the provided freq_hz grid")
    if not isinstance(annealing_seed, (int, np.integer)):
        raise ValueError("annealing_seed must be an integer")
    if not isinstance(annealing_restarts, (int, np.integer)) or annealing_restarts < 1:
        raise ValueError("annealing_restarts must be an integer >= 1")

    policy = parse_nuisance_policy(nuisance_policy)
    specs = tuple(all_seis_parameter_specs())
    raw_specs = specs
    names = tuple(spec.name for spec in specs)
    name_to_index = {name: i for i, name in enumerate(names)}
    prior_builder = prior_builder or SEISPhysicsPriorBuilder()
    prior_set = prior_builder.build(specs)
    theta_initial_raw = prior_set.mean_physical_vector(specs).copy()

    if policy is NuisancePolicy.ALL48_WEAK_PRIOR and candidate_params is not None:
        if set(candidate_params) != set(names) or len(candidate_params) != len(names):
            raise ValueError(
                "candidate_params must be omitted or contain every SEIS parameter "
                "when nuisance_policy='all48_weak_prior'"
            )

    if noise_level is None:
        from .pipeline import estimate_noise_level

        noise_level = float(
            np.median([estimate_noise_level(dataset) for dataset in datasets])
        )
    if not np.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("noise_level must be finite and positive")

    gate = gate or TieredIdentifiabilityGate(assumed_noise_level=noise_level)
    if not np.isclose(gate.assumed_noise_level, noise_level, rtol=0.25):
        logger.warning(
            "gate.assumed_noise_level (%.4g) disagrees with noise_level (%.4g)",
            gate.assumed_noise_level,
            noise_level,
        )

    theta_full = prior_set.mean_physical_vector(specs)
    condition_pool = [(float(t), float(s)) for t, s in conditions]

    # --- Condition selection (greedy trace gain) ---
    traces = condition_information_traces(
        SEISModel, condition_pool, theta_full, freq
    )
    ordered_conditions = sorted(traces, key=lambda c: -traces[c])
    selected_conditions = [ordered_conditions[0]]
    accumulated = traces[ordered_conditions[0]]
    for condition in ordered_conditions[1:]:
        if len(selected_conditions) >= max_conditions:
            break
        if traces[condition] < min_info_gain_ratio * accumulated:
            break
        selected_conditions.append(condition)
        accumulated += traces[condition]

    reparameterizer: ArrheniusReparameterizer | None = None
    physics_reparam: PhysicsProductReparameterizer | None = None

    def wrap(model: Any) -> Any:
        # Chain order matters: the optimizer/gate speak PHYSICS coordinates,
        # which the outer wrapper converts to Arrhenius coordinates, which
        # the inner wrapper converts to the raw Gen-1 parameters.
        if reparameterizer is not None:
            model = reparameterizer.wrap_model(model)
        if physics_reparam is not None:
            model = physics_reparam.wrap_model(model)
        return model

    def stacked_problem(active: list[tuple[float, float]]):
        order = [condition_pool.index(c) for c in active]
        stacked_freq = np.tile(freq, len(active))
        z = np.concatenate([datasets[i].z_obs for i in order])
        context = stacked_seis_condition_context(freq, tuple(active))
        dataset = EISDataset(freq_hz=stacked_freq, z_obs=z, context=context)
        model = StackedSEISModel(
            conditions=tuple(active),
            parameter_names=names,
            compute_backend=compute_backend,
        )
        return dataset, wrap(model)

    # --- Strategy 1+2: Arrhenius recentring at the information centroid ---
    coordinate_notes: dict[str, str] = {}
    if use_arrhenius_reparameterization:
        ws_dataset, ws_model = stacked_problem(selected_conditions)  # raw
        workspace0 = _build_workspace(
            ws_dataset, ws_model, specs, theta_full, prior_set, noise_level,
            use_frequency_weighting=use_frequency_weighting,
        )
        t_c = optimal_reference_temperatures(
            workspace0.data_fim, specs, theta_full
        )
        reparameterizer = ArrheniusReparameterizer(reference_temperatures=t_c)
        new_specs, theta_full = reparameterizer.reparameterize(specs, theta_full)
        prior_set = reparameterizer.transform_prior_set(prior_set, specs)
        specs = tuple(new_specs)
        coordinate_notes = reparameterizer.coordinate_descriptions(names)

    # --- Strategy A: physics-product coordinates (tau_d, R_sei, C_sei, ...) ---
    physics_notes: dict[str, str] = {}
    if use_physics_reparameterization:
        physics_reparam = PhysicsProductReparameterizer()
        new_specs, theta_full = physics_reparam.reparameterize(specs, theta_full)
        prior_set = physics_reparam.transform_prior_set(prior_set, specs)
        specs = tuple(new_specs)
        physics_notes = physics_reparam.coordinate_descriptions(names)

    theta_initial_fit = theta_full.copy()

    # --- Tier assignment on the fully stacked problem ---
    grouper = SEISParameterGrouper(specs, gate)
    tier_dataset, tier_model = stacked_problem(selected_conditions)
    grouper.assign_tiers(
        tier_dataset, tier_model, theta_full, prior_set,
        use_frequency_weighting=use_frequency_weighting,
    )
    candidate_order = grouper.ordered_candidates()

    if policy is NuisancePolicy.ALL48_WEAK_PRIOR:
        seed_names = list(names)
    else:
        seed_names = (
            list(candidate_params) if candidate_params else list(DEFAULT_TIER_A_SEED)
        )
    # The strict gate contract says protected parameters remain free.  The V2
    # annealing search previously treated them as removable unless they also
    # happened to survive claim maximisation, which could then pass a subset
    # lacking one of the protected names back into the Gen-1 selector and
    # raise ``protected parameters must occur in parameter_specs``.  Make the
    # protection policy an explicit subset-search constraint.
    required = (
        list(range(len(names)))
        if policy is NuisancePolicy.ALL48_WEAK_PRIOR
        else _required_subset_indices(
            names,
            seed_names,
            explicit_candidates=candidate_params is not None,
            protected_names=gate.protected_names,
        )
    )
    seed_idx = [name_to_index[n] for n in seed_names]
    candidate_idx = list(
        dict.fromkeys(seed_idx + [name_to_index[n] for n in candidate_order])
    )

    attempted: list[int] = list(seed_idx)
    fit_result = None
    history: list[dict[str, Any]] = []

    def subset_model_for(attempted_names: tuple[str, ...]) -> SubsetModel:
        """Reduced model that freezes non-attempted parameters in the FINAL
        coordinate system.

        Freezing must happen in the same coordinates the gate and the
        workspace reason in: with the physics transforms active, fixing
        the diffusion coordinate tau_d = rs^2/Ds lets Ds co-vary with rs,
        whereas freezing raw Ds (what a bare ``StackedSEISModel`` subset
        would do) is a different conditioning and demonstrably loses
        confirmed claims.
        """

        full_model = wrap(
            StackedSEISModel(
                conditions=tuple(selected_conditions),
                parameter_names=names,
                compute_backend=compute_backend,
            )
        )
        fixed = {
            name: float(theta_full[name_to_index[name]])
            for name in names
            if name not in attempted_names
        }
        return SubsetModel(
            full_model=full_model, full_specs=tuple(specs), fixed_values=fixed
        )

    def fit_attempted(indices: Sequence[int], dataset: EISDataset):
        """MAP fit (with BCD staging) of an attempted subset; updates theta."""

        attempted_names = tuple(names[i] for i in sorted(indices))
        sub_specs = [specs[name_to_index[n]] for n in attempted_names]
        sub_model = subset_model_for(attempted_names)
        fit = _bcd_map_fit(
            dataset=dataset,
            sub_model=sub_model,
            sub_specs=sub_specs,
            prior_set=prior_set,
            noise_level=noise_level,
            initial_guess={
                n: float(theta_full[name_to_index[n]]) for n in attempted_names
            },
            tier_of=grouper.tiers,
            use_bcd=use_bcd,
            n_starts=n_starts,
            max_nfev=max_nfev,
        )
        for n, v in fit.theta_best.items():
            theta_full[name_to_index[n]] = float(v)
        return fit, sub_model, sub_specs

    for iteration in range(max_iter):
        dataset, model = stacked_problem(selected_conditions)
        workspace = _build_workspace(
            dataset, model, specs, theta_full, prior_set, noise_level,
            use_frequency_weighting=use_frequency_weighting,
        )

        if policy is NuisancePolicy.ALL48_WEAK_PRIOR:
            new_attempted = list(range(len(names)))
        else:
            new_attempted = _max_claim_subset(
                workspace,
                candidate_idx,
                seed_indices=attempted,
                required_indices=required,
                max_ci95=gate.max_posterior_ci95,
                min_gain=gate.min_information_gain,
                annealing_steps=annealing_steps,
                rng_seed=int(annealing_seed),
                n_restarts=int(annealing_restarts),
            )

        fit_result, _, _ = fit_attempted(new_attempted, dataset)

        history.append(
            {
                "iteration": iteration,
                "n_conditions": len(selected_conditions),
                "n_attempted": len(new_attempted),
                "fit_cost": float(fit_result.final_cost),
            }
        )

        stable = sorted(new_attempted) == sorted(attempted)
        attempted = sorted(new_attempted)

        # Condition growth (post-fit reference point).
        added_condition = False
        if len(selected_conditions) < min(max_conditions, len(condition_pool)):
            remaining = [c for c in ordered_conditions if c not in selected_conditions]
            for condition in remaining:
                if traces[condition] >= min_info_gain_ratio * accumulated:
                    selected_conditions.append(condition)
                    accumulated += traces[condition]
                    added_condition = True
                    break

        if stable and not added_condition:
            break

    # Adaptive empirical-Bayes prior tightening was removed from the
    # publication candidate. The legacy switch is parsed only to warn and is
    # deliberately ignored, so it cannot move a fit or promote any label.
    tightening_status = resolve_legacy_prior_tightening(use_prior_tightening)

    # --- Final gate evaluation, bubble retention, and profile rescue ---
    dataset, model48 = stacked_problem(selected_conditions)
    rescue_log: dict[str, Any] = {}

    def gate_on(indices: Sequence[int]):
        subset_names = tuple(names[i] for i in sorted(indices))
        subset_specs = [specs[name_to_index[n]] for n in subset_names]
        subset_model = subset_model_for(subset_names)
        theta_subset = np.asarray(
            [theta_full[name_to_index[n]] for n in subset_names], dtype=float
        )
        return (
            gate.evaluate(dataset, subset_model, subset_specs, theta_subset, prior_set),
            subset_model,
            subset_specs,
        )

    def n_confirmed(sel: TieredSelection) -> int:
        return sum(
            1
            for tier in sel.tiers.values()
            if tier in (ParameterTier.STRICT, ParameterTier.PRIOR_INFORMED)
        )

    selection, final_model, final_specs = gate_on(attempted)

    if profile_rescue:
        # Bubble retention: re-admit candidates whose conditional Wald CI
        # lands in the rescue window with solid information gain, so the
        # profile stage can examine them.  Guarded by a claim-count
        # rollback: joint CIs after the extended refit can differ from
        # the pre-fit snapshot, and confirmed claims are never traded
        # for rescue *candidates*.
        core_theta = theta_full.copy()
        core_attempted = list(attempted)
        core_fit = fit_result
        core_state = (selection, final_model, final_specs)

        workspace = _build_workspace(
            dataset, model48, specs, theta_full, prior_set, noise_level,
            use_frequency_weighting=use_frequency_weighting,
        )
        current = sorted(attempted)
        base_claims = workspace.n_claims(
            current, gate.max_posterior_ci95, gate.min_information_gain
        )
        added = 0
        for cand in candidate_idx:
            if cand in current or added >= max_rescue_params:
                continue
            trial = current + [cand]
            gain_t, ci_t = workspace.subset_metrics(trial)
            if not (gain_t[-1] >= 0.35 and 0.20 < ci_t[-1] <= 0.80):
                continue
            if (
                workspace.n_claims(
                    trial, gate.max_posterior_ci95, gate.min_information_gain
                )
                >= base_claims
            ):
                current = sorted(trial)
                added += 1
        if added:
            fit_result, _, _ = fit_attempted(current, dataset)
            attempted = current
            selection, final_model, final_specs = gate_on(attempted)
            if n_confirmed(selection) < n_confirmed(core_state[0]):
                logger.info(
                    "bubble retention rolled back: %d confirmed claims with "
                    "%d bubble members vs %d without",
                    n_confirmed(selection), added, n_confirmed(core_state[0]),
                )
                theta_full[:] = core_theta
                attempted = core_attempted
                fit_result = core_fit
                selection, final_model, final_specs = core_state
                rescue_log["bubble_rolled_back"] = True

        if fit_result is not None:
            selection, rescue_entries = _profile_rescue(
                dataset=dataset,
                sub_model=final_model,
                sub_specs=final_specs,
                prior_set=prior_set,
                noise_level=noise_level,
                map_fit=fit_result,
                selection=selection,
                quantile=profile_rescue_quantile,
                max_rescue_params=max_rescue_params,
                max_nfev=max_nfev,
            )
            rescue_log.update(rescue_entries)

    # --- Optional: exact profile-likelihood certification ---
    # Wald is a local approximation; on this problem it is measurably
    # optimistic (the profile interval is wider), so certification can
    # only confirm or *demote* claims, never inflate them.  Off by
    # default (it costs many nonlinear refits); when on, PRIOR_INFORMED
    # claims whose exact profile CI exceeds the bar are demoted to FIXED,
    # hardening the zero-false-identifiability guarantee.
    certification_log: dict[str, Any] = {}
    if certify_profile and fit_result is not None:
        from .profile_likelihood import ProfileLikelihoodCertifier

        certifier = ProfileLikelihoodCertifier(
            quantile=profile_rescue_quantile,
            max_relative_ci95=gate.max_posterior_ci95,
            mode=profile_certification_mode,
            max_nfev=max_nfev,
        )
        certification = certifier.certify(
            dataset=dataset,
            model=final_model,
            specs=final_specs,
            prior_set=prior_set,
            map_theta=fit_result.theta_best,
            map_cost=float(fit_result.final_cost),
            noise_level=noise_level,
            selection=selection,
        )
        certification_log = {
            "mode": certification.mode,
            "n_certified": certification.n_certified,
            "overclaimed": list(certification.overclaimed),
            "rows": certification.to_rows(),
        }
        if demote_profile_overclaims:
            selection = certifier.downgrade(selection, certification)

    attempted_names = tuple(names[i] for i in attempted)

    # Expand the subset selection to a full-48 report: parameters never
    # attempted are FIXED at their literature reference by construction.
    full_selection = _expand_selection(selection, specs, theta_full)

    strict_claims = [
        name
        for name in full_selection.parameter_names
        if full_selection.tiers[name] is ParameterTier.STRICT
    ]
    prior_informed_claims = [
        name
        for name in full_selection.parameter_names
        if full_selection.tiers[name] is ParameterTier.PRIOR_INFORMED
    ]
    ranking = sorted(
        full_selection.parameter_names,
        key=lambda n: -full_selection.information_gain[n],
    )

    # --- Prior-robust safe-claim certification ---
    # Safe claims are an optional, independent robustness audit.  The
    # publication headline above remains the calibrated STRICT list from
    # the C-target branch; enabling this layer must not change that list.
    # The certifier applies a marginal-CI test (no prior on claimed params,
    # nuisances marginalized with their uncertainty) and a bias-robustness
    # test for nuisance/prior misspecification.
    safe_claim_params: list[str] = []
    safe_claim_log: dict[str, Any] = {}
    if certify_safe_claims:
        from .safe_claim import SafeClaimCertifier

        certifier = SafeClaimCertifier(
            max_marginal_ci95=safe_claim_max_ci95,
            max_bias_fraction=safe_claim_max_bias_fraction,
            nuisance_prior=safe_claim_nuisance_prior,
        )

        # Pass 1: search the safe candidate block over the whole tier-ordered
        # pool (at the current fitted point).  The optimistic attempted set
        # was chosen to maximize prior-supported claims, not prior-robust
        # ones, so the safe candidates need not be a subset of it.
        pass1_ws = _build_workspace(
            dataset, model48, specs, theta_full, prior_set, noise_level
        )
        pass1 = certifier.search(
            names, pass1_ws.data_fim, pass1_ws.prior_sigma, pass1_ws.ci_scale,
            candidate_indices=candidate_idx,
        )

        # Fit the union of the attempted set and the safe candidate block so
        # every certified-safe parameter carries a genuine data-driven value
        # (never a literature default that would re-inject the prior==truth
        # artifact).  Done on an isolated theta snapshot so the main
        # recovery report (identifiable / theta_best) stays exactly as
        # gated above; the safe report is purely additive.
        theta_snapshot = theta_full.copy()
        if safe_claim_refit_union:
            # Fit the union so more safe candidates carry data-driven values
            # (raises the safe count ~6->9 on the 5x5 grid) at the cost of a
            # second heavy BCD fit.  Off by default: the marginal gain in
            # safe count is small next to the ~2x runtime, and the residual
            # false-claim rate is fit-robustness limited, not count limited.
            union = sorted(
                set(attempted) | {name_to_index[n] for n in pass1.claimed_block}
            )
            if union and union != sorted(attempted):
                fit_attempted(union, dataset)  # mutates theta_full (restored below)
            fitted = set(union) if union else set(attempted)
        else:
            fitted = set(attempted)

        # Pass 2: certify at the refitted point, eligible = fitted params only.
        pass2_ws = _build_workspace(
            dataset, model48, specs, theta_full, prior_set, noise_level
        )
        report = certifier.search(
            names, pass2_ws.data_fim, pass2_ws.prior_sigma, pass2_ws.ci_scale,
            candidate_indices=sorted(fitted),
        )
        # Record the data-driven fitted values (original coordinates) for the
        # certified-safe parameters before restoring the main theta.
        theta_phys = (
            physics_reparam.to_original(names, theta_full)
            if physics_reparam is not None else theta_full
        )
        theta_orig = (
            reparameterizer.to_original(names, theta_phys)
            if reparameterizer is not None else theta_phys
        )
        safe_fitted_values = {
            n: float(theta_orig[name_to_index[n]]) for n in report.safe_params
        }
        theta_full[:] = theta_snapshot  # keep the main result consistent

        safe_claim_params = list(report.safe_params)
        safe_claim_log = {
            "n_safe": report.n_safe,
            "fitted_values": safe_fitted_values,
            "claimed_block": list(report.claimed_block),
            "marginal_ci95": report.marginal_ci95,
            "bias_ci95": report.bias_ci95,
            "max_marginal_ci95": report.max_marginal_ci95,
            "max_bias_fraction": report.max_bias_fraction,
            "nuisance_prior": safe_claim_nuisance_prior,
        }
        if export_safe_fim:
            # Full 48x48 data FIM and geometry at the (union-refit) point,
            # plus data-driven values for every fitted param, so any
            # certification tier / threshold can be evaluated post-hoc.
            union_names = [names[i] for i in sorted(fitted)]
            safe_claim_log["fim_export"] = {
                "names": list(names),
                "data_fim": pass2_ws.data_fim.tolist(),
                "prior_sigma": pass2_ws.prior_sigma.tolist(),
                "ci_scale": pass2_ws.ci_scale.tolist(),
                "fitted_names": union_names,
                "fitted_values": {
                    n: float(theta_orig[name_to_index[n]]) for n in union_names
                },
            }
        logger.info(
            "safe-claim certification: %d prior-robust safe (%d strict)",
            report.n_safe, len(strict_claims),
        )

    # theta_best is always reported in ORIGINAL physical coordinates
    # (k_0 = rate at 298.15 K, Ds in m^2/s, ...); the reparameterized
    # values, whose coordinates the identifiability claims refer to, go
    # to metadata stage by stage: physics coordinates first (tau_d,
    # R_sei, ...), then the Arrhenius stage (rates at T_c), then raw.
    theta_arrhenius_stage = (
        physics_reparam.to_original(names, theta_full)
        if physics_reparam is not None
        else theta_full
    )
    theta_report = (
        reparameterizer.to_original(names, theta_arrhenius_stage)
        if reparameterizer is not None
        else theta_arrhenius_stage
    )
    native_bound_status = "NOT_APPLICABLE"
    if physics_reparam is not None:
        try:
            physics_reparam.validate_coupled_raw_bounds(names, theta_full)
            native_bound_status = "PASS"
        except ValueError:
            native_bound_status = "FAIL"
            if policy is NuisancePolicy.ALL48_WEAK_PRIOR:
                raise
    theta_final_fit = np.asarray(theta_full, dtype=float)
    initial_fit_search = encode_parameter_vector(specs, theta_initial_fit)
    final_fit_search = encode_parameter_vector(specs, theta_final_fit)
    prior_sigma_fit = prior_set.sigma_search_vector(specs)
    hard_fixed_names = [name for name in names if name not in attempted_names]
    nuisance_names = (
        list(names)
        if policy is NuisancePolicy.ALL48_WEAK_PRIOR
        else hard_fixed_names
    )
    boundary_hits = {
        spec.name: bool(
            np.isclose(theta_report[i], spec.bounds[0], rtol=0.0, atol=1e-12)
            or np.isclose(theta_report[i], spec.bounds[1], rtol=0.0, atol=1e-12)
        )
        for i, spec in enumerate(raw_specs)
    }
    theta_final_subset = np.asarray(
        [theta_full[name_to_index[spec.name]] for spec in final_specs], dtype=float
    )
    objective_audit = _fit_objective_and_geometry_audit(
        dataset=dataset,
        model=final_model,
        specs=final_specs,
        theta_fit=theta_final_subset,
        prior_set=prior_set,
        noise_level=noise_level,
        use_frequency_weighting=use_frequency_weighting,
    )
    metadata: dict[str, Any] = {
        "history": history,
        "gate_max_posterior_ci95": gate.max_posterior_ci95,
        "gate_min_information_gain": gate.min_information_gain,
        "profile_rescue": rescue_log,
        "prior_tightening": {
            "status": tightening_status,
            "executed": False,
            "changed_fit": False,
            "changed_claims": False,
        },
        "claim_sequences": {
            "headline_data_supported_strict": strict_claims,
            "prior_informed_nonheadline": prior_informed_claims,
            "headline_tiers": ["STRICT"],
            "merged_headline": False,
        },
        "annealing": {
            "steps": int(annealing_steps),
            "seed": int(annealing_seed),
            "restarts": int(annealing_restarts),
        },
        "optimizer": (
            dict(fit_result.metadata) if fit_result is not None else None
        ),
        "profile_certification": certification_log,
        "safe_claim_certification": safe_claim_log,
        "reparameterized_coordinates": coordinate_notes,
        "physics_coordinates": physics_notes,
        "native_coordinate_schema": {
            "version": "eis-pem-v2-publication-native-2",
            "coordinates": (
                physics_reparam.coordinate_manifest(names)
                if physics_reparam is not None
                else []
            ),
            "coupled_raw_bounds": native_bound_status,
        },
        "nuisance_policy": policy.value,
        "jointly_optimized_names": list(attempted_names),
        "hard_fixed_names": hard_fixed_names,
        "nuisance_names": nuisance_names,
        "nuisance_regularization_strength": {
            prior.name: {
                "sigma_search": float(prior.sigma_search),
                "marginal_diagonal_equivalent_weight": float(
                    noise_level / prior.sigma_search
                ),
                "source": prior.source,
            }
            for prior in prior_set.priors
        },
        "nuisance_initial_source": {
            name: "disclosed_prior_mean" for name in names
        },
        "theta_initial_raw": {
            name: float(theta_initial_raw[name_to_index[name]]) for name in names
        },
        "theta_final_raw": {
            name: float(theta_report[name_to_index[name]]) for name in names
        },
        "theta_initial_fit": {
            name: float(theta_initial_fit[name_to_index[name]]) for name in names
        },
        "theta_final_fit": {
            name: float(theta_final_fit[name_to_index[name]]) for name in names
        },
        "movement_search_coordinates": {
            name: float(final_fit_search[i] - initial_fit_search[i])
            for i, name in enumerate(names)
        },
        "movement_prior_sigma": {
            name: float(
                (final_fit_search[i] - initial_fit_search[i]) / prior_sigma_fit[i]
            )
            for i, name in enumerate(names)
        },
        "boundary_hits_raw_coordinates": boundary_hits,
        "nuisance_initial_values": {
            name: float(theta_initial_raw[name_to_index[name]])
            for name in nuisance_names
        },
        "nuisance_final_values": {
            name: float(theta_report[name_to_index[name]]) for name in nuisance_names
        },
        "nuisance_boundary_hits": {
            name: boundary_hits[name] for name in nuisance_names
        },
        "nuisance_movement": {
            name: float(
                final_fit_search[name_to_index[name]]
                - initial_fit_search[name_to_index[name]]
            )
            for name in nuisance_names
        },
        "nominal_displacement": {
            name: float(
                (
                    final_fit_search[name_to_index[name]]
                    - initial_fit_search[name_to_index[name]]
                )
                / prior_sigma_fit[name_to_index[name]]
            )
            for name in nuisance_names
        },
        "data_objective": objective_audit["data_objective"],
        "prior_penalty": objective_audit["prior_penalty"],
        "total_objective": objective_audit["total_objective"],
        "prior_curvature": objective_audit["prior_curvature"],
        "data_only_jacobian": objective_audit["data_only_jacobian"],
        "objective_and_geometry": objective_audit,
        "reference_sensitivity": {
            "status": "NOT_RUN",
            "threshold_local_width_fraction": 0.5,
            "truth_access": False,
        },
        "compute_backend": compute_backend,
    }
    if reparameterizer is not None:
        metadata["arrhenius_reference_temperatures_K"] = {
            name: float(value)
            for name, value in reparameterizer.reference_temperatures.items()
        }
        metadata["theta_best_reparameterized"] = {
            name: float(theta_arrhenius_stage[name_to_index[name]])
            for name in coordinate_notes
        }
    if physics_reparam is not None:
        metadata["physics_coordinate_manifest"] = (
            physics_reparam.coordinate_manifest(names)
        )
        metadata["theta_best_native_coordinates"] = (
            physics_reparam.native_coordinate_values(names, theta_full)
        )
    from .claim_taxonomy import publication_composite_definitions

    metadata["publication_composite_manifest"] = [
        row.to_manifest_row()
        for row in publication_composite_definitions(
            reference_temperatures_K=(
                reparameterizer.reference_temperatures
                if reparameterizer is not None
                else None
            ),
            physics_enabled=(
                physics_reparam.enabled if physics_reparam is not None else ()
            ),
        )
    ]
    metadata["publication_claim_taxonomy"] = {
        "status": "MANDATORY_AUDITS_PENDING",
        "internal_strict_reader_facing": False,
        "truth_access": False,
    }

    return SEISIdentificationResult(
        theta_best={
            name: float(theta_report[name_to_index[name]]) for name in names
        },
        tiered_selection=full_selection,
        identifiable_params=strict_claims,
        n_identifiable=len(strict_claims),
        prior_informed_params=prior_informed_claims,
        safe_claim_params=safe_claim_params,
        n_safe_claims=len(safe_claim_params),
        conditions_used=list(selected_conditions),
        condition_info_gains={str(c): float(traces[c]) for c in condition_pool},
        parameter_sensitivity_ranking=ranking,
        attempted_params=list(attempted_names),
        tier_assignment=dict(grouper.tiers),
        noise_level=float(noise_level),
        final_cost=float(fit_result.final_cost) if fit_result is not None else float("nan"),
        metadata=metadata,
    )


def _expand_selection(
    selection: TieredSelection,
    specs: Sequence[ParameterSpec],
    theta_full: FloatArray,
) -> TieredSelection:
    """Lift a subset TieredSelection to the full 48-parameter space.

    Never-attempted parameters are FIXED at their reference values with
    zero information gain and undefined (infinite) posterior CI — the
    honest statement that nothing was learned about them.
    """

    from .robust import ParameterSelection

    names = tuple(spec.name for spec in specs)
    tiers = {}
    reference = {}
    posterior_ci = {}
    prior_ci = {}
    gain = {}
    for i, name in enumerate(names):
        if name in selection.parameter_names:
            tiers[name] = selection.tiers[name]
            reference[name] = selection.reference_values[name]
            posterior_ci[name] = selection.posterior_ci95[name]
            prior_ci[name] = selection.prior_ci95[name]
            gain[name] = selection.information_gain[name]
        else:
            tiers[name] = ParameterTier.FIXED
            reference[name] = float(theta_full[i])
            posterior_ci[name] = float("inf")
            prior_ci[name] = float("inf")
            gain[name] = 0.0

    strict = selection.strict_selection
    statuses = {}
    reasons = {}
    ci95 = {}
    fixed_values = dict(strict.fixed_values)
    for name in names:
        if name in strict.statuses:
            statuses[name] = strict.statuses[name]
            reasons[name] = strict.reasons[name]
            ci95[name] = strict.ci95_relative[name]
        else:
            statuses[name] = "fixed_identifiability"
            reasons[name] = "never_attempted_fixed_at_literature_value"
            ci95[name] = float("nan")
            fixed_values[name] = reference[name]
    full_strict = ParameterSelection(
        parameter_names=names,
        reference_values=dict(reference),
        free_names=strict.free_names,
        fixed_values=fixed_values,
        statuses=statuses,
        reasons=reasons,
        ci95_relative=ci95,
        original_singular_values=strict.original_singular_values,
        selected_singular_values=strict.selected_singular_values,
        original_condition_number=strict.original_condition_number,
        reduced_condition_number=strict.reduced_condition_number,
        assumed_noise_level=strict.assumed_noise_level,
        max_condition_number=strict.max_condition_number,
        max_relative_ci95=strict.max_relative_ci95,
        protected_names=strict.protected_names,
    )
    return TieredSelection(
        parameter_names=names,
        tiers=tiers,
        reference_values=reference,
        strict_selection=full_strict,
        posterior_ci95=posterior_ci,
        prior_ci95=prior_ci,
        information_gain=gain,
        data_condition_number=selection.data_condition_number,
        posterior_condition_number=selection.posterior_condition_number,
        free_posterior_condition_number=selection.free_posterior_condition_number,
        k_eff=selection.k_eff,
        assumed_noise_level=selection.assumed_noise_level,
    )


def _specs_for(parameter_names: Sequence[str] | None) -> tuple[ParameterSpec, ...]:
    all_specs = {spec.name: spec for spec in all_seis_parameter_specs()}
    if parameter_names is None:
        return tuple(all_specs.values())
    unknown = set(parameter_names) - set(all_specs)
    if unknown:
        raise ValueError(f"unknown SEIS parameters: {sorted(unknown)}")
    return tuple(all_specs[name] for name in parameter_names)
