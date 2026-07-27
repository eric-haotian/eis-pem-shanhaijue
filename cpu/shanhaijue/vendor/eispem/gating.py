# learning AI at www.haotianblog.com
"""Tiered (posterior) identifiability gate — Generation 2.

Generation 1's gate is binary: a parameter either survives the strict
thresholds (SVD condition number, Fisher CI95, correlation) or it is
fixed at its reference value.  On slightly complex models this fixes
most parameters and effectively rejects the model, even when the DRT
carries genuine evidence about them.

Generation 2 keeps the strict gate **unchanged** (it is re-used verbatim
from :mod:`eis_pem.robust`, preserving the published zero
false-identifiability guarantee for that tier) and adds a posterior
analysis on top of it:

* With Gaussian priors of covariance ``Sigma_p`` in search space and a
  relative-residual Jacobian ``J`` with noise level ``sigma_n``, the
  posterior Fisher information is ``F_post = J^T J / sigma_n^2 +
  Sigma_p^{-1}``.  Its condition number is mathematically bounded above
  by the data-only condition number, so directions that are weak in the
  data no longer poison the whole spectrum.
* Each parameter that fails the strict gate is examined under the
  posterior: it is **PRIOR_INFORMED** only when the posterior confidence
  interval passes *and* the data contributes a verifiable share of the
  posterior information (information gain ``1 - var_post/var_prior``
  above threshold).  This gain criterion is the false-identifiability
  guard: a parameter whose posterior merely echoes its prior can never
  be reported as identified.
* Parameters that are stable only because of the prior are
  **PRIOR_DOMINATED** — still optimized (the prior keeps them from
  destabilizing the fit) but explicitly excluded from identifiability
  claims.
* Parameters the posterior cannot bound at all remain **FIXED**, exactly
  as in Generation 1.

The gate also returns the effective number of parameters
``k_eff = tr[F_data (F_data + Sigma_p^{-1})^{-1}]`` (the generalized
degrees of freedom of the MAP problem), which the Generation-2 model
selection uses so complex models are penalized only for the degrees of
freedom the data actually pays for.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .forward_models import ForwardModel
from .parameters import (
    ParameterSpec,
    decode_parameter_vector,
    encode_parameter_vector,
)
from .robust import (
    IdentifiabilitySelector,
    IdentifiabilityStrategy,
    ParameterSelection,
)

from .priors import PriorSet

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


class ParameterTier(str, enum.Enum):
    """Identifiability tier of one parameter."""

    STRICT = "strict"
    PRIOR_INFORMED = "prior_informed"
    PRIOR_DOMINATED = "prior_dominated"
    FIXED = "fixed"


#: Report-facing status strings, aligned with the Generation-1 taxonomy.
TIER_STATUS = {
    ParameterTier.STRICT: "estimated",
    ParameterTier.PRIOR_INFORMED: "estimated_prior_informed",
    ParameterTier.PRIOR_DOMINATED: "estimated_prior_dominated",
    ParameterTier.FIXED: "fixed_identifiability",
}


@dataclass(frozen=True)
class TieredSelection:
    """Result of the tiered gate for one model at one reference point."""

    parameter_names: tuple[str, ...]
    tiers: dict[str, ParameterTier]
    reference_values: dict[str, float]
    strict_selection: ParameterSelection
    posterior_ci95: dict[str, float]
    prior_ci95: dict[str, float]
    information_gain: dict[str, float]
    data_condition_number: float
    posterior_condition_number: float
    free_posterior_condition_number: float
    k_eff: float
    assumed_noise_level: float

    def __post_init__(self) -> None:
        names = set(self.parameter_names)
        for mapping in (
            self.tiers,
            self.reference_values,
            self.posterior_ci95,
            self.prior_ci95,
            self.information_gain,
        ):
            if set(mapping) != names:
                raise ValueError("tier metadata must cover all parameters")

    @property
    def free_names(self) -> tuple[str, ...]:
        """Parameters left free for the MAP fit (all non-FIXED tiers)."""

        return tuple(
            name
            for name in self.parameter_names
            if self.tiers[name] is not ParameterTier.FIXED
        )

    @property
    def fixed_values(self) -> dict[str, float]:
        return {
            name: self.reference_values[name]
            for name in self.parameter_names
            if self.tiers[name] is ParameterTier.FIXED
        }

    @property
    def statuses(self) -> dict[str, str]:
        return {name: TIER_STATUS[self.tiers[name]] for name in self.parameter_names}

    def count(self, tier: ParameterTier) -> int:
        return sum(1 for t in self.tiers.values() if t is tier)

    @property
    def n_strict(self) -> int:
        return self.count(ParameterTier.STRICT)

    @property
    def data_supported_names(self) -> tuple[str, ...]:
        """STRICT-only names eligible for a publication headline."""

        return tuple(
            name
            for name in self.parameter_names
            if self.tiers[name] is ParameterTier.STRICT
        )

    @property
    def prior_informed_names(self) -> tuple[str, ...]:
        """Prior-informed diagnostics, always separate from the headline."""

        return tuple(
            name
            for name in self.parameter_names
            if self.tiers[name] is ParameterTier.PRIOR_INFORMED
        )

    @property
    def n_prior_informed(self) -> int:
        return self.count(ParameterTier.PRIOR_INFORMED)

    @property
    def n_prior_dominated(self) -> int:
        return self.count(ParameterTier.PRIOR_DOMINATED)

    @property
    def n_fixed(self) -> int:
        return self.count(ParameterTier.FIXED)

    @property
    def n_identifiable(self) -> int:
        """Legacy combined count; publication serializers must not use it."""

        return self.n_strict + self.n_prior_informed

    def to_frame(self) -> pd.DataFrame:
        """One audit row per parameter."""

        return pd.DataFrame(
            {
                "parameter": self.parameter_names,
                "tier": [self.tiers[n].value for n in self.parameter_names],
                "status": [self.statuses[n] for n in self.parameter_names],
                "reference_value": [
                    self.reference_values[n] for n in self.parameter_names
                ],
                "strict_status": [
                    self.strict_selection.statuses[n] for n in self.parameter_names
                ],
                "posterior_ci95_relative": [
                    self.posterior_ci95[n] for n in self.parameter_names
                ],
                "prior_ci95_relative": [
                    self.prior_ci95[n] for n in self.parameter_names
                ],
                "information_gain": [
                    self.information_gain[n] for n in self.parameter_names
                ],
                "data_condition_number": self.data_condition_number,
                "posterior_condition_number": self.posterior_condition_number,
                "k_eff": self.k_eff,
            }
        )


@dataclass(frozen=True)
class TieredIdentifiabilityGate:
    """Strict Gen-1 gate + posterior tier analysis.

    Parameters
    ----------
    assumed_noise_level
        Relative measurement noise used to whiten the Fisher information
        (same convention as Gen 1).
    max_posterior_ci95
        Posterior relative CI95 a parameter must satisfy to remain free.
        Deliberately looser than the strict threshold — the strict
        guarantee lives entirely in the STRICT tier.
    min_information_gain
        Minimum fraction of the posterior precision that must come from
        the data (``1 - var_post/var_prior``) for a PRIOR_INFORMED claim.
        0.5 means the data at least halves the prior variance.
    strict_*
        Passed through to the Generation-1 :class:`IdentifiabilitySelector`
        so the strict tier reproduces the published gate exactly.
    max_gate_refinement_iters
        Maximum gate-evaluate → reduced-refit → re-evaluate rounds used by
        the model-comparison loop (see ``selection._fit_and_gate_one``).
        The Fisher information of the reduced problem differs from the
        full one, so borderline parameters can legitimately change tier
        once weak directions are frozen; re-gating at the refined point
        catches those promotions.  1 reproduces one-shot gating.
    """

    assumed_noise_level: float = 0.005
    max_posterior_ci95: float = 0.20
    min_information_gain: float = 0.50
    strict_strategy: IdentifiabilityStrategy = IdentifiabilityStrategy.ADAPTIVE
    strict_max_condition_number: float = 1e4
    strict_max_relative_ci95: float = 0.10
    protected_names: tuple[str, ...] = ()
    step_size: float = 1e-5
    eps: float = 1e-12
    max_gate_refinement_iters: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.assumed_noise_level) or self.assumed_noise_level <= 0:
            raise ValueError("assumed_noise_level must be finite and positive")
        if not 0 < self.max_posterior_ci95:
            raise ValueError("max_posterior_ci95 must be positive")
        if not 0 < self.min_information_gain < 1:
            raise ValueError("min_information_gain must be in (0, 1)")
        if int(self.max_gate_refinement_iters) < 1:
            raise ValueError("max_gate_refinement_iters must be >= 1")

    def evaluate(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        theta_reference: NDArray[np.floating],
        prior_set: PriorSet,
        weights: NDArray[np.floating] | None = None,
    ) -> TieredSelection:
        specs = tuple(parameter_specs)
        names = tuple(spec.name for spec in specs)
        reference = np.asarray(theta_reference, dtype=float)
        if reference.shape != (len(specs),) or not np.all(np.isfinite(reference)):
            raise ValueError("theta_reference must be finite and match parameter_specs")

        # --- Tier 1: the unchanged Generation-1 strict gate ---
        strict_selector = IdentifiabilitySelector(
            max_condition_number=self.strict_max_condition_number,
            assumed_noise_level=self.assumed_noise_level,
            max_relative_ci95=self.strict_max_relative_ci95,
            protected_names=self.protected_names,
            strategy=self.strict_strategy,
        )
        strict = strict_selector.select(dataset, model, specs, reference, weights)
        strict_free = {
            name
            for name in strict.free_names
            if strict.statuses[name] == "estimated"
        }

        # --- Posterior Fisher analysis ---
        jacobian = relative_search_jacobian(
            dataset, model, specs, reference, self.step_size, self.eps, weights
        )
        sigma_n = self.assumed_noise_level
        sigma_prior = prior_set.sigma_search_vector(specs)
        covariance_prior = prior_set.covariance_search_matrix(specs)

        fim_data = jacobian.T @ jacobian / sigma_n**2
        prior_precision = np.linalg.inv(covariance_prior)
        fim_post = fim_data + prior_precision
        cov_post = _stable_inverse(fim_post)
        var_post = np.maximum(np.diag(cov_post), 0.0)

        gain = 1.0 - var_post / np.diag(covariance_prior)
        gain = np.clip(gain, 0.0, 1.0)

        posterior_ci95 = _relative_ci95_from_sd(np.sqrt(var_post), specs, reference)
        prior_ci95 = _relative_ci95_from_sd(sigma_prior, specs, reference)

        data_cond = _condition_number_of(jacobian)
        posterior_cond = _condition_number_of(
            np.vstack((jacobian / sigma_n, np.linalg.cholesky(prior_precision).T))
        )

        # --- Tier assignment ---
        tiers: dict[str, ParameterTier] = {}
        for i, name in enumerate(names):
            if name in strict_free:
                tiers[name] = ParameterTier.STRICT
            elif posterior_ci95[i] <= self.max_posterior_ci95:
                if gain[i] >= self.min_information_gain:
                    tiers[name] = ParameterTier.PRIOR_INFORMED
                else:
                    tiers[name] = ParameterTier.PRIOR_DOMINATED
            else:
                tiers[name] = ParameterTier.FIXED

        # --- Effective degrees of freedom & conditioning on the free set ---
        free_idx = [
            i for i, name in enumerate(names) if tiers[name] is not ParameterTier.FIXED
        ]
        if free_idx:
            fim_data_free = fim_data[np.ix_(free_idx, free_idx)]
            fim_post_free = fim_post[np.ix_(free_idx, free_idx)]
            k_eff = float(np.trace(np.linalg.solve(fim_post_free, fim_data_free)))
            free_posterior_cond = _condition_number_of(
                np.vstack(
                    (
                        jacobian[:, free_idx] / sigma_n,
                        np.linalg.cholesky(
                            prior_precision[np.ix_(free_idx, free_idx)]
                        ).T,
                    )
                )
            )
        else:  # pragma: no cover - degenerate
            k_eff = 0.0
            free_posterior_cond = float("inf")

        return TieredSelection(
            parameter_names=names,
            tiers=tiers,
            reference_values=dict(zip(names, reference, strict=True)),
            strict_selection=strict,
            posterior_ci95=dict(zip(names, posterior_ci95, strict=True)),
            prior_ci95=dict(zip(names, prior_ci95, strict=True)),
            information_gain=dict(zip(names, gain, strict=True)),
            data_condition_number=data_cond,
            posterior_condition_number=posterior_cond,
            free_posterior_condition_number=free_posterior_cond,
            k_eff=k_eff,
            assumed_noise_level=sigma_n,
        )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def relative_search_jacobian(
    dataset: EISDataset,
    model: ForwardModel,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
    step_size: float = 1e-5,
    eps: float = 1e-12,
    weights: NDArray[np.floating] | None = None,
) -> FloatArray:
    """Central-difference Jacobian of relative residuals in search space.

    Same conventions as the Generation-1 selector: predictions are
    normalized by ``|Z_ref|`` and optionally scaled by ``sqrt(weights)``;
    real and imaginary parts are stacked.
    """

    spec_tuple = tuple(specs)
    search_reference = encode_parameter_vector(spec_tuple, theta_reference)
    prediction_reference = model.simulate(dataset.freq_hz, theta_reference)
    scale = np.maximum(np.abs(prediction_reference), eps)
    sqrt_weights = None
    if weights is not None:
        values = np.asarray(weights, dtype=float)
        if values.shape != dataset.freq_hz.shape:
            raise ValueError("weights must have the same shape as dataset frequencies")
        sqrt_weights = np.sqrt(values)

    def delta(search_values: FloatArray) -> FloatArray:
        prediction = model.simulate(
            dataset.freq_hz, decode_parameter_vector(spec_tuple, search_values)
        )
        rel = (prediction - prediction_reference) / scale
        if sqrt_weights is not None:
            rel = rel * sqrt_weights
        return np.concatenate((rel.real, rel.imag))

    jacobian = np.empty((dataset.freq_hz.size * 2, len(spec_tuple)), dtype=float)
    for index in range(len(spec_tuple)):
        upper = search_reference.copy()
        lower = search_reference.copy()
        upper[index] += step_size
        lower[index] -= step_size
        jacobian[:, index] = (delta(upper) - delta(lower)) / (2.0 * step_size)
    return jacobian


def _relative_ci95_from_sd(
    sd_search: FloatArray,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
) -> FloatArray:
    """Search-space standard deviations → relative CI95 (Gen-1 convention)."""

    relative_sd = np.asarray(
        [
            np.log(10.0) * sd if spec.log_transform else sd / max(abs(value), 1e-30)
            for spec, value, sd in zip(specs, theta_reference, sd_search, strict=True)
        ],
        dtype=float,
    )
    return 1.96 * relative_sd


def _condition_number_of(matrix: FloatArray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _stable_inverse(matrix: FloatArray) -> FloatArray:
    """Invert a symmetric positive-definite matrix with a jitter fallback."""

    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        jitter = 1e-12 * float(np.trace(matrix)) / max(matrix.shape[0], 1)
        return np.linalg.pinv(matrix + jitter * np.eye(matrix.shape[0]), hermitian=True)
