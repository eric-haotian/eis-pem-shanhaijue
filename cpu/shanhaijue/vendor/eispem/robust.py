# learning AI at www.haotianblog.com
"""Identifiability-aware parameter selection for robust SEIS PEM fits."""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .forward_models import ForwardModel
from .identifiability import (
    estimator_correlation_matrix,
    max_abs_off_diagonal,
    max_abs_off_diagonal_by_parameter,
    sensitivity_collinearity_matrix,
)
from .parameters import ParameterSpec, decode_parameter_vector, encode_parameter_vector
from .seis_model import DEFAULT_SELECTED_PARAMETER_NAMES

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


def recommend_tikhonov_alpha(
    quality_score: float,
    valid_fraction: float,
    estimated_noise_level: float,
) -> float:
    """Recommend search-space Tikhonov regularization for robust fits.

    The regularizer anchors weak directions to the reference parameter vector.
    It is disabled for clean, information-rich datasets and strengthened only
    when the data-quality gate indicates that outliers, drift, inductive tails,
    or high noise make the EIS-only inverse problem underdetermined.
    """

    for name, value in (
        ("quality_score", quality_score),
        ("valid_fraction", valid_fraction),
        ("estimated_noise_level", estimated_noise_level),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= quality_score <= 1:
        raise ValueError("quality_score must be between 0 and 1")
    if not 0 <= valid_fraction <= 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    if estimated_noise_level < 0:
        raise ValueError("estimated_noise_level must be non-negative")

    if quality_score < 0.40 or valid_fraction < 0.55 or estimated_noise_level > 0.05:
        return 0.3
    if quality_score < 0.70 or valid_fraction < 0.65 or estimated_noise_level > 0.02:
        return 0.03
    return 0.0


def recommend_robust_f_scale(estimated_noise_level: float) -> float:
    """Return the residual scale for robust least-squares loss.

    Robust losses operate in the residual-vector units. The robust real-data
    path uses relative residuals, so the estimated relative noise level is the
    natural transition scale. The lower bound prevents over-clipping clean data;
    the upper bound prevents very noisy data from effectively reverting to
    ordinary least squares.
    """

    if not np.isfinite(estimated_noise_level) or estimated_noise_level < 0:
        raise ValueError("estimated_noise_level must be finite and non-negative")
    return float(np.clip(estimated_noise_level, 0.01, 0.10))


class IdentifiabilityStrategy(enum.Enum):
    """Parameter selection strategy.

    CLASSIC: Original three-stage selection (correlation, SVD, CI95).
    ADAPTIVE: Four-criterion joint evaluation with iterative refinement.
              Recommended for real experimental data with noise, limited
              frequency range, and model mismatch.
    """

    CLASSIC = "classic"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class ParameterSelection:
    """Selected free parameters and documented constraints for one experiment."""

    parameter_names: tuple[str, ...]
    reference_values: dict[str, float]
    free_names: tuple[str, ...]
    fixed_values: dict[str, float]
    statuses: dict[str, str]
    reasons: dict[str, str]
    ci95_relative: dict[str, float]
    original_singular_values: FloatArray
    selected_singular_values: FloatArray
    original_condition_number: float
    reduced_condition_number: float
    assumed_noise_level: float
    max_condition_number: float
    max_relative_ci95: float
    protected_names: tuple[str, ...]
    max_sensitivity_collinearity: dict[str, float] = field(default_factory=dict)
    max_estimator_correlation: dict[str, float] = field(default_factory=dict)
    max_sensitivity_collinearity_threshold: float = 0.95
    max_estimator_correlation_threshold: float = 0.95
    estimator_correlation_is_gate_constraint: bool = False
    converged: bool = True
    termination_reason: str = "criteria_satisfied"
    iterations: int = 0
    removal_trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parameter_set = set(self.parameter_names)
        free_set = set(self.free_names)
        fixed_set = set(self.fixed_values)
        if not self.parameter_names or len(parameter_set) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique and non-empty")
        if (
            free_set.intersection(fixed_set)
            or free_set.union(fixed_set) != parameter_set
        ):
            raise ValueError("free and fixed parameters must partition parameter_names")
        for mapping in (
            self.reference_values,
            self.statuses,
            self.reasons,
            self.ci95_relative,
        ):
            if set(mapping) != parameter_set:
                raise ValueError("selection metadata must cover all parameters")
        if not self.max_sensitivity_collinearity:
            object.__setattr__(
                self,
                "max_sensitivity_collinearity",
                {name: float("nan") for name in self.parameter_names},
            )
        if not self.max_estimator_correlation:
            object.__setattr__(
                self,
                "max_estimator_correlation",
                {name: float("nan") for name in self.parameter_names},
            )
        if set(self.max_sensitivity_collinearity) != parameter_set:
            raise ValueError(
                "sensitivity-collinearity metadata must cover all parameters"
            )
        if set(self.max_estimator_correlation) != parameter_set:
            raise ValueError(
                "estimator-correlation metadata must cover all parameters"
            )
        if not set(self.protected_names).issubset(parameter_set):
            raise ValueError("protected_names must occur in parameter_names")

    @property
    def fixed_names(self) -> tuple[str, ...]:
        """Return constrained parameter names in original parameter order."""

        return tuple(name for name in self.parameter_names if name in self.fixed_values)

    @property
    def candidate_strict_names(self) -> tuple[str, ...]:
        """Return locally admissible candidates before the global gate decision."""

        return tuple(
            name
            for name in self.free_names
            if self.statuses[name] == "estimated"
            and name not in self.protected_names
        )

    @property
    def declared_strict_names(self) -> tuple[str, ...]:
        """Return strict declarations only after all gate criteria are satisfied."""

        return self.candidate_strict_names if self.converged else ()

    @property
    def reported_protected_names(self) -> tuple[str, ...]:
        """Return fitted parameters that cannot receive a strict declaration."""

        strict = set(self.declared_strict_names)
        return tuple(name for name in self.free_names if name not in strict)

    def to_frame(self) -> pd.DataFrame:
        """Return one audit row per complete-model physical parameter."""

        return pd.DataFrame(
            {
                "parameter": self.parameter_names,
                "status": [self.statuses[name] for name in self.parameter_names],
                "reason": [self.reasons[name] for name in self.parameter_names],
                "reference_value": [
                    self.reference_values[name] for name in self.parameter_names
                ],
                "fixed_value": [
                    self.fixed_values.get(name, np.nan) for name in self.parameter_names
                ],
                "ci95_relative": [
                    self.ci95_relative[name] for name in self.parameter_names
                ],
                "max_sensitivity_collinearity": [
                    self.max_sensitivity_collinearity[name]
                    for name in self.parameter_names
                ],
                "max_estimator_correlation": [
                    self.max_estimator_correlation[name]
                    for name in self.parameter_names
                ],
                "is_protected": [
                    name in self.protected_names for name in self.parameter_names
                ],
                "original_condition_number": self.original_condition_number,
                "reduced_condition_number": self.reduced_condition_number,
                "selection_converged": self.converged,
                "selection_termination_reason": self.termination_reason,
                "selection_iterations": self.iterations,
                "estimator_correlation_is_gate_constraint": (
                    self.estimator_correlation_is_gate_constraint
                ),
            }
        )

    def singular_values_frame(self) -> pd.DataFrame:
        """Return full and selected-model spectra for conditioning diagnostics."""

        frames = []
        for stage, singular_values in (
            ("full", self.original_singular_values),
            ("selected", self.selected_singular_values),
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "stage": stage,
                        "singular_value_index": np.arange(1, singular_values.size + 1),
                        "singular_value": singular_values,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def export_csv(self, path: str | Path) -> Path:
        """Export parameter selection decisions and their numerical context."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)
        return output_path

    def export_singular_values_csv(self, path: str | Path) -> Path:
        """Export singular values before and after parameter selection."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.singular_values_frame().to_csv(output_path, index=False)
        return output_path


@dataclass(frozen=True)
class IdentifiabilitySelector:
    """Select EIS-supported parameters while preserving named target parameters.

    Two strategies are available:

    **CLASSIC** (default): Three-stage selection \u2014 sensitivity-collinearity
    pre-screening, SVD condition-number reduction, and CI95 pruning.

    **ADAPTIVE**: Four-criterion joint evaluation with iterative refinement.
    Uses rank(J), \u03ba(J), CI95, Jacobian-column collinearity, and
    covariance-derived estimator correlation. Estimator correlation is always
    reported and is used for removal only when explicitly enabled. The
    deterministic greedy heuristic does not guarantee a globally maximum-
    cardinality feasible subset.
    """

    max_condition_number: float = 1e4
    assumed_noise_level: float = 0.005
    max_relative_ci95: float = 0.10
    protected_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES
    step_size: float = 1e-5
    eps: float = 1e-12
    covariance_rcond: float = 1e-14

    # Adaptive strategy fields (only used when strategy=ADAPTIVE)
    strategy: IdentifiabilityStrategy = IdentifiabilityStrategy.CLASSIC
    min_rank_fraction: float = 0.5
    max_sensitivity_collinearity_threshold: float = 0.95
    max_estimator_correlation_threshold: float = 0.95
    apply_estimator_correlation_gate: bool = False
    eigenvalue_gap_ratio: float = 100.0
    max_selection_iterations: int = 50

    def __post_init__(self) -> None:
        for name, value in (
            ("max_condition_number", self.max_condition_number),
            ("max_relative_ci95", self.max_relative_ci95),
            ("step_size", self.step_size),
            ("eps", self.eps),
            ("covariance_rcond", self.covariance_rcond),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.assumed_noise_level) or self.assumed_noise_level < 0:
            raise ValueError("assumed_noise_level must be finite and non-negative")
        if len(set(self.protected_names)) != len(self.protected_names):
            raise ValueError("protected_names must be unique")
        if not isinstance(self.strategy, IdentifiabilityStrategy):
            raise ValueError("strategy must be an IdentifiabilityStrategy")
        if not 0 < self.min_rank_fraction <= 1:
            raise ValueError("min_rank_fraction must be in (0, 1]")
        if not 0 < self.max_sensitivity_collinearity_threshold < 1:
            raise ValueError(
                "max_sensitivity_collinearity_threshold must be in (0, 1)"
            )
        if not 0 < self.max_estimator_correlation_threshold < 1:
            raise ValueError(
                "max_estimator_correlation_threshold must be in (0, 1)"
            )
        if not np.isfinite(self.eigenvalue_gap_ratio) or self.eigenvalue_gap_ratio <= 1:
            raise ValueError("eigenvalue_gap_ratio must be finite and > 1")
        if self.max_selection_iterations < 1:
            raise ValueError("max_selection_iterations must be >= 1")

    def select(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        theta_reference: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None,
    ) -> ParameterSelection:
        """Select the free vector for a local relative-residual PEM problem."""

        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        parameter_names = tuple(spec.name for spec in specs)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter names must be unique")
        if not set(self.protected_names).issubset(parameter_names):
            raise ValueError("protected parameters must occur in parameter_specs")
        reference = np.asarray(theta_reference, dtype=float)
        if reference.shape != (len(specs),) or not np.all(np.isfinite(reference)):
            raise ValueError("theta_reference must be finite and match parameter_specs")

        jacobian = _relative_prediction_jacobian(
            dataset=dataset,
            model=model,
            specs=specs,
            theta_reference=reference,
            step_size=self.step_size,
            eps=self.eps,
            weights=weights,
        )

        if self.strategy == IdentifiabilityStrategy.ADAPTIVE:
            return self._select_adaptive(jacobian, specs, parameter_names, reference)
        return self._select_classic(jacobian, specs, parameter_names, reference)

    def _select_classic(
        self,
        jacobian: FloatArray,
        specs: tuple[ParameterSpec, ...],
        parameter_names: tuple[str, ...],
        reference: FloatArray,
    ) -> ParameterSelection:
        """Deterministic three-stage greedy backward elimination."""

        full_singular_values, original_condition_number = _spectrum(jacobian)
        active = list(range(len(specs)))
        fixed_values: dict[str, float] = {}
        statuses = {name: "estimated" for name in parameter_names}
        reasons = {
            name: "selected_under_identifiability_thresholds"
            for name in parameter_names
        }
        ci95 = {name: float("nan") for name in parameter_names}
        max_gamma = {name: float("nan") for name in parameter_names}
        max_rho = {name: float("nan") for name in parameter_names}
        protected = set(self.protected_names)
        removal_trace: list[str] = []

        sensitivity_norms = np.linalg.norm(jacobian, axis=0)
        gamma = sensitivity_collinearity_matrix(jacobian)
        rho = estimator_correlation_matrix(
            jacobian,
            sigma2=self.assumed_noise_level**2,
            rcond=self.covariance_rcond,
        )
        gamma_by_parameter = max_abs_off_diagonal_by_parameter(gamma)
        rho_by_parameter = max_abs_off_diagonal_by_parameter(rho)

        removed_by_dependence: set[int] = set()
        for i in range(len(specs)):
            if i in removed_by_dependence or parameter_names[i] in protected:
                continue
            for j in range(i + 1, len(specs)):
                if j in removed_by_dependence or parameter_names[j] in protected:
                    continue
                high_gamma = (
                    abs(gamma[i, j])
                    > self.max_sensitivity_collinearity_threshold
                )
                high_rho = bool(
                    self.apply_estimator_correlation_gate
                    and abs(rho[i, j]) > self.max_estimator_correlation_threshold
                )
                if not (high_gamma or high_rho):
                    continue
                victim = i if sensitivity_norms[i] < sensitivity_norms[j] else j
                removed_by_dependence.add(victim)
                name = parameter_names[victim]
                fixed_values[name] = float(reference[victim])
                statuses[name] = "fixed_identifiability"
                if high_gamma and high_rho:
                    reason = (
                        "sensitivity_collinearity_and_estimator_correlation_"
                        "above_threshold"
                    )
                elif high_gamma:
                    reason = "sensitivity_collinearity_above_threshold"
                else:
                    reason = "estimator_correlation_above_threshold"
                reasons[name] = reason
                max_gamma[name] = float(gamma_by_parameter[victim])
                max_rho[name] = float(rho_by_parameter[victim])
                removal_trace.append(f"{name}:{reason}")
        active = [index for index in active if index not in removed_by_dependence]

        while active:
            active_jacobian = jacobian[:, active]
            _, singular_values, right_vectors = np.linalg.svd(
                active_jacobian, full_matrices=False
            )
            condition_number = _condition_number(singular_values)
            if condition_number <= self.max_condition_number:
                break
            removal_position = next(
                (
                    int(position)
                    for position in np.argsort(
                        -np.abs(right_vectors[-1]), kind="mergesort"
                    )
                    if parameter_names[active[int(position)]] not in protected
                ),
                None,
            )
            if removal_position is None:
                break
            active_gamma = sensitivity_collinearity_matrix(active_jacobian)
            active_rho = estimator_correlation_matrix(
                active_jacobian,
                sigma2=self.assumed_noise_level**2,
                rcond=self.covariance_rcond,
            )
            gamma_values = max_abs_off_diagonal_by_parameter(active_gamma)
            rho_values = max_abs_off_diagonal_by_parameter(active_rho)
            index = active.pop(removal_position)
            name = parameter_names[index]
            fixed_values[name] = float(reference[index])
            statuses[name] = "fixed_identifiability"
            reasons[name] = "weak_singular_direction"
            max_gamma[name] = float(gamma_values[removal_position])
            max_rho[name] = float(rho_values[removal_position])
            removal_trace.append(f"{name}:weak_singular_direction")

        while active:
            active_jacobian = jacobian[:, active]
            active_ci95 = _relative_ci95(
                active_jacobian,
                [specs[index] for index in active],
                reference[active],
                self.assumed_noise_level,
                self.covariance_rcond,
            )
            candidates = [
                (float(active_ci95[position]), position, index)
                for position, index in enumerate(active)
                if parameter_names[index] not in protected
                and active_ci95[position] > self.max_relative_ci95
            ]
            if not candidates:
                break
            removed_ci95, removal_position, index = min(
                candidates, key=lambda item: (-item[0], item[2])
            )
            active_gamma = sensitivity_collinearity_matrix(active_jacobian)
            active_rho = estimator_correlation_matrix(
                active_jacobian,
                sigma2=self.assumed_noise_level**2,
                rcond=self.covariance_rcond,
            )
            gamma_values = max_abs_off_diagonal_by_parameter(active_gamma)
            rho_values = max_abs_off_diagonal_by_parameter(active_rho)
            active.pop(removal_position)
            name = parameter_names[index]
            fixed_values[name] = float(reference[index])
            statuses[name] = "fixed_identifiability"
            reasons[name] = "predicted_ci95_above_limit"
            ci95[name] = removed_ci95
            max_gamma[name] = float(gamma_values[removal_position])
            max_rho[name] = float(rho_values[removal_position])
            removal_trace.append(f"{name}:predicted_ci95_above_limit")

        selected_jacobian = jacobian[:, active]
        selected_singular_values, reduced_condition_number = _spectrum(
            selected_jacobian
        )
        selected_ci95 = _relative_ci95(
            selected_jacobian,
            [specs[index] for index in active],
            reference[active],
            self.assumed_noise_level,
            self.covariance_rcond,
        )
        selected_gamma = sensitivity_collinearity_matrix(selected_jacobian)
        selected_rho = estimator_correlation_matrix(
            selected_jacobian,
            sigma2=self.assumed_noise_level**2,
            rcond=self.covariance_rcond,
        )
        selected_gamma_max = max_abs_off_diagonal_by_parameter(selected_gamma)
        selected_rho_max = max_abs_off_diagonal_by_parameter(selected_rho)
        strict_positions = [
            position
            for position, index in enumerate(active)
            if parameter_names[index] not in protected
        ]
        for position, index in enumerate(active):
            name = parameter_names[index]
            ci95[name] = float(selected_ci95[position])
            max_gamma[name] = float(selected_gamma_max[position])
            max_rho[name] = float(selected_rho_max[position])
            if name in protected and (
                selected_ci95[position] > self.max_relative_ci95
                or selected_gamma_max[position]
                > self.max_sensitivity_collinearity_threshold
                or (
                    self.apply_estimator_correlation_gate
                    and selected_rho_max[position]
                    > self.max_estimator_correlation_threshold
                )
            ):
                statuses[name] = "protected_high_uncertainty"
                reasons[name] = "protected_parameter_retained_despite_gate_limit"

        selected_rank = int(np.linalg.matrix_rank(selected_jacobian))
        criteria_satisfied = (
            selected_rank == len(active)
            and reduced_condition_number <= self.max_condition_number
            and all(
                selected_gamma_max[position]
                <= self.max_sensitivity_collinearity_threshold
                for position in strict_positions
            )
            and (
                not self.apply_estimator_correlation_gate
                or all(
                    selected_rho_max[position]
                    <= self.max_estimator_correlation_threshold
                    for position in strict_positions
                )
            )
            and all(
                value <= self.max_relative_ci95
                for value, index in zip(selected_ci95, active, strict=True)
                if parameter_names[index] not in protected
            )
        )
        termination_reason = (
            "criteria_satisfied" if criteria_satisfied else "protected_boundary"
        )

        return ParameterSelection(
            parameter_names=parameter_names,
            reference_values=dict(zip(parameter_names, reference, strict=True)),
            free_names=tuple(parameter_names[index] for index in active),
            fixed_values=fixed_values,
            statuses=statuses,
            reasons=reasons,
            ci95_relative=ci95,
            original_singular_values=full_singular_values,
            selected_singular_values=selected_singular_values,
            original_condition_number=original_condition_number,
            reduced_condition_number=reduced_condition_number,
            assumed_noise_level=self.assumed_noise_level,
            max_condition_number=self.max_condition_number,
            max_relative_ci95=self.max_relative_ci95,
            protected_names=self.protected_names,
            max_sensitivity_collinearity=max_gamma,
            max_estimator_correlation=max_rho,
            max_sensitivity_collinearity_threshold=(
                self.max_sensitivity_collinearity_threshold
            ),
            max_estimator_correlation_threshold=(
                self.max_estimator_correlation_threshold
            ),
            estimator_correlation_is_gate_constraint=(
                self.apply_estimator_correlation_gate
            ),
            converged=criteria_satisfied,
            termination_reason=termination_reason,
            iterations=len(removal_trace),
            removal_trace=tuple(removal_trace),
        )

    def _select_adaptive(
        self,
        jacobian: FloatArray,
        specs: tuple[ParameterSpec, ...],
        parameter_names: tuple[str, ...],
        reference: FloatArray,
    ) -> ParameterSelection:
        """Construct a diagnostically admissible subset by backward elimination.

        Every iteration recomputes rank, condition number, data-only CI95,
        sensitivity collinearity, and estimator correlation. The highest-score
        non-protected parameter is removed. Exact score ties are resolved by
        original parameter order. This deterministic heuristic does not solve
        the conceptual maximum-cardinality subset problem globally.
        """

        full_singular_values, original_condition_number = _spectrum(jacobian)
        active = list(range(len(specs)))
        fixed_values: dict[str, float] = {}
        statuses = {name: "estimated" for name in parameter_names}
        reasons = {
            name: "selected_under_identifiability_thresholds"
            for name in parameter_names
        }
        ci95 = {name: float("nan") for name in parameter_names}
        max_gamma = {name: float("nan") for name in parameter_names}
        max_rho = {name: float("nan") for name in parameter_names}
        protected = set(self.protected_names)
        removal_trace: list[str] = []
        converged = False
        termination_reason = "max_iterations"

        for iteration in range(self.max_selection_iterations):
            if len(active) <= len(protected):
                termination_reason = "protected_boundary"
                break

            active_jacobian = jacobian[:, active]
            n_active = len(active)
            singular_values = np.linalg.svd(active_jacobian, compute_uv=False)
            effective_rank = _effective_rank(
                singular_values, self.eigenvalue_gap_ratio
            )
            min_required_rank = max(
                1, int(np.ceil(n_active * self.min_rank_fraction))
            )
            rank_deficient = effective_rank < min_required_rank
            condition_number = _condition_number(singular_values)
            ill_conditioned = condition_number > self.max_condition_number
            active_ci95 = _relative_ci95(
                active_jacobian,
                [specs[index] for index in active],
                reference[active],
                self.assumed_noise_level,
                self.covariance_rcond,
            )
            gamma = sensitivity_collinearity_matrix(active_jacobian)
            rho = estimator_correlation_matrix(
                active_jacobian,
                sigma2=self.assumed_noise_level**2,
                rcond=self.covariance_rcond,
            )
            gamma_by_parameter = max_abs_off_diagonal_by_parameter(gamma)
            rho_by_parameter = max_abs_off_diagonal_by_parameter(rho)
            strict_positions = [
                pos
                for pos, index in enumerate(active)
                if parameter_names[index] not in protected
            ]
            high_gamma = any(
                gamma_by_parameter[pos]
                > self.max_sensitivity_collinearity_threshold
                for pos in strict_positions
            )
            high_rho = bool(
                self.apply_estimator_correlation_gate
                and any(
                    rho_by_parameter[pos]
                    > self.max_estimator_correlation_threshold
                    for pos in strict_positions
                )
            )
            all_ci95_ok = all(
                active_ci95[pos] <= self.max_relative_ci95
                or parameter_names[active[pos]] in protected
                for pos in range(n_active)
            )

            if (
                not rank_deficient
                and not ill_conditioned
                and all_ci95_ok
                and not high_gamma
                and not high_rho
            ):
                converged = True
                termination_reason = "criteria_satisfied"
                logger.info(
                    "Adaptive selection converged after %d iterations: "
                    "%d free parameters, kappa=%.2e, rank=%d/%d",
                    iteration,
                    n_active,
                    condition_number,
                    effective_rank,
                    n_active,
                )
                break

            removal_scores: list[tuple[float, int, int, str]] = []
            sensitivity_norms = np.linalg.norm(active_jacobian, axis=0)
            max_sens = (
                float(np.max(sensitivity_norms))
                if sensitivity_norms.size
                else 1.0
            )
            _, _, right_vectors = np.linalg.svd(
                active_jacobian, full_matrices=False
            )

            for pos, idx in enumerate(active):
                if parameter_names[idx] in protected:
                    continue
                score = 0.0
                reason_parts: list[str] = []
                ci_value = float(active_ci95[pos])
                if ci_value > self.max_relative_ci95:
                    score += 10.0 * (ci_value / self.max_relative_ci95)
                    reason_parts.append("high_ci95")
                if (
                    gamma_by_parameter[pos]
                    > self.max_sensitivity_collinearity_threshold
                ):
                    score += 2.5 * float(gamma_by_parameter[pos])
                    reason_parts.append("high_sensitivity_collinearity")
                if (
                    self.apply_estimator_correlation_gate
                    and rho_by_parameter[pos]
                    > self.max_estimator_correlation_threshold
                ):
                    score += 2.5 * float(rho_by_parameter[pos])
                    reason_parts.append("high_estimator_correlation")
                relative_sensitivity = (
                    sensitivity_norms[pos] / max_sens if max_sens > 0 else 0.0
                )
                score += 1.0 * (1.0 - relative_sensitivity)
                if relative_sensitivity < 0.01:
                    reason_parts.append("near_zero_sensitivity")
                if ill_conditioned and right_vectors.size:
                    alignment = abs(right_vectors[-1, pos])
                    score += 3.0 * alignment
                    if alignment > 0.3:
                        reason_parts.append("weak_singular_direction")
                reason = (
                    "+".join(reason_parts)
                    if reason_parts
                    else "lowest_information_score"
                )
                removal_scores.append((float(score), pos, idx, reason))

            if not removal_scores:
                termination_reason = "no_removable_nonprotected_parameter"
                break

            score, removal_pos, idx, removal_reason = min(
                removal_scores,
                key=lambda item: (-round(item[0], 12), item[2]),
            )
            active.pop(removal_pos)
            name = parameter_names[idx]
            fixed_values[name] = float(reference[idx])
            statuses[name] = "fixed_identifiability"
            reasons[name] = f"adaptive_{removal_reason}"
            ci95[name] = float(active_ci95[removal_pos])
            max_gamma[name] = float(gamma_by_parameter[removal_pos])
            max_rho[name] = float(rho_by_parameter[removal_pos])
            removal_trace.append(
                f"{iteration}:{name}:{removal_reason}:score={score:.12g}"
            )

        selected_jacobian = jacobian[:, active]
        selected_singular_values, reduced_condition_number = _spectrum(
            selected_jacobian
        )
        selected_ci95 = _relative_ci95(
            selected_jacobian,
            [specs[index] for index in active],
            reference[active],
            self.assumed_noise_level,
            self.covariance_rcond,
        )
        selected_gamma = sensitivity_collinearity_matrix(selected_jacobian)
        selected_rho = estimator_correlation_matrix(
            selected_jacobian,
            sigma2=self.assumed_noise_level**2,
            rcond=self.covariance_rcond,
        )
        selected_gamma_max = max_abs_off_diagonal_by_parameter(selected_gamma)
        selected_rho_max = max_abs_off_diagonal_by_parameter(selected_rho)
        strict_positions = [
            position
            for position, index in enumerate(active)
            if parameter_names[index] not in protected
        ]
        selected_effective_rank = _effective_rank(
            selected_singular_values, self.eigenvalue_gap_ratio
        )
        selected_min_rank = max(
            1, int(np.ceil(len(active) * self.min_rank_fraction))
        ) if active else 0
        final_criteria_satisfied = (
            selected_effective_rank >= selected_min_rank
            and reduced_condition_number <= self.max_condition_number
            and all(
                selected_gamma_max[position]
                <= self.max_sensitivity_collinearity_threshold
                for position in strict_positions
            )
            and (
                not self.apply_estimator_correlation_gate
                or all(
                    selected_rho_max[position]
                    <= self.max_estimator_correlation_threshold
                    for position in strict_positions
                )
            )
            and all(
                value <= self.max_relative_ci95
                for value, index in zip(selected_ci95, active, strict=True)
                if parameter_names[index] not in protected
            )
        )
        if final_criteria_satisfied:
            converged = True
            termination_reason = "criteria_satisfied"

        for position, index in enumerate(active):
            name = parameter_names[index]
            ci95[name] = float(selected_ci95[position])
            max_gamma[name] = float(selected_gamma_max[position])
            max_rho[name] = float(selected_rho_max[position])
            if name in protected and (
                selected_ci95[position] > self.max_relative_ci95
                or selected_gamma_max[position]
                > self.max_sensitivity_collinearity_threshold
                or (
                    self.apply_estimator_correlation_gate
                    and selected_rho_max[position]
                    > self.max_estimator_correlation_threshold
                )
            ):
                statuses[name] = "protected_high_uncertainty"
                reasons[name] = "protected_parameter_retained_despite_gate_limit"

        return ParameterSelection(
            parameter_names=parameter_names,
            reference_values=dict(zip(parameter_names, reference, strict=True)),
            free_names=tuple(parameter_names[index] for index in active),
            fixed_values=fixed_values,
            statuses=statuses,
            reasons=reasons,
            ci95_relative=ci95,
            original_singular_values=full_singular_values,
            selected_singular_values=selected_singular_values,
            original_condition_number=original_condition_number,
            reduced_condition_number=reduced_condition_number,
            assumed_noise_level=self.assumed_noise_level,
            max_condition_number=self.max_condition_number,
            max_relative_ci95=self.max_relative_ci95,
            protected_names=self.protected_names,
            max_sensitivity_collinearity=max_gamma,
            max_estimator_correlation=max_rho,
            max_sensitivity_collinearity_threshold=(
                self.max_sensitivity_collinearity_threshold
            ),
            max_estimator_correlation_threshold=(
                self.max_estimator_correlation_threshold
            ),
            estimator_correlation_is_gate_constraint=(
                self.apply_estimator_correlation_gate
            ),
            converged=converged,
            termination_reason=termination_reason,
            iterations=len(removal_trace),
            removal_trace=tuple(removal_trace),
        )


def expand_selection_with_low_confidence_parameters(
    selection: ParameterSelection,
    dataset: EISDataset,
    model: ForwardModel,
    parameter_specs: Sequence[ParameterSpec],
    theta_reference: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    *,
    max_condition_number: float = 1e7,
    max_relative_ci95: float = 1.5,
    max_sensitivity_collinearity: float = 0.9995,
    max_estimator_correlation: float = 0.9995,
    min_sensitivity_ratio: float = 1e-5,
    max_additions: int | None = None,
    prior_stabilized_regularization: float = 8.0,
    prior_stabilized_condition_number: float = 1e4,
    prior_stabilized_min_sensitivity_ratio: float = 1e-8,
    step_size: float = 1e-5,
    eps: float = 1e-12,
    covariance_rcond: float = 1e-14,
) -> ParameterSelection:
    """Greedily reintroduce weak parameters under a low-confidence budget.

    The strict selector answers: "which parameters are independently supported
    under the configured CI/conditioning thresholds?"  This helper answers a
    different, intentionally more aggressive question: "which additional
    parameters can be fitted without making the reduced problem numerically
    singular?"  Reintroduced parameters are marked
    ``estimated_low_confidence`` so downstream reports do not claim high
    confidence identifiability.
    """

    specs = tuple(parameter_specs)
    parameter_names = tuple(spec.name for spec in specs)
    if parameter_names != selection.parameter_names:
        raise ValueError("parameter_specs must match selection parameter order")
    reference = np.asarray(theta_reference, dtype=float)
    if reference.shape != (len(specs),):
        raise ValueError("theta_reference must match parameter_specs")
    if max_condition_number <= 0 or max_relative_ci95 <= 0:
        raise ValueError("low-confidence budgets must be positive")
    if not 0 < max_sensitivity_collinearity <= 1:
        raise ValueError("max_sensitivity_collinearity must be in (0, 1]")
    if not 0 < max_estimator_correlation <= 1:
        raise ValueError("max_estimator_correlation must be in (0, 1]")
    if min_sensitivity_ratio < 0:
        raise ValueError("min_sensitivity_ratio must be non-negative")
    if prior_stabilized_regularization < 0:
        raise ValueError("prior_stabilized_regularization must be non-negative")
    if prior_stabilized_condition_number <= 0:
        raise ValueError("prior_stabilized_condition_number must be positive")
    if prior_stabilized_min_sensitivity_ratio < 0:
        raise ValueError("prior_stabilized_min_sensitivity_ratio must be non-negative")

    jacobian = _relative_prediction_jacobian(
        dataset=dataset,
        model=model,
        specs=specs,
        theta_reference=reference,
        step_size=step_size,
        eps=eps,
        weights=weights,
    )
    name_to_index = {name: idx for idx, name in enumerate(parameter_names)}
    active = [name_to_index[name] for name in selection.free_names]
    remaining = [name_to_index[name] for name in selection.fixed_names]
    if not remaining:
        return selection

    sensitivity_norms = np.linalg.norm(jacobian, axis=0)
    max_sensitivity = float(np.max(sensitivity_norms)) if sensitivity_norms.size else 0.0
    if max_sensitivity <= 0:
        return selection

    added: list[int] = []
    addition_limit = len(remaining) if max_additions is None else max(0, max_additions)
    protected = set(selection.protected_names)

    while remaining and len(added) < addition_limit:
        best: tuple[float, int, FloatArray, FloatArray, float] | None = None
        for candidate in remaining:
            if sensitivity_norms[candidate] / max_sensitivity < min_sensitivity_ratio:
                continue
            trial = sorted(active + [candidate])
            trial_jacobian = jacobian[:, trial]
            singular_values, condition_number = _spectrum(trial_jacobian)
            if condition_number > max_condition_number:
                continue
            trial_ci95 = _relative_ci95(
                trial_jacobian,
                [specs[index] for index in trial],
                reference[trial],
                selection.assumed_noise_level,
                covariance_rcond,
            )
            ci_ok = all(
                ci <= max_relative_ci95 or parameter_names[index] in protected
                for ci, index in zip(trial_ci95, trial, strict=True)
            )
            if not ci_ok:
                continue
            candidate_position = trial.index(candidate)
            candidate_gamma, candidate_rho = _max_abs_parameter_diagnostics(
                trial_jacobian,
                candidate_position,
                sigma2=selection.assumed_noise_level**2,
                covariance_rcond=covariance_rcond,
            )
            if (
                candidate_gamma > max_sensitivity_collinearity
                or candidate_rho > max_estimator_correlation
            ):
                continue
            score = (
                condition_number / max_condition_number
                + trial_ci95[candidate_position] / max_relative_ci95
                + 0.5 * candidate_gamma
                + 0.5 * candidate_rho
                - 0.01 * sensitivity_norms[candidate] / max_sensitivity
            )
            if best is None or score < best[0]:
                best = (float(score), candidate, singular_values, trial_ci95, condition_number)

        if best is None:
            break
        _, candidate, _, _, _ = best
        active = sorted(active + [candidate])
        remaining.remove(candidate)
        added.append(candidate)

    prior_added: list[int] = []
    if prior_stabilized_regularization > 0 and remaining:
        while remaining:
            best_prior: tuple[float, int] | None = None
            for candidate in remaining:
                if (
                    sensitivity_norms[candidate] / max_sensitivity
                    < prior_stabilized_min_sensitivity_ratio
                ):
                    continue
                trial = sorted(active + [candidate])
                trial_jacobian = jacobian[:, trial]
                regularized_jacobian = np.vstack(
                    (
                        trial_jacobian,
                        np.diag(
                            np.full(
                                len(trial),
                                float(prior_stabilized_regularization),
                                dtype=float,
                            )
                        ),
                    )
                )
                _, regularized_condition = _spectrum(regularized_jacobian)
                if regularized_condition > prior_stabilized_condition_number:
                    continue
                candidate_position = trial.index(candidate)
                candidate_gamma, candidate_rho = _max_abs_parameter_diagnostics(
                    trial_jacobian,
                    candidate_position,
                    sigma2=selection.assumed_noise_level**2,
                    covariance_rcond=covariance_rcond,
                )
                score = (
                    regularized_condition / prior_stabilized_condition_number
                    + 0.5 * candidate_gamma
                    + 0.5 * candidate_rho
                    - 0.01 * sensitivity_norms[candidate] / max_sensitivity
                )
                if best_prior is None or score < best_prior[0]:
                    best_prior = (float(score), candidate)
            if best_prior is None:
                break
            _, candidate = best_prior
            active = sorted(active + [candidate])
            remaining.remove(candidate)
            prior_added.append(candidate)

    if not added and not prior_added:
        return selection

    active_jacobian = jacobian[:, active]
    selected_singular_values, reduced_condition_number = _spectrum(active_jacobian)
    selected_ci95 = _relative_ci95(
        active_jacobian,
        [specs[index] for index in active],
        reference[active],
        selection.assumed_noise_level,
        covariance_rcond,
    )
    fixed_values = dict(selection.fixed_values)
    statuses = dict(selection.statuses)
    reasons = dict(selection.reasons)
    ci95 = dict(selection.ci95_relative)
    max_gamma = dict(selection.max_sensitivity_collinearity)
    max_rho = dict(selection.max_estimator_correlation)
    active_gamma = sensitivity_collinearity_matrix(active_jacobian)
    active_rho = estimator_correlation_matrix(
        active_jacobian,
        sigma2=selection.assumed_noise_level**2,
        rcond=covariance_rcond,
    )
    active_gamma_max = max_abs_off_diagonal_by_parameter(active_gamma)
    active_rho_max = max_abs_off_diagonal_by_parameter(active_rho)

    for position, index in enumerate(active):
        name = parameter_names[index]
        ci95[name] = float(selected_ci95[position])
        max_gamma[name] = float(active_gamma_max[position])
        max_rho[name] = float(active_rho_max[position])
    for index in added:
        name = parameter_names[index]
        fixed_values.pop(name, None)
        statuses[name] = "estimated_low_confidence"
        reasons[name] = "reintroduced_under_low_confidence_condition_budget"
    for index in prior_added:
        name = parameter_names[index]
        fixed_values.pop(name, None)
        statuses[name] = "estimated_prior_stabilized"
        reasons[name] = "reintroduced_with_prior_stabilized_regularized_budget"

    return ParameterSelection(
        parameter_names=parameter_names,
        reference_values=dict(selection.reference_values),
        free_names=tuple(parameter_names[index] for index in active),
        fixed_values=fixed_values,
        statuses=statuses,
        reasons=reasons,
        ci95_relative=ci95,
        original_singular_values=selection.original_singular_values,
        selected_singular_values=selected_singular_values,
        original_condition_number=selection.original_condition_number,
        reduced_condition_number=reduced_condition_number,
        assumed_noise_level=selection.assumed_noise_level,
        max_condition_number=selection.max_condition_number,
        max_relative_ci95=selection.max_relative_ci95,
        protected_names=selection.protected_names,
        max_sensitivity_collinearity=max_gamma,
        max_estimator_correlation=max_rho,
        max_sensitivity_collinearity_threshold=(
            selection.max_sensitivity_collinearity_threshold
        ),
        max_estimator_correlation_threshold=(
            selection.max_estimator_correlation_threshold
        ),
        estimator_correlation_is_gate_constraint=(
            selection.estimator_correlation_is_gate_constraint
        ),
        converged=selection.converged,
        termination_reason="low_confidence_expansion",
        iterations=selection.iterations + len(added) + len(prior_added),
        removal_trace=selection.removal_trace,
    )


def _max_abs_parameter_diagnostics(
    jacobian: FloatArray,
    column_position: int,
    *,
    sigma2: float,
    covariance_rcond: float,
) -> tuple[float, float]:
    gamma = sensitivity_collinearity_matrix(jacobian)
    rho = estimator_correlation_matrix(
        jacobian, sigma2=sigma2, rcond=covariance_rcond
    )
    return (
        float(max_abs_off_diagonal_by_parameter(gamma)[column_position]),
        float(max_abs_off_diagonal_by_parameter(rho)[column_position]),
    )


@dataclass(frozen=True)
class ReducedParameterModel:
    """Expose selected free parameters while simulating the complete model."""

    full_model: ForwardModel
    selection: ParameterSelection

    def __post_init__(self) -> None:
        model_names = getattr(self.full_model, "parameter_names", None)
        if (
            model_names is not None
            and tuple(model_names) != self.selection.parameter_names
        ):
            raise ValueError("full model parameter order must match the selection")

    def expand_theta(self, theta_free: NDArray[np.floating]) -> FloatArray:
        """Expand physical free-parameter values to complete-model order."""

        free_values = np.asarray(theta_free, dtype=float)
        if free_values.shape != (len(self.selection.free_names),):
            raise ValueError("theta_free must match selected free parameter count")
        values = dict(self.selection.fixed_values)
        values.update(dict(zip(self.selection.free_names, free_values, strict=True)))
        return np.asarray(
            [values[name] for name in self.selection.parameter_names], dtype=float
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        """Simulate the full physical model from only selected free values."""

        return self.full_model.simulate(freq_hz, self.expand_theta(theta))


def _relative_prediction_jacobian(
    dataset: EISDataset,
    model: ForwardModel,
    specs: tuple[ParameterSpec, ...],
    theta_reference: FloatArray,
    step_size: float,
    eps: float,
    weights: NDArray[np.floating] | None = None,
) -> FloatArray:
    search_reference = encode_parameter_vector(specs, theta_reference)
    prediction_reference = model.simulate(dataset.freq_hz, theta_reference)
    scale = np.maximum(np.abs(prediction_reference), eps)
    sqrt_weights = _sqrt_observation_weights(weights, dataset.freq_hz.shape)

    def decode(sv: FloatArray) -> FloatArray:
        return decode_parameter_vector(specs, sv)

    def delta(search_values: FloatArray) -> FloatArray:
        prediction_delta = (
            model.simulate(dataset.freq_hz, decode(search_values))
            - prediction_reference
        ) / scale
        if sqrt_weights is not None:
            prediction_delta = prediction_delta * sqrt_weights
        return np.concatenate((prediction_delta.real, prediction_delta.imag))

    jacobian = np.empty((dataset.freq_hz.size * 2, len(specs)), dtype=float)
    for index in range(len(specs)):
        upper = search_reference.copy()
        lower = search_reference.copy()
        upper[index] += step_size
        lower[index] -= step_size
        jacobian[:, index] = (delta(upper) - delta(lower)) / (2.0 * step_size)
    return jacobian


def relative_prediction_jacobian(
    dataset: EISDataset,
    model: ForwardModel,
    specs: Sequence[ParameterSpec],
    theta_reference: NDArray[np.floating],
    *,
    step_size: float = 1e-5,
    eps: float = 1e-12,
    weights: NDArray[np.floating] | None = None,
) -> FloatArray:
    """Return the gate's weighted relative-prediction Jacobian.

    Matrix subset controls use this public wrapper so QR, FIM, and the
    identifiability gate consume the same local sensitivity matrix.
    """

    return _relative_prediction_jacobian(
        dataset=dataset,
        model=model,
        specs=tuple(specs),
        theta_reference=np.asarray(theta_reference, dtype=float),
        step_size=step_size,
        eps=eps,
        weights=weights,
    )


def _sqrt_observation_weights(
    weights: NDArray[np.floating] | None,
    expected_shape: tuple[int, ...],
) -> FloatArray | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float)
    if values.shape != expected_shape:
        raise ValueError("weights must have the same shape as dataset frequencies")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    return np.sqrt(values)


def _spectrum(jacobian: FloatArray) -> tuple[FloatArray, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return singular_values, _condition_number(singular_values)


def _condition_number(singular_values: FloatArray) -> float:
    if singular_values.size == 0 or singular_values[-1] <= 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _effective_rank(
    singular_values: FloatArray, gap_threshold: float = 100.0
) -> int:
    """Effective rank based on eigenvalue gap analysis.

    More conservative than numerical rank.  Scans singular values from
    largest to smallest; the effective rank is the position just before
    the first gap whose ratio exceeds *gap_threshold*.

    Parameters
    ----------
    singular_values
        Sorted (descending) singular values from an SVD.
    gap_threshold
        A consecutive ratio ``σ_i / σ_{i+1} > gap_threshold`` defines
        a gap.  Default 100.
    """

    values = np.asarray(singular_values, dtype=float)
    if values.size == 0:
        return 0
    # Drop numerically zero entries
    positive = values[values > 0]
    if positive.size <= 1:
        return int(positive.size)
    for k in range(positive.size - 1):
        ratio = positive[k] / positive[k + 1]
        if ratio > gap_threshold:
            return k + 1
    return int(positive.size)


def _relative_ci95(
    jacobian: FloatArray,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
    noise_level: float,
    covariance_rcond: float,
) -> FloatArray:
    information = jacobian.T @ jacobian
    covariance = noise_level**2 * np.linalg.pinv(
        information, rcond=covariance_rcond, hermitian=True
    )
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    relative_standard_deviation = np.asarray(
        [
            np.log(10.0) * deviation if spec.log_transform else deviation / abs(value)
            for spec, value, deviation in zip(
                specs, theta_reference, standard_deviation, strict=True
            )
        ],
        dtype=float,
    )
    return 1.96 * relative_standard_deviation
