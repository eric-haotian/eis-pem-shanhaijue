# learning AI at www.haotianblog.com
"""Auditable identifiability and uncertainty statistics.

This module keeps three quantities that are often conflated in EIS inverse
problems separate:

* sensitivity-column collinearity is the cosine between columns of the data
  residual Jacobian;
* estimator correlation is computed from the covariance implied by the
  inverse Fisher matrix;
* regularized curvature is reported separately from data-supported
  uncertainty and is never used as evidence that the data identify a
  parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_jacobian(jacobian: NDArray[np.floating]) -> FloatArray:
    values = np.asarray(jacobian, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("jacobian must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("jacobian must contain only finite values")
    return values


def sensitivity_collinearity_matrix(
    jacobian: NDArray[np.floating],
    eps: float = 1e-30,
) -> FloatArray:
    """Return Jacobian-column cosines ``gamma_ij``.

    This is a property of the local sensitivity directions. It is not a
    parameter-estimator correlation matrix.
    """

    values = _as_jacobian(jacobian)
    norms = np.linalg.norm(values, axis=0)
    denominator = np.outer(norms, norms)
    gram = values.T @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.divide(
            gram,
            denominator,
            out=np.zeros_like(gram, dtype=float),
            where=denominator > eps,
        )
    gamma = np.clip(gamma, -1.0, 1.0)
    np.fill_diagonal(gamma, np.where(norms > eps, 1.0, 0.0))
    return gamma


def estimator_covariance(
    jacobian: NDArray[np.floating],
    *,
    sigma2: float = 1.0,
    rcond: float = 1e-14,
) -> FloatArray:
    """Return ``sigma2 * pinv(J.T @ J)`` in the Jacobian coordinate system."""

    values = _as_jacobian(jacobian)
    if not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be finite and non-negative")
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be finite and positive")
    information = values.T @ values
    return float(sigma2) * np.linalg.pinv(
        information, rcond=rcond, hermitian=True
    )


def covariance_to_correlation(
    covariance: NDArray[np.floating],
    eps: float = 1e-30,
) -> FloatArray:
    """Normalize a covariance matrix while preserving covariance signs."""

    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("covariance must contain only finite values")
    variances = np.maximum(np.diag(values), 0.0)
    standard_deviations = np.sqrt(variances)
    denominator = np.outer(standard_deviations, standard_deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            values,
            denominator,
            out=np.zeros_like(values, dtype=float),
            where=denominator > eps,
        )
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, np.where(variances > eps, 1.0, 0.0))
    return correlation


def estimator_correlation_matrix(
    jacobian: NDArray[np.floating],
    *,
    sigma2: float = 1.0,
    rcond: float = 1e-14,
) -> FloatArray:
    """Return parameter-estimator correlation ``rho_ij``."""

    return covariance_to_correlation(
        estimator_covariance(jacobian, sigma2=sigma2, rcond=rcond)
    )


def max_abs_off_diagonal_by_parameter(matrix: NDArray[np.floating]) -> FloatArray:
    """Return the largest absolute off-diagonal entry for each parameter."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if values.shape[0] == 0:
        return np.array([], dtype=float)
    off_diagonal = np.abs(values.copy())
    np.fill_diagonal(off_diagonal, 0.0)
    return np.max(off_diagonal, axis=1)


def max_abs_off_diagonal(matrix: NDArray[np.floating]) -> float:
    """Return the largest absolute off-diagonal matrix entry."""

    per_parameter = max_abs_off_diagonal_by_parameter(matrix)
    return float(np.max(per_parameter)) if per_parameter.size else 0.0


@dataclass(frozen=True)
class TruthBasedIdentifiabilityMetrics:
    """Truth-based classification metrics for a synthetic benchmark.

    ``false_identifiability_rate`` and ``strict_precision`` use the number of
    parameters declared strict as denominator. If no parameter is declared
    strict, both are ``NaN``. ``identifiable_parameter_recall`` uses all fitted
    candidate parameters satisfying the frozen truth-error tolerance as its
    denominator. Protected and fixed parameters are never counted as strict.
    """

    parameter_names: tuple[str, ...]
    relative_errors: dict[str, float]
    free_names: tuple[str, ...]
    strict_names: tuple[str, ...]
    protected_names: tuple[str, ...]
    fixed_names: tuple[str, ...]
    error_tolerance: float
    false_identifiable_count: int
    false_identifiability_rate: float
    strict_precision: float
    identifiable_parameter_recall: float
    false_negative_rate: float
    truth_accurate_count: int

    @property
    def free_count(self) -> int:
        return len(self.free_names)

    @property
    def strict_count(self) -> int:
        return len(self.strict_names)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "parameter_error_tolerance": self.error_tolerance,
            "free_count": self.free_count,
            "strict_count": self.strict_count,
            "protected_count": len(self.protected_names),
            "fixed_count": len(self.fixed_names),
            "truth_accurate_count": self.truth_accurate_count,
            "false_identifiable_count": self.false_identifiable_count,
            "false_identifiability_rate": self.false_identifiability_rate,
            "strict_precision": self.strict_precision,
            "identifiable_parameter_recall": self.identifiable_parameter_recall,
            "false_negative_rate": self.false_negative_rate,
        }


def truth_based_identifiability_metrics(
    parameter_names: Sequence[str],
    estimates: Mapping[str, float],
    truth: Mapping[str, float],
    *,
    free_names: Sequence[str],
    strict_names: Sequence[str],
    protected_names: Sequence[str] = (),
    fixed_names: Sequence[str] = (),
    error_tolerance: float = 0.10,
    eps: float = 1e-30,
) -> TruthBasedIdentifiabilityMetrics:
    """Evaluate strict-label precision/recall against frozen synthetic truth."""

    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and non-empty")
    if not np.isfinite(error_tolerance) or error_tolerance <= 0:
        raise ValueError("error_tolerance must be finite and positive")
    name_set = set(names)
    raw_subsets = (
        ("free_names", tuple(free_names)),
        ("strict_names", tuple(strict_names)),
        ("protected_names", tuple(protected_names)),
        ("fixed_names", tuple(fixed_names)),
    )
    for label, subset in raw_subsets:
        if len(set(subset)) != len(subset):
            raise ValueError(f"{label} must not contain duplicates")
        unknown = set(subset).difference(name_set)
        if unknown:
            raise ValueError(f"{label} contains unknown parameters: {sorted(unknown)}")
    raw = {label: set(subset) for label, subset in raw_subsets}
    free = tuple(name for name in names if name in raw["free_names"])
    strict = tuple(name for name in names if name in raw["strict_names"])
    protected = tuple(name for name in names if name in raw["protected_names"])
    fixed = tuple(name for name in names if name in raw["fixed_names"])
    if set(free).intersection(fixed):
        raise ValueError("free and fixed parameters must be disjoint")
    if set(free).union(fixed) != name_set:
        raise ValueError("free and fixed parameters must partition parameter_names")
    if not set(strict).issubset(free):
        raise ValueError("strict parameters must be a subset of free parameters")
    if not set(protected).issubset(free):
        raise ValueError("protected parameters must be a subset of free parameters")
    if set(strict).intersection(protected):
        raise ValueError("strict and protected parameters must be disjoint")
    if set(strict).intersection(fixed):
        raise ValueError("fixed parameters cannot be declared strict")

    relative_errors: dict[str, float] = {}
    for name in names:
        if name not in truth:
            raise ValueError(f"truth is missing parameter {name!r}")
        estimate = estimates.get(name, np.nan)
        reference = float(truth[name])
        if not np.isfinite(estimate) or not np.isfinite(reference):
            relative_errors[name] = float("nan")
        else:
            relative_errors[name] = float(
                abs(float(estimate) - reference) / max(abs(reference), eps)
            )

    truth_accurate = {
        name
        for name in free
        if np.isfinite(relative_errors[name])
        and relative_errors[name] <= error_tolerance
    }
    strict_set = set(strict)
    false_strict = {
        name
        for name in strict
        if not np.isfinite(relative_errors[name])
        or relative_errors[name] > error_tolerance
    }
    strict_true_positive = strict_set.intersection(truth_accurate)
    false_negative = truth_accurate.difference(strict_set)

    strict_count = len(strict)
    truth_accurate_count = len(truth_accurate)
    false_rate = (
        len(false_strict) / strict_count if strict_count else float("nan")
    )
    precision = (
        len(strict_true_positive) / strict_count if strict_count else float("nan")
    )
    recall = (
        len(strict_true_positive) / truth_accurate_count
        if truth_accurate_count
        else float("nan")
    )
    false_negative_rate = (
        len(false_negative) / truth_accurate_count
        if truth_accurate_count
        else float("nan")
    )
    return TruthBasedIdentifiabilityMetrics(
        parameter_names=names,
        relative_errors=relative_errors,
        free_names=free,
        strict_names=strict,
        protected_names=protected,
        fixed_names=fixed,
        error_tolerance=float(error_tolerance),
        false_identifiable_count=len(false_strict),
        false_identifiability_rate=float(false_rate),
        strict_precision=float(precision),
        identifiable_parameter_recall=float(recall),
        false_negative_rate=float(false_negative_rate),
        truth_accurate_count=truth_accurate_count,
    )


def robust_effective_row_weights(
    residual_vector: NDArray[np.floating],
    *,
    loss: str,
    f_scale: float,
) -> FloatArray:
    """Return square-root IRLS row weights for supported SciPy losses."""

    residual = np.asarray(residual_vector, dtype=float)
    if residual.ndim != 1 or not np.all(np.isfinite(residual)):
        raise ValueError("residual_vector must be a finite one-dimensional array")
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("f_scale must be finite and positive")
    z = (residual / f_scale) ** 2
    if loss == "linear":
        rho_prime = np.ones_like(z)
    elif loss == "soft_l1":
        rho_prime = 1.0 / np.sqrt(1.0 + z)
    elif loss == "huber":
        rho_prime = np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(z))
    elif loss == "cauchy":
        rho_prime = 1.0 / (1.0 + z)
    elif loss == "arctan":
        rho_prime = 1.0 / (1.0 + z**2)
    else:
        raise ValueError(f"unsupported robust loss: {loss!r}")
    return np.sqrt(np.maximum(rho_prime, 0.0))


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """Separate data, robust-effective, and prior-regularized uncertainty."""

    data_only_covariance: FloatArray
    robust_effective_covariance: FloatArray
    sandwich_covariance: FloatArray
    regularized_estimator_covariance: FloatArray | None
    data_only_correlation: FloatArray
    robust_effective_correlation: FloatArray
    sigma2: float
    degrees_of_freedom: int
    sigma_mode: str
    loss: str
    boundary_hit: NDArray[np.bool_] | None


def uncertainty_decomposition(
    data_jacobian: NDArray[np.floating],
    residual_vector: NDArray[np.floating],
    *,
    loss: str = "linear",
    f_scale: float = 1.0,
    known_sigma: float | None = None,
    prior_precision: NDArray[np.floating] | None = None,
    parameter_values: NDArray[np.floating] | None = None,
    parameter_bounds: NDArray[np.floating] | None = None,
    boundary_tolerance: float = 1e-6,
    rcond: float = 1e-14,
) -> UncertaintyDecomposition:
    """Compute uncertainty without treating prior curvature as data evidence.

    The data-only covariance uses the unmodified residual Jacobian. The
    robust-effective covariance applies IRLS weights implied by ``loss``. The
    optional regularized covariance adds prior curvature only in a separately
    named field. A heteroscedasticity-consistent sandwich covariance is also
    returned for the robust effective Jacobian.
    """

    jacobian = _as_jacobian(data_jacobian)
    residual = np.asarray(residual_vector, dtype=float)
    if residual.shape != (jacobian.shape[0],) or not np.all(np.isfinite(residual)):
        raise ValueError("residual_vector must match the Jacobian row count")
    dof = max(jacobian.shape[0] - jacobian.shape[1], 1)
    if known_sigma is None:
        sigma2 = float(np.sum(residual**2) / dof)
        sigma_mode = "residual_estimated_with_dof_correction"
    else:
        if not np.isfinite(known_sigma) or known_sigma < 0:
            raise ValueError("known_sigma must be finite and non-negative")
        sigma2 = float(known_sigma**2)
        sigma_mode = "known_sigma"

    data_cov = estimator_covariance(jacobian, sigma2=sigma2, rcond=rcond)
    row_weights = robust_effective_row_weights(
        residual, loss=loss, f_scale=f_scale
    )
    effective_jacobian = jacobian * row_weights[:, None]
    effective_residual = residual * row_weights
    effective_sigma2 = (
        float(np.sum(effective_residual**2) / dof)
        if known_sigma is None
        else sigma2
    )
    robust_cov = estimator_covariance(
        effective_jacobian, sigma2=effective_sigma2, rcond=rcond
    )

    bread = np.linalg.pinv(
        effective_jacobian.T @ effective_jacobian,
        rcond=rcond,
        hermitian=True,
    )
    meat = effective_jacobian.T @ (
        effective_residual[:, None] ** 2 * effective_jacobian
    )
    sandwich = bread @ meat @ bread
    sandwich *= jacobian.shape[0] / dof

    regularized = None
    if prior_precision is not None:
        prior = np.asarray(prior_precision, dtype=float)
        if prior.shape != (jacobian.shape[1], jacobian.shape[1]):
            raise ValueError("prior_precision must be square with one row per parameter")
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior_precision must be finite")
        data_information = effective_jacobian.T @ effective_jacobian / max(
            effective_sigma2, 1e-30
        )
        regularized = np.linalg.pinv(
            data_information + prior, rcond=rcond, hermitian=True
        )

    boundary_hit = None
    if parameter_values is not None or parameter_bounds is not None:
        if parameter_values is None or parameter_bounds is None:
            raise ValueError("parameter_values and parameter_bounds must be supplied together")
        values = np.asarray(parameter_values, dtype=float)
        bounds = np.asarray(parameter_bounds, dtype=float)
        if values.shape != (jacobian.shape[1],) or bounds.shape != (
            jacobian.shape[1],
            2,
        ):
            raise ValueError("parameter values/bounds must match the Jacobian columns")
        span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-30)
        boundary_hit = (
            (values - bounds[:, 0] <= boundary_tolerance * span)
            | (bounds[:, 1] - values <= boundary_tolerance * span)
        )

    return UncertaintyDecomposition(
        data_only_covariance=data_cov,
        robust_effective_covariance=robust_cov,
        sandwich_covariance=sandwich,
        regularized_estimator_covariance=regularized,
        data_only_correlation=covariance_to_correlation(data_cov),
        robust_effective_correlation=covariance_to_correlation(robust_cov),
        sigma2=sigma2,
        degrees_of_freedom=dof,
        sigma_mode=sigma_mode,
        loss=loss,
        boundary_hit=boundary_hit,
    )
