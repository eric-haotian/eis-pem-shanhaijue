# learning AI at www.haotianblog.com
"""Optimisation wrappers for PEM parameter identification."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares

from .costs import EISPredictionErrorCost
from .dataset import EISDataset
from .forward_models import ForwardModel
from .parameters import (
    ParameterSpec,
    decode_parameter_vector,
    encode_parameter_vector,
    initial_optimization_vector,
    optimization_bounds_array,
)
from .physics_priors import PhysicsPrior, prior_penalty_vector
from .results import IdentificationResult

if TYPE_CHECKING:
    from .measurements import ParameterMeasurementDataset

logger = logging.getLogger(__name__)

SCIPY_LEAST_SQUARES_LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")


def _validate_least_squares_loss(loss: str, f_scale: float) -> None:
    if loss not in SCIPY_LEAST_SQUARES_LOSSES:
        raise ValueError(
            f"loss must be one of {list(SCIPY_LEAST_SQUARES_LOSSES)}, got {loss!r}"
        )
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("f_scale must be finite and positive")


@dataclass(frozen=True)
class DifferentialEvolutionOptimizer:
    """Bounded global optimiser for a complex-impedance PEM cost."""

    seed: int = 42
    workers: int = 1
    polish: bool = True
    popsize: int = 15
    maxiter: int = 1000
    tol: float = 1e-8
    relative: bool = False
    eps: float = 1e-12

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        priors: Sequence[PhysicsPrior] = (),
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("parameter names must be unique")

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            eps=self.eps,
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return decode_parameter_vector(specs, sv)

        def objective(search_values: NDArray[np.floating]) -> float:
            theta_vector = decode(search_values)
            base_cost = cost(theta_vector)
            if priors:
                penalties = prior_penalty_vector(
                    priors, [spec.name for spec in specs], theta_vector
                )
                return base_cost + float(np.sum(penalties**2))
            return base_cost

        scipy_result = differential_evolution(
            objective,
            bounds=[tuple(bounds) for bounds in optimization_bounds_array(specs)],
            seed=self.seed,
            workers=self.workers,
            updating="immediate" if self.workers == 1 else "deferred",
            polish=self.polish,
            popsize=self.popsize,
            maxiter=self.maxiter,
            tol=self.tol,
        )
        theta_vector = decode(scipy_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        theta_best = {
            spec.name: float(value)
            for spec, value in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "differential_evolution",
            "seed": self.seed,
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "nfev": int(scipy_result.nfev),
            "nit": int(scipy_result.nit),
            "search_space": {
                spec.name: "log10" if spec.log_transform else "linear" for spec in specs
            },
            "relative_residuals": self.relative,
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=float(scipy_result.fun),
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


@dataclass(frozen=True)
class LeastSquaresOptimizer:
    """Bounded local PEM optimiser suitable for higher-dimensional models."""

    relative: bool = False
    eps: float = 1e-12
    max_nfev: int = 20000
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    modulus_weighting: bool = False
    n_starts: int = 1
    loss: str = "linear"
    f_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_least_squares_loss(self.loss, self.f_scale)
        if self.n_starts < 1:
            raise ValueError("n_starts must be >= 1")

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        auxiliary_measurements: ParameterMeasurementDataset | None = None,
        priors: Sequence[PhysicsPrior] = (),
        initial_guess: dict[str, float] | None = None,
        regularization_weights: dict[str, float] | None = None,
        regularization_reference: dict[str, float] | None = None,
        regularization_matrix: NDArray[np.floating] | None = None,
        regularization_matrix_reference: NDArray[np.floating] | None = None,
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("parameter names must be unique")
        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )
        regularization_weights_vector, regularization_reference_vector = (
            _regularization_setup(
                specs, regularization_weights, regularization_reference
            )
        )
        matrix, matrix_reference = _matrix_regularization_setup(
            specs, regularization_matrix, regularization_matrix_reference
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return decode_parameter_vector(specs, sv)

        def residual_vector(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
            theta_vector = decode(search_values)
            residuals = cost.residuals(theta_vector)
            vector = np.concatenate((residuals.real, residuals.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate(
                    (
                        vector,
                        auxiliary_measurements.relative_residuals(
                            tuple(spec.name for spec in specs), theta_vector
                        ),
                    ),
                )
            if priors:
                vector = np.concatenate(
                    (
                        vector,
                        prior_penalty_vector(
                            priors, [spec.name for spec in specs], theta_vector
                        ),
                    )
                )
            if (
                regularization_weights_vector is not None
                and regularization_reference_vector is not None
            ):
                vector = np.concatenate(
                    (
                        vector,
                        regularization_weights_vector
                        * (search_values - regularization_reference_vector),
                    )
                )
            if matrix is not None and matrix_reference is not None:
                vector = np.concatenate(
                    (vector, matrix @ (search_values - matrix_reference))
                )
            return vector

        initial_values = np.array([s.initial_value for s in specs], dtype=float)
        if initial_guess is not None:
            for i, s in enumerate(specs):
                if s.name in initial_guess:
                    val = initial_guess[s.name]
                    val = max(s.bounds[0], min(s.bounds[1], val))
                    if s.log_transform and val <= 0:
                        val = s.initial_value
                    initial_values[i] = val

        initial = initial_values.copy()
        log_mask = np.array([s.log_transform for s in specs])
        initial[log_mask] = np.log10(initial[log_mask])
        bounds = np.array([s.optimization_bounds for s in specs], dtype=float)
        # Clamp initial to bounds
        initial = np.clip(initial, bounds[:, 0], bounds[:, 1])

        # Generate multi-start initial points
        start_points = [initial]
        if self.n_starts > 1:
            rng = np.random.default_rng(seed=0)
            extra = rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(self.n_starts - 1, len(specs))
            )
            start_points.extend(extra[i] for i in range(extra.shape[0]))

        best_result = None
        best_cost_value = float("inf")

        for x0 in start_points:
            scipy_result = least_squares(
                residual_vector,
                x0=x0,
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                loss=self.loss,
                f_scale=self.f_scale,
            )
            cost_value = 2.0 * float(scipy_result.cost)
            if cost_value < best_cost_value:
                best_cost_value = cost_value
                best_result = scipy_result

        scipy_result = best_result
        theta_vector = decode(scipy_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        theta_best = {
            spec.name: float(value)
            for spec, value in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "least_squares",
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "nfev": int(scipy_result.nfev),
            "cost": float(scipy_result.cost),
            "optimality": float(scipy_result.optimality),
            "search_space": {
                spec.name: "log10" if spec.log_transform else "linear" for spec in specs
            },
            "relative_residuals": self.relative,
            "modulus_weighting": self.modulus_weighting,
            "n_starts": self.n_starts,
            "loss": self.loss,
            "f_scale": self.f_scale,
            "auxiliary_measurement_count": (
                0
                if auxiliary_measurements is None
                else len(auxiliary_measurements.parameter_names)
            ),
            "selective_regularization_count": (
                0
                if regularization_weights_vector is None
                else int(np.count_nonzero(regularization_weights_vector))
            ),
            "selective_regularization_max": (
                0.0
                if regularization_weights_vector is None
                else float(np.max(regularization_weights_vector))
            ),
            "matrix_regularization_rows": (
                0 if matrix is None else int(matrix.shape[0])
            ),
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=best_cost_value,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# New optimizers for real-data robustness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridOptimizer:
    """Two-stage optimizer: global (DE) exploration + local (TRF) refinement.

    Stage 1 runs Differential Evolution with a modest budget to identify
    the basin of attraction, then Stage 2 refines with bounded least
    squares from the DE optimum.  This avoids getting trapped in local
    minima while still achieving high-precision convergence.
    """

    # DE stage settings
    de_seed: int = 42
    de_popsize: int = 10
    de_maxiter: int = 200
    de_tol: float = 1e-6
    de_workers: int = 1

    # Local stage settings
    ls_max_nfev: int = 20000
    ls_ftol: float = 1e-10
    ls_xtol: float = 1e-10
    ls_gtol: float = 1e-10

    # Shared settings
    relative: bool = True
    eps: float = 1e-12
    modulus_weighting: bool = False
    loss: str = "linear"
    f_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_least_squares_loss(self.loss, self.f_scale)

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        auxiliary_measurements: ParameterMeasurementDataset | None = None,
        priors: Sequence[PhysicsPrior] = (),
        initial_guess: dict[str, float] | None = None,
        regularization_weights: dict[str, float] | None = None,
        regularization_reference: dict[str, float] | None = None,
    ) -> IdentificationResult:
        """Run DE for global search, then TRF for local refinement."""

        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return decode_parameter_vector(specs, sv)

        reg_weights, reg_reference = _regularization_setup(
            specs, regularization_weights, regularization_reference
        )

        def scalar_cost(sv: NDArray[np.floating]) -> float:
            theta = decode(sv)
            base_cost = cost(theta)
            if priors:
                penalties = prior_penalty_vector(
                    priors, [spec.name for spec in specs], theta
                )
                base_cost += float(np.sum(penalties**2))
            if reg_weights is not None and reg_reference is not None:
                regularization = reg_weights * (sv - reg_reference)
                base_cost += float(np.sum(regularization**2))
            return base_cost

        def residual_vector(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            theta = decode(sv)
            residuals = cost.residuals(theta)
            vector = np.concatenate((residuals.real, residuals.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate((
                    vector,
                    auxiliary_measurements.relative_residuals(
                        tuple(s.name for s in specs), theta
                    ),
                ))
            if priors:
                vector = np.concatenate(
                    (
                        vector,
                        prior_penalty_vector(
                            priors, [spec.name for spec in specs], theta
                        ),
                    )
                )
            if reg_weights is not None and reg_reference is not None:
                vector = np.concatenate((vector, reg_weights * (sv - reg_reference)))
            return vector

        bounds_array = optimization_bounds_array(specs)
        bounds_list = [tuple(bounds) for bounds in bounds_array]

        # Build initial point from guess (if available) to seed DE
        x0 = None
        if initial_guess is not None:
            initial_values = np.array([s.initial_value for s in specs], dtype=float)
            for i, s in enumerate(specs):
                if s.name in initial_guess:
                    val = initial_guess[s.name]
                    val = max(s.bounds[0], min(s.bounds[1], val))
                    if s.log_transform and val <= 0:
                        val = s.initial_value
                    initial_values[i] = val
            x0 = np.array([
                np.log10(v) if s.log_transform else v
                for s, v in zip(specs, initial_values)
            ], dtype=float)
            x0 = np.clip(x0, bounds_array[:, 0], bounds_array[:, 1])

        # Stage 1: Global search with DE
        logger.info("HybridOptimizer Stage 1: Differential Evolution")
        de_result = differential_evolution(
            scalar_cost,
            bounds=bounds_list,
            seed=self.de_seed,
            workers=self.de_workers,
            updating="immediate" if self.de_workers == 1 else "deferred",
            polish=False,  # We do our own local refinement
            popsize=self.de_popsize,
            maxiter=self.de_maxiter,
            tol=self.de_tol,
            x0=x0,
        )
        logger.info(
            "DE converged: cost=%.6e, nfev=%d", de_result.fun, de_result.nfev
        )

        # Stage 2: Local refinement with TRF
        logger.info("HybridOptimizer Stage 2: Local TRF refinement")
        ls_result = least_squares(
            residual_vector,
            x0=de_result.x,
            bounds=(bounds_array[:, 0], bounds_array[:, 1]),
            max_nfev=self.ls_max_nfev,
            ftol=self.ls_ftol,
            xtol=self.ls_xtol,
            gtol=self.ls_gtol,
            loss=self.loss,
            f_scale=self.f_scale,
        )

        theta_vector = decode(ls_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        final_cost = 2.0 * float(ls_result.cost)

        theta_best = {
            spec.name: float(v)
            for spec, v in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "hybrid_de_trf",
            "de_nfev": int(de_result.nfev),
            "de_cost": float(de_result.fun),
            "de_success": bool(de_result.success),
            "ls_nfev": int(ls_result.nfev),
            "ls_cost": float(ls_result.cost),
            "ls_success": bool(ls_result.success),
            "ls_message": str(ls_result.message),
            "total_nfev": int(de_result.nfev + ls_result.nfev),
            "relative_residuals": self.relative,
            "loss": self.loss,
            "f_scale": self.f_scale,
            "selective_regularization_count": (
                0 if reg_weights is None else int(np.count_nonzero(reg_weights))
            ),
            "selective_regularization_max": (
                0.0 if reg_weights is None else float(np.max(reg_weights))
            ),
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=final_cost,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


@dataclass(frozen=True)
class AdaptiveLeastSquaresOptimizer:
    """Least-squares optimizer with LHS multi-start and Tikhonov regularization.

    Improvements over :class:`LeastSquaresOptimizer`:

    1. **Latin Hypercube Sampling (LHS)** for multi-start initial points
       instead of uniform random, giving better coverage of the search space.
    2. **Tikhonov regularization** (``alpha``) adds a small penalty
       ``α²‖x - x₀‖²`` to the residual vector, preventing over-fitting
       and improving conditioning for noisy real data.
    3. **Post-fit quality flag** that checks whether the fit is trustworthy.
    """

    relative: bool = True
    eps: float = 1e-12
    max_nfev: int = 20000
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    modulus_weighting: bool = False
    n_starts: int = 8
    alpha: float = 0.0  # Tikhonov regularization strength
    seed: int = 42
    loss: str = "linear"
    f_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_least_squares_loss(self.loss, self.f_scale)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.n_starts < 1:
            raise ValueError("n_starts must be >= 1")

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        auxiliary_measurements: ParameterMeasurementDataset | None = None,
        priors: Sequence[PhysicsPrior] = (),
        initial_guess: dict[str, float] | None = None,
        regularization_weights: dict[str, float] | None = None,
        regularization_reference: dict[str, float] | None = None,
        regularization_matrix: NDArray[np.floating] | None = None,
        regularization_matrix_reference: NDArray[np.floating] | None = None,
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return decode_parameter_vector(specs, sv)

        initial_values = np.array([s.initial_value for s in specs], dtype=float)
        if initial_guess is not None:
            for i, s in enumerate(specs):
                if s.name in initial_guess:
                    val = initial_guess[s.name]
                    # Clamp to physical bounds
                    val = max(s.bounds[0], min(s.bounds[1], val))
                    # For log-transformed params, ensure positive
                    if s.log_transform and val <= 0:
                        val = s.initial_value
                    initial_values[i] = val

        # Build initial point in search space
        initial = np.array([
            np.log10(v) if s.log_transform else v
            for s, v in zip(specs, initial_values)
        ], dtype=float)
        bounds = optimization_bounds_array(specs)
        selective_weights, selective_reference = _regularization_setup(
            specs, regularization_weights, regularization_reference
        )
        matrix, matrix_reference = _matrix_regularization_setup(
            specs, regularization_matrix, regularization_matrix_reference
        )

        # Clamp initial to bounds (safety)
        initial = np.clip(initial, bounds[:, 0], bounds[:, 1])

        def residual_vector(
            sv: NDArray[np.floating],
        ) -> NDArray[np.float64]:
            theta = decode(sv)
            r = cost.residuals(theta)
            vector = np.concatenate((r.real, r.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate((
                    vector,
                    auxiliary_measurements.relative_residuals(
                        tuple(s.name for s in specs), theta
                    ),
                ))
            if priors:
                vector = np.concatenate(
                    (
                        vector,
                        prior_penalty_vector(
                            priors, [spec.name for spec in specs], theta
                        ),
                    )
                )
            # Tikhonov regularization: append α * (x - x0)
            if self.alpha > 0:
                vector = np.concatenate((
                    vector, self.alpha * (sv - initial)
                ))
            if selective_weights is not None and selective_reference is not None:
                vector = np.concatenate(
                    (vector, selective_weights * (sv - selective_reference))
                )
            if matrix is not None and matrix_reference is not None:
                vector = np.concatenate((vector, matrix @ (sv - matrix_reference)))
            return vector

        residual_callable = residual_vector
        jacobian_callable: str | object = "2-point"
        jax_functions = None
        if auxiliary_measurements is None and not priors:
            from .jax_optim import (
                build_jax_least_squares_functions,
                supports_jax,
            )

            if supports_jax(model):
                jax_functions = build_jax_least_squares_functions(
                    model=model,
                    freq_hz=dataset.freq_hz,
                    z_obs=dataset.z_obs,
                    specs=specs,
                    relative=self.relative,
                    eps=self.eps,
                    weights=cost.weights,
                    alpha=self.alpha,
                    alpha_reference=initial,
                    regularization_weights=selective_weights,
                    regularization_reference=selective_reference,
                )
                residual_callable = jax_functions.residual
                jacobian_callable = jax_functions.jacobian

        # Generate LHS start points
        start_points = _latin_hypercube_starts(
            initial, bounds, self.n_starts, self.seed
        )

        best_result = None
        best_cost_value = float("inf")

        for i, x0 in enumerate(start_points):
            try:
                result = least_squares(
                    residual_callable,
                    x0=x0,
                    jac=jacobian_callable,
                    bounds=(bounds[:, 0], bounds[:, 1]),
                    max_nfev=self.max_nfev,
                    ftol=self.ftol,
                    xtol=self.xtol,
                    gtol=self.gtol,
                    loss=self.loss,
                    f_scale=self.f_scale,
                )
                cost_val = 2.0 * float(result.cost)
                if cost_val < best_cost_value:
                    best_cost_value = cost_val
                    best_result = result
                    logger.debug(
                        "Start %d/%d: cost=%.6e (new best)",
                        i + 1, self.n_starts, cost_val,
                    )
            except Exception as exc:
                logger.warning(
                    "Start %d/%d failed: %s", i + 1, self.n_starts, exc
                )

        if best_result is None:
            raise RuntimeError("All optimizer starts failed")

        theta_vector = decode(best_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit

        theta_best = {
            spec.name: float(v)
            for spec, v in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "adaptive_least_squares",
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "nfev": int(best_result.nfev),
            "cost": float(best_result.cost),
            "optimality": float(best_result.optimality),
            "n_starts": self.n_starts,
            "seed": self.seed,
            "alpha": self.alpha,
            "relative_residuals": self.relative,
            "modulus_weighting": self.modulus_weighting,
            "loss": self.loss,
            "f_scale": self.f_scale,
            "selective_regularization_count": (
                0
                if selective_weights is None
                else int(np.count_nonzero(selective_weights))
            ),
            "selective_regularization_max": (
                0.0
                if selective_weights is None
                else float(np.max(selective_weights))
            ),
            "matrix_regularization_rows": (
                0 if matrix is None else int(matrix.shape[0])
            ),
            "jacobian_backend": (
                "jax_exact" if jax_functions is not None else "scipy_2-point"
            ),
            "jax_platform": (
                None if jax_functions is None else jax_functions.platform
            ),
            "jax_device": None if jax_functions is None else jax_functions.device,
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=best_cost_value,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


def _latin_hypercube_starts(
    initial: NDArray[np.float64],
    bounds: NDArray[np.float64],
    n_starts: int,
    seed: int,
) -> list[NDArray[np.float64]]:
    """Generate LHS-distributed initial points for multi-start optimization.

    The first point is always the provided *initial* guess.  Remaining
    points are sampled using Latin Hypercube Sampling within the parameter
    bounds to maximize coverage.

    Parameters
    ----------
    initial
        Default initial point in search space.
    bounds
        (n_params, 2) array of [lower, upper] bounds.
    n_starts
        Total number of start points to generate.
    seed
        Random seed for reproducibility.
    """

    points = [initial.copy()]
    if n_starts <= 1:
        return points

    rng = np.random.default_rng(seed)
    ndim = len(initial)
    n_samples = n_starts - 1

    # LHS: vectorized interval sampling + per-column shuffle
    low, high = bounds[:, 0], bounds[:, 1]
    edges_frac = np.linspace(0.0, 1.0, n_samples + 1)  # (n_samples+1,)
    # Random point within each interval for all dims at once
    frac = edges_frac[:-1, np.newaxis] + rng.uniform(size=(n_samples, ndim)) / n_samples
    lhs_samples = low + frac * (high - low)  # (n_samples, ndim)
    # Shuffle each column independently
    for j in range(ndim):
        rng.shuffle(lhs_samples[:, j])

    points.extend(lhs_samples[i] for i in range(n_samples))
    return points


def _regularization_setup(
    specs: Sequence[ParameterSpec],
    regularization_weights: dict[str, float] | None,
    regularization_reference: dict[str, float] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Build search-space vectors for selective parameter regularization."""

    if not regularization_weights:
        return None, None
    spec_tuple = tuple(specs)
    names = tuple(spec.name for spec in spec_tuple)
    unknown = set(regularization_weights).difference(names)
    if unknown:
        raise ValueError(
            f"regularization weights include unknown parameters: {sorted(unknown)}"
        )
    weights = np.asarray(
        [float(regularization_weights.get(spec.name, 0.0)) for spec in spec_tuple],
        dtype=float,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("regularization weights must be finite and non-negative")
    if not np.any(weights > 0):
        return None, None

    reference_values = np.asarray([spec.initial_value for spec in spec_tuple], dtype=float)
    if regularization_reference:
        for index, spec in enumerate(spec_tuple):
            if spec.name not in regularization_reference:
                continue
            value = float(regularization_reference[spec.name])
            lower, upper = spec.bounds
            value = min(max(value, lower), upper)
            if spec.log_transform and value <= 0:
                value = spec.initial_value
            reference_values[index] = value

    reference = encode_parameter_vector(spec_tuple, reference_values)
    return weights, reference


def _matrix_regularization_setup(
    specs: Sequence[ParameterSpec],
    regularization_matrix: NDArray[np.floating] | None,
    regularization_reference: NDArray[np.floating] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate a full search-space regularization residual factor."""

    matrix = (
        None
        if regularization_matrix is None
        else np.asarray(regularization_matrix, dtype=float)
    )
    reference = (
        None
        if regularization_reference is None
        else np.asarray(regularization_reference, dtype=float)
    )
    if (matrix is None) != (reference is None):
        raise ValueError(
            "regularization_matrix and its reference must be supplied together"
        )
    if matrix is None or reference is None:
        return None, None
    if matrix.ndim != 2 or matrix.shape[1] != len(specs):
        raise ValueError("regularization_matrix must have one column per parameter")
    if reference.shape != (len(specs),):
        raise ValueError("regularization_matrix_reference must match parameter count")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(reference)):
        raise ValueError("matrix regularization must be finite")
    return matrix, reference
