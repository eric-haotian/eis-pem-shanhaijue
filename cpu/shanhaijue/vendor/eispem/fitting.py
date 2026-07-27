# learning AI at www.haotianblog.com
"""Hierarchical MAP fitting for Generation-2 models.

Complex models fail Generation 1's gate not only because of genuine
non-identifiability but also because a cold, all-parameters-at-once
local fit lands in a poor basin, and the gate then judges the Jacobian
at that poor point.  Generation 2 fits in stages:

* **Stage A (shape-frozen)** — CPE exponents (``alpha*``) and fractional
  orders (``gamma*``) are frozen at their DRT-informed values; the
  well-behaved magnitude/time-constant subspace is MAP-fitted first.
* **Stage B (full MAP)** — every parameter is released, warm-started
  from Stage A, with the DRT priors as Gaussian regularization rows.
* **Stage C (data-only polish, optional)** — a pure maximum-likelihood
  refinement from the MAP point.  It is accepted only when it improves
  the data cost *and* stays within ``polish_sigma_tolerance`` prior
  standard deviations of every informed prior; otherwise the MAP result
  stands.  This removes prior-induced bias on well-identified parameters
  without allowing weakly identified ones to run away.

All optimization is delegated to the untouched Generation-1
:class:`eis_pem.AdaptiveLeastSquaresOptimizer`; its
``regularization_weights`` / ``regularization_reference`` hooks implement
exactly the Gaussian-prior MAP residual rows (see
:meth:`eis_pem_v2.priors.PriorSet.regularization_for_map`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .dataset import EISDataset
from .optimizers import AdaptiveLeastSquaresOptimizer
from .parameters import ParameterSpec
from .results import IdentificationResult

from .priors import PriorSet

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsetModel:
    """Expose a subset of a model's parameters, fixing the rest.

    Unlike :class:`eis_pem.robust.ReducedParameterModel` this does not
    require a full :class:`ParameterSelection`, so it can be used for
    staged fitting with arbitrary frozen subsets.
    """

    full_model: Any
    full_specs: tuple[ParameterSpec, ...]
    fixed_values: dict[str, float]

    def __post_init__(self) -> None:
        names = {spec.name for spec in self.full_specs}
        unknown = set(self.fixed_values) - names
        if unknown:
            raise ValueError(f"fixed values reference unknown parameters: {unknown}")
        if len(self.fixed_values) >= len(self.full_specs):
            raise ValueError("at least one parameter must remain free")

    @property
    def name(self) -> str:
        return getattr(self.full_model, "name", "model") + "_subset"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(
            spec for spec in self.full_specs if spec.name not in self.fixed_values
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameter_specs)

    @property
    def n_params(self) -> int:
        return len(self.parameter_specs)

    def expand_theta(self, theta_free: NDArray[np.floating]) -> FloatArray:
        free = np.asarray(theta_free, dtype=float)
        if free.shape != (self.n_params,):
            raise ValueError("theta_free must match the free parameter count")
        values = dict(self.fixed_values)
        values.update(dict(zip(self.parameter_names, free, strict=True)))
        return np.asarray(
            [values[spec.name] for spec in self.full_specs], dtype=float
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        return self.full_model.simulate(freq_hz, self.expand_theta(theta))

    def expand_theta_jax(self, theta_free: Any) -> Any:
        """Expand a free vector without converting JAX tracers to NumPy."""

        import jax.numpy as jnp

        free_index = {name: i for i, name in enumerate(self.parameter_names)}
        values = [
            theta_free[free_index[spec.name]]
            if spec.name in free_index
            else self.fixed_values[spec.name]
            for spec in self.full_specs
        ]
        return jnp.stack(values)

    def simulate_jax(self, freq_hz: Any, theta: Any) -> Any:
        simulator = getattr(self.full_model, "simulate_jax", None)
        if simulator is None:
            raise AttributeError("full model does not expose a JAX simulation path")
        return simulator(freq_hz, self.expand_theta_jax(theta))

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guesser = getattr(self.full_model, "guess_from_drt", None)
        if guesser is None:
            return None
        guess = guesser(drt_result)
        if not guess:
            return None
        return {k: v for k, v in guess.items() if k not in self.fixed_values}


@dataclass(frozen=True)
class HierarchicalMAPFitter:
    """Staged MAP fitting with DRT priors (see module docstring)."""

    n_starts: int = 6
    max_nfev: int = 8000
    ftol: float = 1e-11
    xtol: float = 1e-11
    gtol: float = 1e-11
    seed: int = 42
    shape_prefixes: tuple[str, ...] = ("alpha", "gamma")
    polish_data_only: bool = True
    polish_sigma_tolerance: float = 3.0
    loss: str = "linear"
    f_scale: float = 1.0

    def fit(
        self,
        dataset: EISDataset,
        model: Any,
        parameter_specs: Sequence[ParameterSpec],
        prior_set: PriorSet,
        noise_level: float,
        weights: NDArray[np.floating] | None = None,
        initial_guess: dict[str, float] | None = None,
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        names = tuple(spec.name for spec in specs)

        # Starting point: explicit guess > prior means > spec defaults.
        theta0 = prior_set.mean_physical_vector(specs)
        if initial_guess:
            for i, spec in enumerate(specs):
                if spec.name in initial_guess:
                    theta0[i] = min(
                        max(float(initial_guess[spec.name]), spec.bounds[0]),
                        spec.bounds[1],
                    )
        guess0 = dict(zip(names, theta0, strict=True))

        reg_matrix, reg_reference = prior_set.regularization_geometry_for_map(
            specs, noise_level, fallback_theta=theta0
        )

        stage_costs: dict[str, float] = {}

        # --- Stage A: shape parameters frozen ---
        shape_names = tuple(
            name for name in names if name.startswith(self.shape_prefixes)
        )
        theta_a = dict(guess0)
        if shape_names and len(shape_names) < len(names):
            subset = SubsetModel(
                full_model=model,
                full_specs=specs,
                fixed_values={name: guess0[name] for name in shape_names},
            )
            free_names = subset.parameter_names
            free_specs = tuple(subset.parameter_specs)
            free_matrix, free_reference = prior_set.regularization_geometry_for_map(
                free_specs, noise_level
            )
            result_a = self._map_optimizer(n_starts=self.n_starts).fit(
                dataset=dataset,
                model=subset,
                parameter_specs=subset.parameter_specs,
                weights=weights,
                initial_guess={n: guess0[n] for n in free_names},
                regularization_matrix=free_matrix,
                regularization_matrix_reference=free_reference,
            )
            theta_a.update(result_a.theta_best)
            stage_costs["stage_a_shape_frozen"] = float(result_a.final_cost)

        # --- Stage B: full MAP fit, warm-started from Stage A ---
        result_b = self._map_optimizer(n_starts=self.n_starts).fit(
            dataset=dataset,
            model=model,
            parameter_specs=specs,
            weights=weights,
            initial_guess=theta_a,
            regularization_matrix=reg_matrix,
            regularization_matrix_reference=reg_reference,
        )
        stage_costs["stage_b_full_map"] = float(result_b.final_cost)
        best = result_b

        # --- Stage C: optional data-only polish with prior guard ---
        # The data-only theta is recorded even when the polish is rejected:
        # prior-width calibration (empirical Bayes) needs the unregularized
        # estimate precisely when it disagrees with the priors.
        stage_c_theta: dict[str, float] | None = None
        if self.polish_data_only:
            result_c = self._map_optimizer(n_starts=1).fit(
                dataset=dataset,
                model=model,
                parameter_specs=specs,
                weights=weights,
                initial_guess=dict(result_b.theta_best),
            )
            stage_costs["stage_c_polish"] = float(result_c.final_cost)
            stage_c_theta = {k: float(v) for k, v in result_c.theta_best.items()}
            if result_c.final_cost < result_b.final_cost and self._within_prior_budget(
                specs, result_c.theta_best, prior_set
            ):
                best = result_c

        metadata = dict(best.metadata)
        metadata.update(
            {
                "fitter": "hierarchical_map",
                "stages": stage_costs,
                "prior_informed_parameters": list(prior_set.informed_names),
                "noise_level": float(noise_level),
                "polished": best is not result_b,
                "stage_c_theta": stage_c_theta,
            }
        )
        return IdentificationResult(
            theta_best=dict(best.theta_best),
            final_cost=float(best.final_cost),
            z_fit=best.z_fit,
            residuals=best.residuals,
            dataset=dataset,
            metadata=metadata,
        )

    def _map_optimizer(self, n_starts: int) -> AdaptiveLeastSquaresOptimizer:
        return AdaptiveLeastSquaresOptimizer(
            relative=True,
            n_starts=n_starts,
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            seed=self.seed,
            loss=self.loss,
            f_scale=self.f_scale,
        )

    def _within_prior_budget(
        self,
        specs: Sequence[ParameterSpec],
        theta_best: dict[str, float],
        prior_set: PriorSet,
    ) -> bool:
        """True when every informed prior is honoured within tolerance."""

        for spec in specs:
            prior = prior_set.get(spec.name)
            if prior is None:
                continue
            value = theta_best.get(spec.name)
            if value is None:
                continue
            if spec.log_transform:
                if value <= 0 or prior.mean_physical <= 0:
                    return False
                distance = abs(np.log10(value) - np.log10(prior.mean_physical))
            else:
                distance = abs(value - prior.mean_physical)
            if distance > self.polish_sigma_tolerance * prior.sigma_search:
                logger.debug(
                    "polish rejected: %s moved %.3f search units (> %.1f sigma)",
                    spec.name,
                    distance,
                    self.polish_sigma_tolerance,
                )
                return False
        return True
