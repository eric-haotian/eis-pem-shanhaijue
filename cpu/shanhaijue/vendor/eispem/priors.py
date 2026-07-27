# learning AI at www.haotianblog.com
"""DRT-derived Gaussian parameter priors for MAP fitting and gating.

Generation 1 used the DRT only to produce initial guesses.  Generation 2
promotes the same evidence into explicit Gaussian priors in *search
space* (log10 for log-transformed parameters, linear otherwise):

* the DRT guess becomes the prior **mean**;
* the prior **width** is set per parameter kind (arc resistances and time
  constants are well constrained by DRT peaks; diffusion magnitudes only
  weakly by the tail) and widened when the DRT reconstruction itself fits
  the spectrum poorly.

These priors serve two mathematically linked roles:

1. **MAP estimation** — appended as residual rows
   ``(noise/sigma) * (s - mu)`` through the Generation-1 optimizer's
   ``regularization_weights`` / ``regularization_reference`` hooks, which
   is exactly the Gaussian-prior maximum a posteriori objective.
2. **Posterior identifiability** — the prior covariance enters the
   posterior Fisher information ``J^T J / sigma_n^2 + Sigma_p^{-1}``
   evaluated by the tiered gate, provably bounding the condition number
   of the augmented problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .parameters import ParameterSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussianParameterPrior:
    """Gaussian prior on one parameter, expressed in search space.

    ``mean_physical`` is stored in physical units; ``sigma_search`` is the
    standard deviation of the search-space coordinate (decades for
    log-transformed parameters, native units otherwise).
    """

    name: str
    mean_physical: float
    sigma_search: float
    source: str = "drt"

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_physical):
            raise ValueError(f"{self.name} prior mean must be finite")
        if not np.isfinite(self.sigma_search) or self.sigma_search <= 0:
            raise ValueError(f"{self.name} prior sigma must be finite and positive")


@dataclass(frozen=True)
class GaussianSearchCovariance:
    """One off-diagonal covariance in parameter search coordinates."""

    name_a: str
    name_b: str
    covariance: float
    source: str = "coordinate_pushforward"

    def __post_init__(self) -> None:
        if not self.name_a or not self.name_b or self.name_a == self.name_b:
            raise ValueError("covariance names must be distinct and non-empty")
        if not np.isfinite(self.covariance):
            raise ValueError("search covariance must be finite")


def weak_sigma_search(spec: ParameterSpec) -> float:
    """Weakly-informative fallback width: a quarter of the search span."""

    lower, upper = spec.optimization_bounds
    return max((upper - lower) / 4.0, 1e-6)


@dataclass(frozen=True)
class PriorSet:
    """Collection of parameter priors with spec-aware fallbacks."""

    priors: tuple[GaussianParameterPrior, ...] = ()
    search_covariances: tuple[GaussianSearchCovariance, ...] = ()

    def __post_init__(self) -> None:
        names = [p.name for p in self.priors]
        if len(set(names)) != len(names):
            raise ValueError("prior names must be unique")
        pairs = [frozenset((c.name_a, c.name_b)) for c in self.search_covariances]
        if len(set(pairs)) != len(pairs):
            raise ValueError("search covariance pairs must be unique")

    def get(self, name: str) -> GaussianParameterPrior | None:
        for prior in self.priors:
            if prior.name == name:
                return prior
        return None

    @property
    def informed_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.priors)

    def sigma_search_vector(self, specs: Sequence[ParameterSpec]) -> np.ndarray:
        """Per-parameter search-space sigmas, with weak fallbacks."""

        sigmas = np.empty(len(specs), dtype=float)
        for i, spec in enumerate(specs):
            prior = self.get(spec.name)
            sigmas[i] = prior.sigma_search if prior else weak_sigma_search(spec)
        return sigmas

    def covariance_search_matrix(
        self, specs: Sequence[ParameterSpec]
    ) -> np.ndarray:
        """Full Gaussian covariance in the supplied search coordinates."""

        spec_tuple = tuple(specs)
        names = tuple(spec.name for spec in spec_tuple)
        index = {name: i for i, name in enumerate(names)}
        sigma = self.sigma_search_vector(spec_tuple)
        covariance = np.diag(sigma**2)
        for entry in self.search_covariances:
            if entry.name_a not in index or entry.name_b not in index:
                continue
            i, j = index[entry.name_a], index[entry.name_b]
            covariance[i, j] = covariance[j, i] = float(entry.covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.size and eigenvalues[0] <= 0:
            raise ValueError(
                "search-space prior covariance must be positive definite"
            )
        return covariance

    def precision_search_matrix(
        self, specs: Sequence[ParameterSpec]
    ) -> np.ndarray:
        """Full Gaussian precision in the supplied search coordinates."""

        return np.linalg.inv(self.covariance_search_matrix(specs))

    def mean_physical_vector(
        self,
        specs: Sequence[ParameterSpec],
        fallback_theta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-parameter prior means (physical units), clamped to bounds."""

        means = np.empty(len(specs), dtype=float)
        for i, spec in enumerate(specs):
            prior = self.get(spec.name)
            if prior is not None:
                value = prior.mean_physical
            elif fallback_theta is not None:
                value = float(fallback_theta[i])
            else:
                value = spec.initial_value
            means[i] = min(max(value, spec.bounds[0]), spec.bounds[1])
        return means

    def regularization_for_map(
        self,
        specs: Sequence[ParameterSpec],
        noise_level: float,
        fallback_theta: np.ndarray | None = None,
        include_weak: bool = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return MAP regularization dicts for the Gen-1 optimizers.

        The Gen-1 optimizers append residual rows ``w * (s - s_ref)``.
        For those rows to represent the Gaussian prior term of the MAP
        objective alongside relative data residuals with noise level
        ``sigma_n``, the weight must be ``sigma_n / sigma_prior``.
        """

        if not np.isfinite(noise_level) or noise_level <= 0:
            raise ValueError("noise_level must be finite and positive")
        if self.search_covariances:
            raise RuntimeError(
                "correlated priors require regularization_geometry_for_map"
            )
        sigmas = self.sigma_search_vector(specs)
        means = self.mean_physical_vector(specs, fallback_theta)
        weights: dict[str, float] = {}
        references: dict[str, float] = {}
        for spec, sigma, mean in zip(specs, sigmas, means, strict=True):
            if not include_weak and self.get(spec.name) is None:
                continue
            weights[spec.name] = float(noise_level / sigma)
            references[spec.name] = float(mean)
        return weights, references

    def regularization_geometry_for_map(
        self,
        specs: Sequence[ParameterSpec],
        noise_level: float,
        fallback_theta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a full MAP residual factor and search-space reference.

        If ``P`` is the prior precision, the returned matrix ``W`` satisfies
        ``W.T @ W = noise_level**2 * P``.  This preserves covariance induced
        by a coordinate transformation instead of silently diagonalizing it.
        """

        if not np.isfinite(noise_level) or noise_level <= 0:
            raise ValueError("noise_level must be finite and positive")
        spec_tuple = tuple(specs)
        precision = self.precision_search_matrix(spec_tuple)
        factor = noise_level * np.linalg.cholesky(precision).T
        means = self.mean_physical_vector(spec_tuple, fallback_theta)
        reference = np.asarray(
            [
                spec.to_optimization(value)
                for spec, value in zip(spec_tuple, means, strict=True)
            ],
            dtype=float,
        )
        return factor, reference

    def to_rows(self, specs: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
        """Audit rows describing the effective prior on every parameter."""

        rows = []
        for spec in specs:
            prior = self.get(spec.name)
            rows.append(
                {
                    "parameter": spec.name,
                    "prior_mean": (
                        float(prior.mean_physical) if prior else spec.initial_value
                    ),
                    "prior_sigma_search": (
                        float(prior.sigma_search)
                        if prior
                        else weak_sigma_search(spec)
                    ),
                    "search_space": "log10" if spec.log_transform else "linear",
                    "source": prior.source if prior else "weak_bounds",
                }
            )
        return rows


# ---------------------------------------------------------------------------
# DRT prior builder
# ---------------------------------------------------------------------------

#: Default prior widths (search-space standard deviations) by parameter
#: kind.  Log-transformed parameters: width in decades.  DRT peak
#: positions constrain time constants tightly; peak areas constrain arc
#: resistances; the diffusion tail constrains Rd/tau_d only loosely.
DEFAULT_SIGMA_BY_KIND: Mapping[str, float] = {
    "rs": 0.15,
    "r_arc": 0.30,
    "tau_arc": 0.35,
    "alpha": 0.08,
    "gamma": 0.05,
    "q": 0.50,
    "r_diff": 0.60,
    "tau_diff": 0.70,
    "inductance": 1.00,
    "r_ion": 0.50,
}


def classify_parameter_kind(name: str) -> str:
    """Map a parameter name to its prior-width kind."""

    if name == "Rs":
        return "rs"
    if name == "L":
        return "inductance"
    if name.startswith("alpha"):
        return "alpha"
    if name.startswith("gamma"):
        return "gamma"
    if name.startswith("tau_d") or name.startswith("tau_g"):
        return "tau_diff"
    if name.startswith("tau"):
        return "tau_arc"
    if name.startswith("Q"):
        return "q"
    if name in ("Rd", "Rg", "Aw"):
        return "r_diff"
    if name == "Rion":
        return "r_ion"
    if name.startswith("R"):
        return "r_arc"
    return "q"


@dataclass(frozen=True)
class DRTPriorBuilder:
    """Convert a DRT analysis into per-parameter Gaussian priors.

    Prior means come from the model's own ``guess_from_drt`` mapping (the
    same evidence Generation 1 used for initialization).  Prior widths
    are per-kind defaults widened by the DRT reconstruction error, so a
    noisy or poorly resolved DRT automatically yields weaker priors.

    **Empirical-Bayes width calibration** (``calibrate_sigma``): a prior
    whose mean disagrees with the data can masquerade as identifiability
    — the posterior CI passes because the prior forces it.  When a
    data-only fit is available, :meth:`calibrated` widens every informed
    prior whose mean sits more than ``calibration_tolerance`` standard
    deviations from the data-only estimate (in search space) by
    ``min(deviation / sigma, max_calibration_factor)``, so a wrong prior
    loses exactly the influence it should not have had.
    """

    sigma_by_kind: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SIGMA_BY_KIND)
    )
    quality_widening_gain: float = 10.0
    max_widening_factor: float = 3.0
    calibrate_sigma: bool = True
    calibration_tolerance: float = 1.5
    max_calibration_factor: float = 3.0

    def build(
        self,
        model: Any,
        drt_result: Any,
        specs: Sequence[ParameterSpec] | None = None,
        theta_data_fit: np.ndarray | None = None,
    ) -> PriorSet:
        spec_list = tuple(specs) if specs is not None else tuple(model.parameter_specs)
        guess = None
        guesser = getattr(model, "guess_from_drt", None)
        if guesser is not None and drt_result is not None:
            try:
                guess = guesser(drt_result)
            except Exception:  # pragma: no cover - defensive
                logger.warning("guess_from_drt failed for %s", model.name, exc_info=True)
        if not guess:
            return PriorSet()

        widening = self._widening_factor(drt_result)
        priors: list[GaussianParameterPrior] = []
        for spec in spec_list:
            if spec.name not in guess:
                continue
            mean = float(guess[spec.name])
            mean = min(max(mean, spec.bounds[0]), spec.bounds[1])
            kind = classify_parameter_kind(spec.name)
            sigma = float(self.sigma_by_kind.get(kind, 0.5)) * widening
            sigma = min(sigma, weak_sigma_search(spec))
            priors.append(
                GaussianParameterPrior(
                    name=spec.name,
                    mean_physical=mean,
                    sigma_search=sigma,
                    source=f"drt:{kind}",
                )
            )
        prior_set = PriorSet(priors=tuple(priors))
        if theta_data_fit is not None:
            prior_set = self.calibrated(prior_set, spec_list, theta_data_fit)
        return prior_set

    def calibrated(
        self,
        prior_set: PriorSet,
        specs: Sequence[ParameterSpec],
        theta_data_fit: np.ndarray,
    ) -> PriorSet:
        """One-shot empirical-Bayes prior width correction.

        Returns the *same* object when nothing changes, so callers can
        use an identity check to decide whether a refit is needed.
        """

        if not self.calibrate_sigma or not prior_set.priors:
            return prior_set
        spec_list = tuple(specs)
        values = np.asarray(theta_data_fit, dtype=float)
        if values.shape != (len(spec_list),):
            raise ValueError("theta_data_fit must match parameter_specs length")
        spec_by_name = {spec.name: spec for spec in spec_list}
        value_by_name = dict(
            zip((spec.name for spec in spec_list), values, strict=True)
        )

        changed = False
        updated: list[GaussianParameterPrior] = []
        for prior in prior_set.priors:
            spec = spec_by_name.get(prior.name)
            value = value_by_name.get(prior.name)
            if spec is None or value is None or not np.isfinite(value):
                updated.append(prior)
                continue
            if spec.log_transform:
                if value <= 0 or prior.mean_physical <= 0:
                    updated.append(prior)
                    continue
                deviation = abs(np.log10(value) - np.log10(prior.mean_physical))
            else:
                deviation = abs(value - prior.mean_physical)
            if deviation <= self.calibration_tolerance * prior.sigma_search:
                updated.append(prior)
                continue
            factor = min(
                deviation / prior.sigma_search, self.max_calibration_factor
            )
            sigma = min(prior.sigma_search * factor, weak_sigma_search(spec))
            if sigma <= prior.sigma_search:
                updated.append(prior)
                continue
            changed = True
            logger.info(
                "prior calibration: %s widened %.3f -> %.3f "
                "(mean %.3g vs data fit %.3g)",
                prior.name,
                prior.sigma_search,
                sigma,
                prior.mean_physical,
                value,
            )
            updated.append(
                replace(prior, sigma_search=sigma, source=prior.source + "+calibrated")
            )
        if not changed:
            return prior_set
        return PriorSet(priors=tuple(updated))

    def _widening_factor(self, drt_result: Any) -> float:
        residuals = getattr(drt_result, "residuals", None)
        if residuals is None or np.size(residuals) == 0:
            return self.max_widening_factor
        median_residual = float(np.median(np.abs(residuals)))
        factor = 1.0 + self.quality_widening_gain * median_residual
        return float(np.clip(factor, 1.0, self.max_widening_factor))
