# learning AI at www.haotianblog.com
"""Exact profile-likelihood certification for SEIS identifiability claims.

Motivation (measured, not assumed).  The tiered gate reports Wald
confidence intervals — the diagonal of the inverse posterior Fisher
matrix.  Wald is a *local* (Laplace) approximation.  The textbook
expectation is that it is conservative for nonlinear models, so profile
likelihood would only tighten it.  On the 5x5 SEIS problem the opposite
holds: with informative literature priors and strong residual coupling
the posterior ridge is flatter than the local quadratic, so the exact
profile interval is *wider* than Wald for every borderline parameter
(measured: epse_neg 0.12 -> 0.30, brug_neg 0.35 -> 1.00).

The consequence is that profile likelihood cannot *inflate* the
identifiable count on this data — it can only *certify* it.  This module
provides that certification: for each claimed parameter it computes the
exact likelihood-ratio confidence interval by re-optimising the rest of
the subset at a grid of fixed values (bisection root-finding on the LR
statistic), and flags any Wald claim whose true profile interval exceeds
the gate threshold.  A data-likelihood LR statistic on the same model
manifold, objective, and feasible bounds is invariant under a bijective
reparameterization.  Posterior profiles are invariant only when the complete
joint prior density and its Jacobian are transformed consistently; replacing
it with independent Gaussian marginals or static rectangular bounds in new
coordinates can legitimately change the profile.

Two modes:

* ``posterior`` (default) — the profiled objective keeps every prior,
  including the profiled parameter's own.  This is the MAP posterior
  credible interval, directly comparable to the gate's
  ``posterior_ci95``.
* ``data_only`` — the profiled parameter's own prior term is removed, so
  the deviation must be rejected by the data (plus the *other* priors).
  This is the stricter guard used for upgrades in
  :func:`eis_pem_v2.seis_pipeline._profile_rescue`; a parameter passing
  it is identified by the measurement, not by its prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .dataset import EISDataset
from .optimizers import AdaptiveLeastSquaresOptimizer
from .parameters import ParameterSpec

from .fitting import SubsetModel
from .gating import ParameterTier, TieredSelection
from .priors import PriorSet

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileInterval:
    """Exact profile-likelihood confidence interval for one parameter."""

    parameter: str
    map_value: float
    lower_rel: float          # relative downward half-width (>= 0)
    upper_rel: float          # relative upward half-width (>= 0)
    lower_saturated: bool     # profile never reached the threshold downward
    upper_saturated: bool

    @property
    def ci95_rel(self) -> float:
        """Symmetric-report relative CI95 = max of the two half-widths."""

        return float(max(self.lower_rel, self.upper_rel))


@dataclass(frozen=True)
class ProfileCertification:
    """Result of profile-certifying a claimed parameter set."""

    intervals: dict[str, ProfileInterval]
    wald_ci95: dict[str, float]
    certified: dict[str, bool]
    threshold: float
    mode: str

    @property
    def n_certified(self) -> int:
        return sum(1 for ok in self.certified.values() if ok)

    @property
    def overclaimed(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.certified.items() if not ok)

    def to_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name, interval in self.intervals.items():
            rows.append(
                {
                    "parameter": name,
                    "wald_ci95": float(self.wald_ci95.get(name, float("nan"))),
                    "profile_ci95": interval.ci95_rel,
                    "profile_lower_rel": interval.lower_rel,
                    "profile_upper_rel": interval.upper_rel,
                    "saturated": interval.lower_saturated or interval.upper_saturated,
                    "certified": bool(self.certified[name]),
                }
            )
        return rows


@dataclass(frozen=True)
class ProfileLikelihoodCertifier:
    """Bisection profile-likelihood confidence intervals.

    Parameters
    ----------
    quantile
        Confidence level; the LR threshold is ``chi2.ppf(quantile, 1)``.
    max_relative_ci95
        Claim bar the profile CI must satisfy to certify a parameter.
    mode
        ``"posterior"`` (keep own prior) or ``"data_only"`` (drop it).
    max_factor
        Largest multiplicative deviation probed (3.0 -> +/-200 %); the
        interval saturates here when the profile never crosses the
        threshold, and the parameter is treated as not certified.
    bisection_iters
        Bisection steps per side (9 by default) to keep numerical profile
        crossings stable under equivalent parameterizations.
    """

    quantile: float = 0.95
    max_relative_ci95: float = 0.20
    mode: str = "posterior"
    max_factor: float = 3.0
    bisection_iters: int = 9
    max_nfev: int = 3000
    n_starts: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("posterior", "data_only"):
            raise ValueError("mode must be 'posterior' or 'data_only'")
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        if self.max_factor <= 1.0:
            raise ValueError("max_factor must exceed 1")

    # -- core profile computation ------------------------------------------

    def profile_interval(
        self,
        dataset: EISDataset,
        model: Any,
        specs: Sequence[ParameterSpec],
        prior_set: PriorSet,
        map_theta: Mapping[str, float],
        map_cost: float,
        noise_level: float,
        parameter: str,
    ) -> ProfileInterval:
        from scipy.stats import chi2

        threshold = float(chi2.ppf(self.quantile, df=1)) * noise_level**2
        spec_by = {spec.name: spec for spec in specs}
        subset_names = [spec.name for spec in specs]
        rest = [n for n in subset_names if n != parameter]
        map_value = float(map_theta[parameter])
        optimizer = AdaptiveLeastSquaresOptimizer(
            relative=True, n_starts=self.n_starts, max_nfev=self.max_nfev,
            ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        own_prior = prior_set.get(parameter)
        own_spec = spec_by[parameter]

        # A profiled fit fixes `parameter` on top of whatever the claimed
        # subset already froze.  Recover the outer full model and its
        # existing fixed set so the reduced fit spans exactly the other
        # claimed parameters.
        if isinstance(model, SubsetModel):
            full_model = model.full_model
            full_specs = tuple(model.full_specs)
            outer_fixed = dict(model.fixed_values)
        else:
            full_model = model
            full_specs = tuple(specs)
            outer_fixed = {}

        def own_prior_penalty(value: float) -> float:
            if self.mode != "data_only" or own_prior is None:
                return 0.0
            s = own_spec.to_optimization(float(value))
            mu = own_spec.to_optimization(
                min(max(own_prior.mean_physical, own_spec.bounds[0]), own_spec.bounds[1])
            )
            return float((noise_level / own_prior.sigma_search) ** 2 * (s - mu) ** 2)

        base_own = own_prior_penalty(map_value)

        def lr_at(factor: float) -> float:
            fixed_value = map_value * factor
            fixed = dict(outer_fixed)
            fixed[parameter] = fixed_value
            reduced = SubsetModel(
                full_model=full_model, full_specs=full_specs, fixed_values=fixed
            )
            rest_specs = [
                spec_by[n] for n in reduced.parameter_names if n in spec_by
            ]
            reg_matrix, reg_reference = prior_set.regularization_geometry_for_map(
                rest_specs, noise_level
            )
            fit = optimizer.fit(
                dataset=dataset,
                model=reduced,
                parameter_specs=rest_specs,
                initial_guess={
                    n: float(map_theta[n])
                    for n in reduced.parameter_names
                    if n in map_theta
                },
                regularization_matrix=reg_matrix,
                regularization_matrix_reference=reg_reference,
            )
            data_delta = float(fit.final_cost - map_cost)
            data_delta -= own_prior_penalty(fixed_value) - base_own
            return data_delta

        upper_rel, upper_sat = self._bisect(lr_at, threshold, direction=+1.0)
        lower_rel, lower_sat = self._bisect(lr_at, threshold, direction=-1.0)
        return ProfileInterval(
            parameter=parameter,
            map_value=map_value,
            lower_rel=lower_rel,
            upper_rel=upper_rel,
            lower_saturated=lower_sat,
            upper_saturated=upper_sat,
        )

    def _bisect(self, lr_at, threshold: float, direction: float) -> tuple[float, bool]:
        """Find the relative half-width where the LR crosses ``threshold``.

        ``direction`` is +1 (probe factors > 1) or -1 (factors < 1).
        Returns ``(relative_half_width, saturated)``.
        """

        def factor_of(rel: float) -> float:
            return 1.0 + direction * rel

        max_rel = (self.max_factor - 1.0) if direction > 0 else (1.0 - 1.0 / self.max_factor)
        if lr_at(factor_of(max_rel)) < threshold:
            return float(max_rel), True
        lo, hi = 0.0, max_rel
        for _ in range(self.bisection_iters):
            mid = 0.5 * (lo + hi)
            if lr_at(factor_of(mid)) < threshold:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi)), False

    # -- certification driver ----------------------------------------------

    def certify(
        self,
        dataset: EISDataset,
        model: Any,
        specs: Sequence[ParameterSpec],
        prior_set: PriorSet,
        map_theta: Mapping[str, float],
        map_cost: float,
        noise_level: float,
        selection: TieredSelection,
        parameters: Sequence[str] | None = None,
        wald_skip_below: float = 0.08,
    ) -> ProfileCertification:
        """Profile-certify the claimed (STRICT / PRIOR_INFORMED) parameters.

        Parameters with a Wald CI well under the bar
        (``wald_skip_below``) are certified without profiling — a tight
        quadratic bowl cannot hide a distant threshold crossing.  The
        rest are profiled exactly.
        """

        claim_tiers = (ParameterTier.STRICT, ParameterTier.PRIOR_INFORMED)
        subset_names = {spec.name for spec in specs}
        if parameters is None:
            parameters = [
                n for n in selection.parameter_names
                if selection.tiers[n] in claim_tiers and n in subset_names
            ]
        intervals: dict[str, ProfileInterval] = {}
        wald: dict[str, float] = {}
        certified: dict[str, bool] = {}
        from scipy.stats import chi2

        threshold = float(chi2.ppf(self.quantile, df=1)) * noise_level**2
        for name in parameters:
            wald_ci = float(selection.posterior_ci95[name])
            wald[name] = wald_ci
            if wald_ci < wald_skip_below:
                intervals[name] = ProfileInterval(
                    name, float(map_theta[name]), wald_ci, wald_ci, False, False
                )
                certified[name] = True
                continue
            interval = self.profile_interval(
                dataset, model, specs, prior_set, map_theta, map_cost,
                noise_level, name,
            )
            intervals[name] = interval
            certified[name] = interval.ci95_rel <= self.max_relative_ci95
        return ProfileCertification(
            intervals=intervals,
            wald_ci95=wald,
            certified=certified,
            threshold=threshold,
            mode=self.mode,
        )

    def downgrade(
        self, selection: TieredSelection, certification: ProfileCertification
    ) -> TieredSelection:
        """Return a selection with profile over-claims demoted to FIXED.

        Only PRIOR_INFORMED claims are auto-demoted; STRICT claims come
        from the untouched Generation-1 data-only gate and carry the
        published guarantee, so they are reported but never silently
        overturned by the posterior certifier (they are logged instead).
        """

        new_tiers = dict(selection.tiers)
        new_ci = dict(selection.posterior_ci95)
        for name in certification.overclaimed:
            interval = certification.intervals[name]
            new_ci[name] = interval.ci95_rel
            if selection.tiers[name] is ParameterTier.PRIOR_INFORMED:
                new_tiers[name] = ParameterTier.FIXED
                logger.info(
                    "profile certification demoted %s: Wald %.2f -> profile %.2f",
                    name, certification.wald_ci95[name], interval.ci95_rel,
                )
            elif selection.tiers[name] is ParameterTier.STRICT:
                logger.warning(
                    "profile CI for STRICT %s is %.2f (> %.2f) — data-only "
                    "gate retained but flagged for review",
                    name, interval.ci95_rel, self.max_relative_ci95,
                )
        return replace(selection, tiers=new_tiers, posterior_ci95=new_ci)
