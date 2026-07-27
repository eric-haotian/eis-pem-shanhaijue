# learning AI at www.haotianblog.com
"""Publication nuisance-policy and reference-cloud audit primitives.

These utilities deliberately accept no truth-bearing inputs.  The reference
cloud perturbs disclosed prior means, compares independently fitted estimates,
and may only flag/demote claims; it never grants a positive claim.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .parameters import ParameterSpec

from .priors import PriorSet


class NuisancePolicy(str, enum.Enum):
    """How parameters outside the released candidate set enter the fit."""

    LEGACY_HARD_FIXED = "legacy_hard_fixed"
    ALL48_WEAK_PRIOR = "all48_weak_prior"


def parse_nuisance_policy(value: NuisancePolicy | str) -> NuisancePolicy:
    if isinstance(value, NuisancePolicy):
        return value
    try:
        return NuisancePolicy(str(value))
    except ValueError as exc:
        allowed = [policy.value for policy in NuisancePolicy]
        raise ValueError(f"nuisance_policy must be one of {allowed}") from exc


@dataclass(frozen=True)
class PriorMeanPerturbation:
    """One truth-free prior-mean shift in the active fit coordinates."""

    nuisance_name: str
    direction: str
    sigma_search: float
    base_mean: float
    perturbed_mean: float


@dataclass(frozen=True)
class ReferenceCloudFit:
    """Estimates from one independently executed reference-cloud fit."""

    nuisance_name: str
    direction: str
    perturbed_mean: float
    estimates: Mapping[str, float]


@dataclass(frozen=True)
class ReferenceSensitivityAudit:
    """Demotion-only comparison of claims across a prior-mean cloud."""

    threshold_width_fraction: float
    rows: tuple[dict[str, object], ...]
    reference_sensitive_names: tuple[str, ...]
    truth_access: bool = False

    def __post_init__(self) -> None:
        if self.truth_access:
            raise ValueError("reference-sensitivity audit must not access truth")


def prior_mean_perturbations(
    prior_set: PriorSet,
    specs: Sequence[ParameterSpec],
    nuisance_names: Sequence[str],
    *,
    sigma_multiplier: float = 1.0,
) -> tuple[PriorMeanPerturbation, ...]:
    """Generate +/- prior-sigma perturbations without using generating truth."""

    if not np.isfinite(sigma_multiplier) or sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be finite and positive")
    spec_by_name = {spec.name: spec for spec in specs}
    unknown = set(nuisance_names) - set(spec_by_name)
    if unknown:
        raise ValueError(f"unknown nuisance parameters: {sorted(unknown)}")
    rows: list[PriorMeanPerturbation] = []
    for name in dict.fromkeys(nuisance_names):
        spec = spec_by_name[name]
        prior = prior_set.get(name)
        mean = float(prior.mean_physical if prior is not None else spec.initial_value)
        sigma = float(
            prior.sigma_search
            if prior is not None
            else prior_set.sigma_search_vector((spec,))[0]
        )
        for sign, direction in ((1.0, "+1sigma"), (-1.0, "-1sigma")):
            if spec.log_transform:
                shifted = mean * 10.0 ** (sign * sigma_multiplier * sigma)
            else:
                shifted = mean + sign * sigma_multiplier * sigma
            shifted = float(np.clip(shifted, *spec.bounds))
            rows.append(
                PriorMeanPerturbation(
                    nuisance_name=name,
                    direction=direction,
                    sigma_search=sigma,
                    base_mean=mean,
                    perturbed_mean=shifted,
                )
            )
    return tuple(rows)


def audit_reference_sensitivity(
    *,
    base_estimates: Mapping[str, float],
    local_relative_widths: Mapping[str, float],
    claimed_names: Sequence[str],
    perturbation_fits: Sequence[ReferenceCloudFit],
    threshold_width_fraction: float = 0.5,
    eps: float = 1e-30,
) -> ReferenceSensitivityAudit:
    """Flag claims whose fitted coordinate moves too far in a reference cloud.

    ``local_relative_widths`` are explicitly local diagnostic scales, not
    calibrated confidence intervals.  The function is side-effect free and
    does not accept a truth vector, score, or recovery label.
    """

    if not np.isfinite(threshold_width_fraction) or threshold_width_fraction <= 0:
        raise ValueError("threshold_width_fraction must be finite and positive")
    claims = tuple(dict.fromkeys(claimed_names))
    missing = set(claims) - set(base_estimates)
    if missing:
        raise ValueError(f"base estimates missing claims: {sorted(missing)}")
    missing_width = set(claims) - set(local_relative_widths)
    if missing_width:
        raise ValueError(f"local widths missing claims: {sorted(missing_width)}")

    rows: list[dict[str, object]] = []
    flagged: set[str] = set()
    for perturbation in perturbation_fits:
        missing_fit = set(claims) - set(perturbation.estimates)
        if missing_fit:
            raise ValueError(
                f"perturbation fit missing claims: {sorted(missing_fit)}"
            )
        for name in claims:
            base = float(base_estimates[name])
            shifted = float(perturbation.estimates[name])
            width = float(local_relative_widths[name])
            relative_shift = abs(shifted - base) / max(abs(base), eps)
            shift_over_width = (
                relative_shift / width
                if np.isfinite(width) and width > 0
                else float("inf")
            )
            is_sensitive = bool(shift_over_width > threshold_width_fraction)
            if is_sensitive:
                flagged.add(name)
            rows.append(
                {
                    "parameter": name,
                    "nuisance_name": perturbation.nuisance_name,
                    "direction": perturbation.direction,
                    "perturbed_mean": float(perturbation.perturbed_mean),
                    "base_estimate": base,
                    "perturbed_estimate": shifted,
                    "relative_shift": float(relative_shift),
                    "local_relative_width": width,
                    "shift_over_local_width": float(shift_over_width),
                    "reference_sensitive": is_sensitive,
                    "truth_access": False,
                }
            )
    return ReferenceSensitivityAudit(
        threshold_width_fraction=float(threshold_width_fraction),
        rows=tuple(rows),
        reference_sensitive_names=tuple(name for name in claims if name in flagged),
    )
