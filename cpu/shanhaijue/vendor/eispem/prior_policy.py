# learning AI at www.haotianblog.com
"""Publication-candidate prior policy and legacy compatibility guards."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PublicationPriorPolicy:
    """Frozen prior-facing rules for publication-candidate execution."""

    prior_tightening: bool = False
    profile_rescue: bool = False
    headline_tiers: tuple[str, ...] = ("STRICT",)
    serialize_prior_informed_separately: bool = True

    def __post_init__(self) -> None:
        if self.prior_tightening:
            raise ValueError(
                "publication candidate permanently forbids prior tightening"
            )
        if self.headline_tiers != ("STRICT",):
            raise ValueError("publication headline_tiers must be STRICT only")
        if not self.serialize_prior_informed_separately:
            raise ValueError("PRIOR_INFORMED must be serialized separately")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["headline_tiers"] = list(self.headline_tiers)
        return payload


def publication_prior_policy_from_mapping(
    config: Mapping[str, Any],
) -> PublicationPriorPolicy:
    """Validate a publication config without accepting legacy loosening."""

    return PublicationPriorPolicy(
        prior_tightening=bool(config.get("prior_tightening", False)),
        profile_rescue=bool(config.get("profile_rescue", False)),
        headline_tiers=tuple(config.get("headline_tiers", ("STRICT",))),
        serialize_prior_informed_separately=bool(
            config.get("serialize_prior_informed_separately", True)
        ),
    )


def resolve_legacy_prior_tightening(requested: bool) -> str:
    """Warn and ignore the removed legacy switch."""

    if requested:
        warnings.warn(
            "use_prior_tightening=True is deprecated and ignored; publication "
            "candidate prior tightening is permanently disabled",
            FutureWarning,
            stacklevel=3,
        )
        return "DISABLED_DEPRECATED_REQUEST_IGNORED"
    return "DISABLED"
