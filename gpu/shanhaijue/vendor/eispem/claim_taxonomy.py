# learning AI at www.haotianblog.com
"""Reader-facing claim taxonomy and separated post-hoc scoring.

Internal gate tiers are inputs to this module, not publication vocabulary.
Every output claim is mapped to one of the frozen nine categories after the
mandatory negative audits.  Truth enters only the explicit evaluation-only
scorer and can neither change an estimate nor change a taxonomy record.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .seis_model import all_seis_parameter_specs

from .gating import ParameterTier
from .physics_reparameterization import DEFAULT_ENABLED
from .seis_reparameterization import ARRHENIUS_PAIRS, R_GAS, T_REF


RAW_RECOVERY_TOLERANCE = 0.10
_RAW_UNITS = {spec.name: spec.unit for spec in all_seis_parameter_specs()}


class ClaimCategory(str, enum.Enum):
    DATA_SUPPORTED_RAW_CANDIDATE = "DATA_SUPPORTED_RAW_CANDIDATE"
    DATA_SUPPORTED_COMPOSITE_CANDIDATE = (
        "DATA_SUPPORTED_COMPOSITE_CANDIDATE"
    )
    PRIOR_INFORMED = "PRIOR_INFORMED"
    PROTECTED_NUISANCE = "PROTECTED_NUISANCE"
    REFERENCE_SENSITIVE = "REFERENCE_SENSITIVE"
    STRATEGY_UNSTABLE = "STRATEGY_UNSTABLE"
    ADEQUACY_UNRESOLVED = "ADEQUACY_UNRESOLVED"
    REJECTED = "REJECTED"
    FIXED_EXTERNAL = "FIXED_EXTERNAL"


class CoordinateClass(str, enum.Enum):
    RAW = "RAW"
    PHYSICALLY_REPORTABLE_COMPOSITE = "PHYSICALLY_REPORTABLE_COMPOSITE"
    NUMERICAL_ONLY = "NUMERICAL_ONLY"


@dataclass(frozen=True)
class CompositeDefinition:
    """One frozen claim-coordinate definition evaluated from raw values."""

    claim_name: str
    storage_parameter: str
    dependencies: tuple[str, ...]
    definition: str
    unit: str
    evaluator: Callable[[Mapping[str, float]], float]
    coordinate_class: CoordinateClass = (
        CoordinateClass.PHYSICALLY_REPORTABLE_COMPOSITE
    )
    mandatory_disclosures: tuple[str, ...] = ()
    reference_temperature_K: float | None = None
    raw_factor_claims_implied: bool = False

    def __post_init__(self) -> None:
        if not self.claim_name or not self.storage_parameter:
            raise ValueError("composite names must be non-empty")
        if not self.dependencies or len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise ValueError("composite dependencies must be non-empty and unique")
        if self.coordinate_class is CoordinateClass.RAW:
            raise ValueError("CompositeDefinition cannot declare a raw coordinate")
        if not self.definition or not self.unit:
            raise ValueError("composite definition and unit are mandatory")
        if self.reference_temperature_K is not None and (
            not np.isfinite(self.reference_temperature_K)
            or self.reference_temperature_K <= 0
        ):
            raise ValueError("reference temperature must be finite and positive")
        if self.raw_factor_claims_implied:
            raise ValueError("a composite may never imply its raw factors")

    def evaluate(self, raw_values: Mapping[str, float]) -> float:
        missing = set(self.dependencies) - set(raw_values)
        if missing:
            raise ValueError(
                f"{self.claim_name} missing raw dependencies: {sorted(missing)}"
            )
        value = float(self.evaluator(raw_values))
        if not np.isfinite(value):
            raise ValueError(f"{self.claim_name} evaluated to a non-finite value")
        return value

    def to_manifest_row(self) -> dict[str, object]:
        return {
            "claim_name": self.claim_name,
            "storage_parameter": self.storage_parameter,
            "coordinate_class": self.coordinate_class.value,
            "dependencies": list(self.dependencies),
            "definition": self.definition,
            "unit": self.unit,
            "mandatory_disclosures": list(self.mandatory_disclosures),
            "reference_temperature_K": self.reference_temperature_K,
            "raw_factor_claims_implied": False,
        }


def _value(raw: Mapping[str, float], name: str) -> float:
    value = float(raw[name])
    if not np.isfinite(value):
        raise ValueError(f"raw value {name} must be finite")
    return value


def _positive_value(raw: Mapping[str, float], name: str) -> float:
    value = _value(raw, name)
    if value <= 0:
        raise ValueError(f"raw value {name} must be positive")
    return value


def _arrhenius_value(
    raw: Mapping[str, float], base: str, ea: str, temperature_K: float
) -> float:
    base_value = _positive_value(raw, base)
    ea_value = _positive_value(raw, ea)
    shift = 1.0 / float(temperature_K) - 1.0 / T_REF
    return float(base_value * np.exp(-ea_value / R_GAS * shift))


def _arrhenius_dependencies(
    base: str, ea: str, temperature_K: float
) -> tuple[str, ...]:
    return (base,) if np.isclose(temperature_K, T_REF) else (base, ea)


def _base_at_temperature(
    raw: Mapping[str, float],
    base: str,
    ea: str,
    temperature_K: float,
) -> float:
    if np.isclose(temperature_K, T_REF):
        return _positive_value(raw, base)
    return _arrhenius_value(raw, base, ea, temperature_K)


def publication_composite_definitions(
    *,
    reference_temperatures_K: Mapping[str, float] | None = None,
    physics_enabled: Sequence[str] = DEFAULT_ENABLED,
) -> tuple[CompositeDefinition, ...]:
    """Build the physical native-coordinate catalogue for one fit.

    The exact information-centroid temperatures are inputs fixed by the
    observation-only inference.  They are embedded in definitions and
    evaluators; no truth or recovery outcome can select them here.
    """

    temperatures = {
        str(name): float(value)
        for name, value in (reference_temperatures_K or {}).items()
    }
    pair_by_base = {base: ea for base, ea in ARRHENIUS_PAIRS}
    enabled = tuple(dict.fromkeys(str(name) for name in physics_enabled))
    known_physics = set(DEFAULT_ENABLED)
    unknown = set(enabled) - known_physics
    if unknown:
        raise ValueError(f"unknown publication physical transforms: {sorted(unknown)}")
    definitions: list[CompositeDefinition] = []
    replaced_bases: set[str] = set()

    for prefix in ("neg", "pos"):
        key = f"tau_d_{prefix}"
        if key in enabled:
            base = f"Ds_{prefix}_0"
            ea = f"Ea_Ds_{prefix}"
            radius = f"rs_{prefix}"
            temperature = temperatures.get(base, T_REF)
            dependencies = tuple(
                dict.fromkeys((*_arrhenius_dependencies(base, ea, temperature), radius))
            )

            def tau(raw, b=base, e=ea, r=radius, t=temperature):
                return _positive_value(raw, r) ** 2 / _base_at_temperature(
                    raw, b, e, t
                )

            definitions.append(
                CompositeDefinition(
                    claim_name=key,
                    storage_parameter=base,
                    dependencies=dependencies,
                    definition=(
                        f"{key} = {radius}^2 / Ds_{prefix}(T_c={temperature:.6g} K)"
                    ),
                    unit="s",
                    evaluator=tau,
                    mandatory_disclosures=(f"T_c={temperature:.6g} K",),
                    reference_temperature_K=temperature,
                )
            )
            replaced_bases.add(base)

        r_key = f"sei_r_{prefix}"
        if r_key in enabled:
            base = f"rou_sei_{prefix}_0"
            radius = f"rs_{prefix}"

            def sei_r(raw, b=base, r=radius):
                return _positive_value(raw, b) * _positive_value(raw, r) / 51.0

            definitions.append(
                CompositeDefinition(
                    claim_name=r_key,
                    storage_parameter=base,
                    dependencies=(base, radius),
                    definition=f"{r_key} = {base} * {radius} / 51",
                    unit="ohm*m^2",
                    evaluator=sei_r,
                    mandatory_disclosures=("delta_sei = rs/50 structural assumption",),
                )
            )
            replaced_bases.add(base)

        c_key = f"sei_c_{prefix}"
        if c_key in enabled:
            base = f"epse_sei_{prefix}"
            radius = f"rs_{prefix}"

            def sei_c(raw, b=base, r=radius):
                return _positive_value(raw, b) * 51.0 / _positive_value(raw, r)

            definitions.append(
                CompositeDefinition(
                    claim_name=c_key,
                    storage_parameter=base,
                    dependencies=(base, radius),
                    definition=f"{c_key} = {base} * 51 / {radius}",
                    unit="F/m^2",
                    evaluator=sei_c,
                    mandatory_disclosures=("delta_sei = rs/50 structural assumption",),
                )
            )
            replaced_bases.add(base)

    if "kappa_eff_neg" in enabled:
        base = "kappa_0"
        ea = "Ea_kappa"
        epse = "epse_neg"
        brug = "brug_neg"
        temperature = temperatures.get(base, T_REF)
        dependencies = tuple(
            dict.fromkeys(
                (*_arrhenius_dependencies(base, ea, temperature), epse, brug)
            )
        )

        def kappa_eff(
            raw, b=base, e=ea, p=epse, g=brug, t=temperature
        ):
            return _base_at_temperature(raw, b, e, t) * _positive_value(
                raw, p
            ) ** _value(raw, g)

        definitions.append(
            CompositeDefinition(
                claim_name="kappa_eff_neg",
                storage_parameter=base,
                dependencies=dependencies,
                definition=(
                    "kappa_eff_neg = kappa(T_c="
                    f"{temperature:.6g} K) * epse_neg^brug_neg"
                ),
                unit="S/m",
                evaluator=kappa_eff,
                mandatory_disclosures=(
                    f"T_c={temperature:.6g} K",
                    "negative-electrode effective conductivity only",
                ),
                reference_temperature_K=temperature,
            )
        )
        replaced_bases.add(base)

    for base, ea in ARRHENIUS_PAIRS:
        temperature = temperatures.get(base, T_REF)
        if base in replaced_bases or np.isclose(temperature, T_REF):
            continue
        stem = base.removesuffix("_0")

        def rate(raw, b=base, e=ea, t=temperature):
            return _arrhenius_value(raw, b, e, t)

        definitions.append(
            CompositeDefinition(
                claim_name=f"{stem}_at_Tc",
                storage_parameter=base,
                dependencies=(base, ea),
                definition=(
                    f"{stem}_at_Tc = {base} * exp[-{ea}/R * "
                    f"(1/{temperature:.6g} - 1/{T_REF})]"
                ),
                unit=_RAW_UNITS[base],
                evaluator=rate,
                mandatory_disclosures=(
                    f"T_c={temperature:.6g} K",
                    "T_c is observation-selected and varies by dataset",
                ),
                reference_temperature_K=temperature,
            )
        )

    validate_coordinate_definitions((), definitions, allow_unknown_raw=True)
    return tuple(definitions)


def validate_coordinate_definitions(
    parameter_names: Sequence[str],
    definitions: Sequence[CompositeDefinition],
    *,
    allow_unknown_raw: bool = False,
) -> tuple[str, ...]:
    """Validate one-to-many/many-to-one provenance without inferring factors."""

    names = set(parameter_names)
    storage_seen: set[str] = set()
    claims_seen: set[str] = set()
    warnings: list[str] = []
    for definition in definitions:
        if definition.storage_parameter in storage_seen:
            raise ValueError(
                f"multiple claim coordinates use storage parameter "
                f"{definition.storage_parameter}"
            )
        if definition.claim_name in claims_seen:
            raise ValueError(f"duplicate claim name {definition.claim_name}")
        storage_seen.add(definition.storage_parameter)
        claims_seen.add(definition.claim_name)
        if names and definition.storage_parameter not in names:
            raise ValueError(
                f"unknown storage parameter {definition.storage_parameter}"
            )
        unknown_dependencies = set(definition.dependencies) - names
        if names and unknown_dependencies and not allow_unknown_raw:
            raise ValueError(
                f"{definition.claim_name} has unknown dependencies: "
                f"{sorted(unknown_dependencies)}"
            )
        if definition.coordinate_class is CoordinateClass.NUMERICAL_ONLY:
            warnings.append(
                f"{definition.claim_name}: numerical-only coordinate cannot be "
                "serialized as a physical claim"
            )
    return tuple(warnings)


@dataclass(frozen=True)
class TaxonomyEvidence:
    reference_audit_complete: bool = False
    strategy_audit_complete: bool = False
    adequacy_audit_complete: bool = False
    profile_audit_complete: bool = False
    reference_sensitive_names: tuple[str, ...] = ()
    strategy_unstable_names: tuple[str, ...] = ()
    adequacy_unresolved_names: tuple[str, ...] = ()
    profile_unresolved_names: tuple[str, ...] = ()
    fit_failed_names: tuple[str, ...] = ()
    fixed_external_names: tuple[str, ...] = ()
    invalid_mapping_names: tuple[str, ...] = ()
    prior_sources: Mapping[str, str] = field(default_factory=dict)
    prior_widths_search: Mapping[str, float] = field(default_factory=dict)
    fixed_external_values: Mapping[str, float] = field(default_factory=dict)
    fixed_external_sources: Mapping[str, str] = field(default_factory=dict)
    audit_details: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    @property
    def mandatory_audits_complete(self) -> bool:
        return all(
            (
                self.reference_audit_complete,
                self.strategy_audit_complete,
                self.adequacy_audit_complete,
                self.profile_audit_complete,
            )
        )


@dataclass(frozen=True)
class ClaimRecord:
    storage_parameter: str
    claim_name: str
    category: ClaimCategory
    coordinate_class: CoordinateClass
    internal_tier: str
    definition: str
    unit: str
    dependencies: tuple[str, ...]
    mandatory_disclosures: tuple[str, ...]
    reasons: tuple[str, ...]
    provenance: Mapping[str, object]
    counts_in_headline: bool
    protected_nuisance: bool = False
    fixed_external: bool = False
    raw_factor_claims_implied: bool = False
    truth_access: bool = False

    def __post_init__(self) -> None:
        if self.truth_access:
            raise ValueError("taxonomy classification must not access truth")
        if self.raw_factor_claims_implied:
            raise ValueError("a taxonomy record may not imply composite factors")
        if (
            self.category is ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE
            and self.coordinate_class
            is not CoordinateClass.PHYSICALLY_REPORTABLE_COMPOSITE
        ):
            raise ValueError("composite candidate requires a physical composite")
        headline = self.category in (
            ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE,
            ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE,
        )
        if self.counts_in_headline != headline:
            raise ValueError("headline count must follow the frozen taxonomy")


def _tier_value(value: ParameterTier | str) -> str:
    if isinstance(value, ParameterTier):
        return value.value
    normalized = str(value).lower()
    allowed = {tier.value for tier in ParameterTier}
    if normalized not in allowed:
        raise ValueError(f"unknown internal gate tier {value!r}")
    return normalized


def classify_publication_claims(
    *,
    parameter_names: Sequence[str],
    internal_tiers: Mapping[str, ParameterTier | str],
    attempted_names: Sequence[str],
    protected_names: Sequence[str],
    evidence: TaxonomyEvidence,
    composite_definitions: Sequence[CompositeDefinition] = (),
    raw_units: Mapping[str, str] | None = None,
) -> tuple[ClaimRecord, ...]:
    """Map internal gate output to exactly one reader-facing category."""

    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("parameter_names must be non-empty and unique")
    if set(internal_tiers) != set(names):
        raise ValueError("internal_tiers must cover every parameter")
    attempted = set(attempted_names)
    protected = set(protected_names)
    unknown = (attempted | protected) - set(names)
    if unknown:
        raise ValueError(f"unknown attempted/protected names: {sorted(unknown)}")
    definitions = tuple(composite_definitions)
    mapping_warnings = validate_coordinate_definitions(names, definitions)
    definition_by_storage = {row.storage_parameter: row for row in definitions}

    reference_sensitive = set(evidence.reference_sensitive_names)
    strategy_unstable = set(evidence.strategy_unstable_names)
    adequacy_unresolved = set(evidence.adequacy_unresolved_names)
    profile_unresolved = set(evidence.profile_unresolved_names)
    fit_failed = set(evidence.fit_failed_names)
    fixed_external = set(evidence.fixed_external_names)
    invalid_mapping = set(evidence.invalid_mapping_names)
    supplied = set().union(
        reference_sensitive,
        strategy_unstable,
        adequacy_unresolved,
        profile_unresolved,
        fit_failed,
        fixed_external,
        invalid_mapping,
    )
    if supplied - set(names):
        raise ValueError(f"taxonomy evidence contains unknown names: {sorted(supplied-set(names))}")
    unknown_provenance = (
        set(evidence.prior_sources)
        | set(evidence.prior_widths_search)
        | set(evidence.fixed_external_values)
        | set(evidence.fixed_external_sources)
        | set(evidence.audit_details)
    ) - set(names)
    if unknown_provenance:
        raise ValueError(
            f"taxonomy provenance contains unknown names: {sorted(unknown_provenance)}"
        )

    records: list[ClaimRecord] = []
    for name in names:
        tier = _tier_value(internal_tiers[name])
        if name in protected and tier == ParameterTier.STRICT.value:
            raise ValueError(
                f"protected nuisance {name} cannot carry the internal STRICT tier"
            )
        definition = definition_by_storage.get(name)
        coordinate_class = (
            CoordinateClass.RAW
            if definition is None
            else definition.coordinate_class
        )
        claim_name = name if definition is None else definition.claim_name
        claim_definition = (
            f"raw parameter {name}"
            if definition is None
            else definition.definition
        )
        unit = (
            (raw_units or _RAW_UNITS).get(name, "unspecified")
            if definition is None
            else definition.unit
        )
        dependencies = (name,) if definition is None else definition.dependencies
        disclosures = list(
            () if definition is None else definition.mandatory_disclosures
        )
        reasons: list[str] = []
        provenance: dict[str, object] = {
            "internal_gate_tier": tier,
            "audit_details": dict(evidence.audit_details.get(name, {})),
        }

        # Frozen precedence: REJECTED > ADEQUACY > STRATEGY > REFERENCE >
        # positive > FIXED_EXTERNAL > PROTECTED_NUISANCE.
        if name in fit_failed:
            category = ClaimCategory.REJECTED
            reasons.append("FIT_FAILED")
        elif name in profile_unresolved:
            category = ClaimCategory.REJECTED
            reasons.append("PROFILE_UNRESOLVED")
        elif name in invalid_mapping:
            category = ClaimCategory.REJECTED
            reasons.append("INVALID_COORDINATE_MAPPING")
        elif coordinate_class is CoordinateClass.NUMERICAL_ONLY:
            category = ClaimCategory.REJECTED
            reasons.append("NUMERICAL_ONLY_COORDINATE")
        elif tier == ParameterTier.PRIOR_DOMINATED.value:
            category = ClaimCategory.REJECTED
            reasons.append("PRIOR_DOMINATED")
        elif (
            tier == ParameterTier.FIXED.value
            and name in attempted
            and name not in protected
        ):
            category = ClaimCategory.REJECTED
            reasons.append("FAILED_GATE_AFTER_ATTEMPT")
        elif (
            tier == ParameterTier.FIXED.value
            and name not in fixed_external
            and name not in protected
        ):
            category = ClaimCategory.REJECTED
            reasons.append("UNDECLARED_FIXED_VALUE")
        elif name in adequacy_unresolved:
            category = ClaimCategory.ADEQUACY_UNRESOLVED
            reasons.append("MODEL_OR_FREQUENCY_ADEQUACY")
        elif name in strategy_unstable:
            category = ClaimCategory.STRATEGY_UNSTABLE
            reasons.append("STABILITY_SCORE_BELOW_0.75")
        elif name in reference_sensitive:
            category = ClaimCategory.REFERENCE_SENSITIVE
            reasons.append("REFERENCE_SHIFT_OVER_0.5_LOCAL_CI95")
        elif tier in (
            ParameterTier.STRICT.value,
            ParameterTier.PRIOR_INFORMED.value,
        ):
            if not evidence.mandatory_audits_complete:
                raise ValueError(
                    f"positive reader-facing claim {name} requires all mandatory audits"
                )
            if tier == ParameterTier.PRIOR_INFORMED.value:
                category = ClaimCategory.PRIOR_INFORMED
                reasons.append("POSTERIOR_GATE_WITH_PRIOR_CURVATURE")
            elif coordinate_class is CoordinateClass.RAW:
                category = ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE
                reasons.append("LOCAL_DATA_GATE_IN_RAW_COORDINATE")
            else:
                category = ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE
                reasons.append("LOCAL_DATA_GATE_IN_PHYSICAL_COMPOSITE_COORDINATE")
        elif name in fixed_external:
            category = ClaimCategory.FIXED_EXTERNAL
            reasons.append("DECLARED_EXTERNAL_DESIGN_VALUE")
        elif name in protected:
            category = ClaimCategory.PROTECTED_NUISANCE
            reasons.append("KEPT_FREE_FOR_FIT_HEALTH")
        else:
            category = ClaimCategory.REJECTED
            reasons.append("NO_REPORTABLE_GATE_OUTCOME")

        details = evidence.audit_details.get(name, {})
        if category is ClaimCategory.PRIOR_INFORMED:
            source = evidence.prior_sources.get(name)
            width = evidence.prior_widths_search.get(name)
            if not source or width is None or not np.isfinite(width) or width <= 0:
                raise ValueError(
                    f"PRIOR_INFORMED {name} requires prior source and positive width"
                )
            disclosures.extend(
                (f"prior_source={source}", f"prior_sigma_search={float(width):.12g}")
            )
            provenance["prior_source"] = source
            provenance["prior_sigma_search"] = float(width)
        elif category is ClaimCategory.FIXED_EXTERNAL:
            value = evidence.fixed_external_values.get(name)
            source = evidence.fixed_external_sources.get(name)
            if value is None or not np.isfinite(value) or not source:
                raise ValueError(
                    f"FIXED_EXTERNAL {name} requires a finite value and source"
                )
            disclosures.extend(
                (f"fixed_value={float(value):.12g}", f"fixed_source={source}")
            )
            provenance["fixed_value"] = float(value)
            provenance["fixed_source"] = source
        elif category is ClaimCategory.PROTECTED_NUISANCE:
            disclosures.extend(
                (
                    "kept free for fit health",
                    "uncertainty unbounded by design",
                    "excluded from all recovery scoring",
                )
            )
        elif category is ClaimCategory.REFERENCE_SENSITIVE:
            required = {"max_shift_over_local_width", "offending_nuisance"}
            if not required <= set(details):
                raise ValueError(
                    f"REFERENCE_SENSITIVE {name} missing sensitivity provenance"
                )
            disclosures.append(
                "reference_sensitivity="
                f"{float(details['max_shift_over_local_width']):.12g};"
                f"nuisance={details['offending_nuisance']}"
            )
        elif category is ClaimCategory.STRATEGY_UNSTABLE:
            required = {"stability_score", "replicate_set"}
            if not required <= set(details):
                raise ValueError(
                    f"STRATEGY_UNSTABLE {name} missing stability provenance"
                )
            disclosures.append(
                f"stability_score={float(details['stability_score']):.12g};"
                f"replicate_set={details['replicate_set']}"
            )
        elif category is ClaimCategory.ADEQUACY_UNRESOLVED:
            if "failed_adequacy_check" not in details:
                raise ValueError(
                    f"ADEQUACY_UNRESOLVED {name} missing adequacy provenance"
                )
            disclosures.append(
                f"failed_adequacy_check={details['failed_adequacy_check']}"
            )

        if definition is not None and any(
            definition.claim_name in warning for warning in mapping_warnings
        ):
            reasons.append("INVALID_MAPPING_WARNING_RETAINED")
        records.append(
            ClaimRecord(
                storage_parameter=name,
                claim_name=claim_name,
                category=category,
                coordinate_class=coordinate_class,
                internal_tier=tier,
                definition=claim_definition,
                unit=unit,
                dependencies=dependencies,
                mandatory_disclosures=tuple(disclosures),
                reasons=tuple(reasons),
                provenance=provenance,
                counts_in_headline=category
                in (
                    ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE,
                    ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE,
                ),
                protected_nuisance=name in protected,
                fixed_external=name in fixed_external,
            )
        )
    return tuple(records)


_UNSAFE_READER_PATTERNS = (
    re.compile(r"\bSTRICT\b", re.IGNORECASE),
    re.compile(r"\bidentified\b", re.IGNORECASE),
    re.compile(r"\bidentifiable\b", re.IGNORECASE),
    re.compile(r"\bmeasured\b", re.IGNORECASE),
    re.compile(r"globally\s+identifiable", re.IGNORECASE),
    re.compile(r"calibrated(?:\s+uncertainty|\s+coverage)?", re.IGNORECASE),
)


def validate_reader_claim_text(text: str) -> None:
    if not text.strip():
        raise ValueError("reader-facing claim text cannot be empty")
    for pattern in _UNSAFE_READER_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"unsafe reader-facing certainty language: {pattern.pattern}"
            )


@dataclass(frozen=True)
class ClaimTables:
    raw_claims: tuple[dict[str, object], ...]
    composite_claims: tuple[dict[str, object], ...]
    prior_informed: tuple[dict[str, object], ...]
    audit: tuple[dict[str, object], ...]
    provenance: tuple[dict[str, object], ...]
    invalid_mapping_warnings: tuple[str, ...]


def serialize_claim_tables(
    records: Sequence[ClaimRecord],
    *,
    invalid_mapping_warnings: Sequence[str] = (),
) -> ClaimTables:
    """Serialize categories without exposing the internal STRICT label."""

    warnings = tuple(str(value) for value in invalid_mapping_warnings)
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for record in records:
        row = {
            "storage_parameter": record.storage_parameter,
            "claim_name": record.claim_name,
            "category": record.category.value,
            "coordinate_class": record.coordinate_class.value,
            "definition": record.definition,
            "unit": record.unit,
            "dependencies": list(record.dependencies),
            "mandatory_disclosures": list(record.mandatory_disclosures),
            "reasons": list(record.reasons),
            "provenance": dict(record.provenance),
            "counts_in_headline": record.counts_in_headline,
            "protected_nuisance": record.protected_nuisance,
            "fixed_external": record.fixed_external,
            "raw_factor_claims_implied": False,
            "calibrated_uncertainty_implied": False,
            "global_identifiability_implied": False,
            "truth_access": False,
        }
        rows.append(row)
        provenance.append(
            {
                "storage_parameter": record.storage_parameter,
                "claim_name": record.claim_name,
                "internal_gate_tier": record.internal_tier,
                "reader_category": record.category.value,
                "coordinate_class": record.coordinate_class.value,
                "dependencies": list(record.dependencies),
                "raw_factor_claims_implied": False,
                "protected_nuisance": record.protected_nuisance,
                "fixed_external": record.fixed_external,
                "provenance": dict(record.provenance),
            }
        )
    raw = tuple(
        row
        for record, row in zip(records, rows, strict=True)
        if record.category is ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE
    )
    composite = tuple(
        row
        for record, row in zip(records, rows, strict=True)
        if record.category is ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE
    )
    prior = tuple(
        row
        for record, row in zip(records, rows, strict=True)
        if record.category is ClaimCategory.PRIOR_INFORMED
    )
    audit = tuple(
        row
        for record, row in zip(records, rows, strict=True)
        if record.category
        not in (
            ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE,
            ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE,
            ClaimCategory.PRIOR_INFORMED,
        )
    )
    return ClaimTables(raw, composite, prior, audit, tuple(provenance), warnings)


def serialize_reader_claim(record: ClaimRecord, claim_text: str) -> dict[str, object]:
    """Serialize one safe sentence together with mandatory coordinate fields."""

    validate_reader_claim_text(claim_text)
    if record.category not in (
        ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE,
        ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE,
        ClaimCategory.PRIOR_INFORMED,
    ):
        raise ValueError("negative/fixed taxonomy categories are audit-only")
    if (
        record.category is ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE
        and (not record.definition or not record.unit or not record.dependencies)
    ):
        raise ValueError("composite claim lacks mandatory mapping fields")
    return {
        "claim_text": claim_text,
        "category": record.category.value,
        "claim_name": record.claim_name,
        "definition": record.definition,
        "unit": record.unit,
        "dependencies": list(record.dependencies),
        "mandatory_disclosures": list(record.mandatory_disclosures),
        "provenance": dict(record.provenance),
        "raw_factor_claims_implied": False,
        "protected_nuisance": record.protected_nuisance,
        "fixed_external": record.fixed_external,
        "calibrated_uncertainty_implied": False,
        "global_identifiability_implied": False,
    }


@dataclass(frozen=True)
class ScoreClassMetrics:
    coordinate_class: str
    candidate_count: int
    pass_count: int
    eligible_point_pass_count: int
    precision: float
    recall_proxy: float


@dataclass(frozen=True)
class SeparatedClaimScore:
    raw_rows: tuple[dict[str, object], ...]
    composite_rows: tuple[dict[str, object], ...]
    prior_informed_rows: tuple[dict[str, object], ...]
    raw_metrics: ScoreClassMetrics
    composite_metrics: ScoreClassMetrics
    threshold: float = RAW_RECOVERY_TOLERANCE
    evaluation_only: bool = True
    truth_access: bool = True
    changes_estimator: bool = False
    changes_taxonomy: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_only or not self.truth_access:
            raise ValueError("claim scoring must remain evaluation-only")
        if self.changes_estimator or self.changes_taxonomy:
            raise ValueError("post-hoc scoring cannot change inference or taxonomy")
        if self.threshold != RAW_RECOVERY_TOLERANCE:
            raise ValueError("the 10 percent recovery threshold is immutable")


def _relative_error(estimate: float, truth: float) -> float:
    if not np.isfinite(estimate) or not np.isfinite(truth):
        return float("nan")
    return float(abs(estimate - truth) / max(abs(truth), 1e-30))


def _metrics(
    coordinate_class: str,
    candidate_rows: Sequence[Mapping[str, object]],
    eligible_pass_count: int,
) -> ScoreClassMetrics:
    count = len(candidate_rows)
    passed = sum(bool(row["point_recovery_pass"]) for row in candidate_rows)
    return ScoreClassMetrics(
        coordinate_class=coordinate_class,
        candidate_count=count,
        pass_count=passed,
        eligible_point_pass_count=int(eligible_pass_count),
        precision=float(passed / count) if count else float("nan"),
        recall_proxy=(
            float(passed / eligible_pass_count)
            if eligible_pass_count
            else float("nan")
        ),
    )


def score_publication_claims(
    *,
    records: Sequence[ClaimRecord],
    estimated_raw: Mapping[str, float],
    truth_raw: Mapping[str, float],
    composite_definitions: Sequence[CompositeDefinition],
    evaluation_only: bool = False,
) -> SeparatedClaimScore:
    """Score raw, composite, and prior-informed rows without pooling them."""

    if not evaluation_only:
        raise ValueError("set evaluation_only=True after inference is immutable")
    record_tuple = tuple(records)
    definition_by_storage = {
        row.storage_parameter: row for row in composite_definitions
    }
    if len(definition_by_storage) != len(tuple(composite_definitions)):
        raise ValueError("composite definitions must have unique storage parameters")

    raw_rows: list[dict[str, object]] = []
    composite_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    raw_eligible_pass = 0
    composite_eligible_pass = 0
    excluded = {
        ClaimCategory.PROTECTED_NUISANCE,
        ClaimCategory.FIXED_EXTERNAL,
    }
    for record in record_tuple:
        if (
            record.category in excluded
            or record.protected_nuisance
            or record.fixed_external
        ):
            continue
        if record.coordinate_class is CoordinateClass.NUMERICAL_ONLY:
            continue
        if record.coordinate_class is CoordinateClass.RAW:
            if record.storage_parameter not in truth_raw:
                raise ValueError(f"truth missing raw parameter {record.storage_parameter}")
            estimate = float(estimated_raw.get(record.storage_parameter, np.nan))
            truth = float(truth_raw[record.storage_parameter])
        else:
            definition = definition_by_storage.get(record.storage_parameter)
            if definition is None:
                raise ValueError(
                    f"missing composite definition for {record.storage_parameter}"
                )
            if (
                definition.claim_name != record.claim_name
                or definition.definition != record.definition
                or definition.dependencies != record.dependencies
                or definition.coordinate_class is not record.coordinate_class
            ):
                raise ValueError(
                    f"scoring definition does not match frozen taxonomy for "
                    f"{record.storage_parameter}"
                )
            estimate = definition.evaluate(estimated_raw)
            truth = definition.evaluate(truth_raw)
        error = _relative_error(estimate, truth)
        row = {
            "storage_parameter": record.storage_parameter,
            "claim_name": record.claim_name,
            "category": record.category.value,
            "coordinate_class": record.coordinate_class.value,
            "estimated_value": estimate,
            "truth_value": truth,
            "relative_error": error,
            "error_tolerance": RAW_RECOVERY_TOLERANCE,
            "point_recovery_pass": bool(
                np.isfinite(error) and error <= RAW_RECOVERY_TOLERANCE
            ),
            "candidate": record.counts_in_headline,
            "uncertainty_certified": False,
            "raw_factor_claims_implied": False,
            "changes_estimator": False,
            "changes_taxonomy": False,
            "evaluation_only": True,
        }
        if row["point_recovery_pass"]:
            if record.coordinate_class is CoordinateClass.RAW:
                raw_eligible_pass += 1
            elif (
                record.coordinate_class
                is CoordinateClass.PHYSICALLY_REPORTABLE_COMPOSITE
            ):
                composite_eligible_pass += 1
        if record.category is ClaimCategory.PRIOR_INFORMED:
            prior_rows.append(row)
        elif record.coordinate_class is CoordinateClass.RAW:
            raw_rows.append(row)
        elif (
            record.coordinate_class
            is CoordinateClass.PHYSICALLY_REPORTABLE_COMPOSITE
        ):
            composite_rows.append(row)

    raw_candidate_rows = [
        row
        for row in raw_rows
        if row["category"] == ClaimCategory.DATA_SUPPORTED_RAW_CANDIDATE.value
    ]
    composite_candidate_rows = [
        row
        for row in composite_rows
        if row["category"]
        == ClaimCategory.DATA_SUPPORTED_COMPOSITE_CANDIDATE.value
    ]
    return SeparatedClaimScore(
        raw_rows=tuple(raw_rows),
        composite_rows=tuple(composite_rows),
        prior_informed_rows=tuple(prior_rows),
        raw_metrics=_metrics("RAW", raw_candidate_rows, raw_eligible_pass),
        composite_metrics=_metrics(
            "PHYSICALLY_REPORTABLE_COMPOSITE",
            composite_candidate_rows,
            composite_eligible_pass,
        ),
    )
