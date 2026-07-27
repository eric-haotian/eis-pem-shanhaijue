# learning AI at www.haotianblog.com
"""Physics-based prior constraints for EIS parameter identification.

Encodes domain knowledge — physical bounds, parameter orderings,
ratio constraints, and sum constraints — as soft penalty terms that
can be added to the optimiser objective function.

Soft priors add a quadratic penalty when violated, allowing the optimiser
to explore nearby space while discouraging physically implausible solutions.
Hard bounds remain in :class:`~eis_pem.parameters.ParameterSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Prior base and concrete types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsPrior:
    """Base class for a single physics-based constraint.

    Attributes
    ----------
    name : str
        Short identifier for the constraint.
    constraint_type : str
        One of ``"bound"``, ``"ratio"``, ``"ordering"``, ``"sum"``,
        ``"penalty"``.
    description : str
        Human-readable explanation.
    weight : float
        Relative importance of this prior in the penalty sum.
    """

    name: str
    constraint_type: Literal["bound", "ratio", "ordering", "sum", "penalty"]
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Prior weight must be non-negative, got {self.weight}")


@dataclass(frozen=True)
class BoundPrior(PhysicsPrior):
    """Soft bound on a single parameter.

    The penalty is zero inside ``[lower, upper]`` and grows quadratically
    outside.

    Attributes
    ----------
    parameter : str
        Parameter name to constrain.
    lower : float | None
        Soft lower bound (None = no lower constraint).
    upper : float | None
        Soft upper bound (None = no upper constraint).
    """

    parameter: str = ""
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lower is not None and self.upper is not None:
            if self.lower >= self.upper:
                raise ValueError(
                    f"BoundPrior '{self.name}': lower ({self.lower}) "
                    f"must be < upper ({self.upper})"
                )


@dataclass(frozen=True)
class RatioPrior(PhysicsPrior):
    """Constraint on the ratio of two parameters.

    Penalises when ``numerator / denominator`` falls outside
    ``expected_range``.

    Attributes
    ----------
    numerator : str
        Parameter name for the numerator.
    denominator : str
        Parameter name for the denominator.
    expected_range : tuple[float, float]
        ``(min_ratio, max_ratio)`` — acceptable ratio range.
    """

    numerator: str = ""
    denominator: str = ""
    expected_range: tuple[float, float] = (0.01, 100.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        lo, hi = self.expected_range
        if lo >= hi:
            raise ValueError(f"RatioPrior '{self.name}': min_ratio must be < max_ratio")
        if lo <= 0:
            raise ValueError(f"RatioPrior '{self.name}': min_ratio must be positive")


@dataclass(frozen=True)
class OrderingPrior(PhysicsPrior):
    """Constraint that one parameter must be smaller than another.

    Penalises when ``smaller >= larger``.

    Attributes
    ----------
    smaller : str
        Parameter name that should be smaller.
    larger : str
        Parameter name that should be larger.
    """

    smaller: str = ""
    larger: str = ""


@dataclass(frozen=True)
class SumPrior(PhysicsPrior):
    """Constraint on the sum of multiple parameters.

    Penalises when ``Σ params`` falls outside ``expected_range``.

    Attributes
    ----------
    parameters : tuple[str, ...]
        Parameter names to sum.
    expected_range : tuple[float, float]
        ``(min_sum, max_sum)`` — acceptable range for the sum.
    """

    parameters: tuple[str, ...] = ()
    expected_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        lo, hi = self.expected_range
        if lo > hi:
            raise ValueError(f"SumPrior '{self.name}': min must be <= max")


# ---------------------------------------------------------------------------
# Penalty computation
# ---------------------------------------------------------------------------


def evaluate_prior(
    prior: PhysicsPrior,
    param_values: dict[str, float],
) -> float:
    """Evaluate a single prior and return its penalty contribution.

    Returns 0 if the constraint is satisfied, a positive value otherwise.
    Missing parameters are silently skipped (return 0).
    """
    if isinstance(prior, BoundPrior):
        val = param_values.get(prior.parameter)
        if val is None:
            return 0.0
        penalty = 0.0
        if prior.lower is not None and val < prior.lower:
            penalty += ((prior.lower - val) / max(abs(prior.lower), 1e-12)) ** 2
        if prior.upper is not None and val > prior.upper:
            penalty += ((val - prior.upper) / max(abs(prior.upper), 1e-12)) ** 2
        return prior.weight * penalty

    if isinstance(prior, RatioPrior):
        num = param_values.get(prior.numerator)
        den = param_values.get(prior.denominator)
        if num is None or den is None or abs(den) < 1e-30:
            return 0.0
        ratio = num / den
        lo, hi = prior.expected_range
        penalty = 0.0
        if ratio < lo:
            penalty = ((lo - ratio) / lo) ** 2
        elif ratio > hi:
            penalty = ((ratio - hi) / hi) ** 2
        return prior.weight * penalty

    if isinstance(prior, OrderingPrior):
        val_small = param_values.get(prior.smaller)
        val_large = param_values.get(prior.larger)
        if val_small is None or val_large is None:
            return 0.0
        if val_small >= val_large:
            violation = (val_small - val_large) / max(abs(val_large), 1e-12)
            return prior.weight * violation**2
        return 0.0

    if isinstance(prior, SumPrior):
        vals = [param_values.get(p) for p in prior.parameters]
        if any(v is None for v in vals):
            return 0.0
        total = sum(v for v in vals if v is not None)
        lo, hi = prior.expected_range
        penalty = 0.0
        if total < lo:
            penalty = ((lo - total) / max(abs(lo), 1e-12)) ** 2
        elif total > hi:
            penalty = ((total - hi) / max(abs(hi), 1e-12)) ** 2
        return prior.weight * penalty

    return 0.0


def prior_penalty(
    priors: Sequence[PhysicsPrior],
    param_names: Sequence[str],
    theta: NDArray[np.floating],
) -> float:
    """Compute total penalty from all physics priors.

    Parameters
    ----------
    priors : Sequence[PhysicsPrior]
        List of physics prior constraints.
    param_names : Sequence[str]
        Parameter names corresponding to ``theta``.
    theta : NDArray
        Current parameter values (physical space).

    Returns
    -------
    float
        Total weighted penalty (0 if all priors satisfied).
    """
    values = dict(zip(param_names, theta.flat))
    return sum(evaluate_prior(p, values) for p in priors)


def prior_penalty_vector(
    priors: Sequence[PhysicsPrior],
    param_names: Sequence[str],
    theta: NDArray[np.floating],
) -> FloatArray:
    """Return a penalty vector (one element per prior) for augmented residuals.

    This vector can be appended to the residual vector in least-squares
    optimisation to implement soft regularisation.
    """
    values = dict(zip(param_names, theta.flat))
    penalties = np.array([
        np.sqrt(max(evaluate_prior(p, values), 0.0)) for p in priors
    ])
    return penalties


def check_prior_violations(
    priors: Sequence[PhysicsPrior],
    param_names: Sequence[str],
    theta: NDArray[np.floating],
) -> list[str]:
    """Return human-readable descriptions of violated priors.

    Returns an empty list if all priors are satisfied.
    """
    values = dict(zip(param_names, theta.flat))
    violations: list[str] = []
    for p in priors:
        penalty = evaluate_prior(p, values)
        if penalty > 1e-12:
            violations.append(f"[{p.name}] {p.description} (penalty={penalty:.4f})")
    return violations


# ---------------------------------------------------------------------------
# Built-in battery priors
# ---------------------------------------------------------------------------


def lithium_ion_priors() -> tuple[PhysicsPrior, ...]:
    """Standard physics priors for lithium-ion battery EIS.

    These encode commonly accepted physical constraints for EIS analysis
    of lithium-ion cells.

    Returns
    -------
    tuple[PhysicsPrior, ...]
        Default set of prior constraints.
    """
    return (
        # --- Resistance bounds ---
        BoundPrior(
            name="Rs_physical",
            constraint_type="bound",
            description="Ohmic resistance typically 1-100 mΩ for Li-ion",
            weight=1.0,
            parameter="Rs",
            lower=1e-4,
            upper=0.5,
        ),
        BoundPrior(
            name="Rct_physical",
            constraint_type="bound",
            description="Charge-transfer resistance typically < 1 Ω",
            weight=1.0,
            parameter="Rct",
            lower=1e-5,
            upper=5.0,
        ),
        BoundPrior(
            name="R1_physical",
            constraint_type="bound",
            description="SEI resistance typically < Rs",
            weight=0.5,
            parameter="R1",
            lower=1e-6,
            upper=1.0,
        ),
        BoundPrior(
            name="R2_physical",
            constraint_type="bound",
            description="Charge-transfer resistance",
            weight=0.5,
            parameter="R2",
            lower=1e-6,
            upper=5.0,
        ),
        # --- CPE exponent bounds ---
        BoundPrior(
            name="alpha_CPE_range",
            constraint_type="bound",
            description="CPE exponent α typically 0.7-1.0 for Li-ion",
            weight=2.0,
            parameter="alpha",
            lower=0.6,
            upper=1.0,
        ),
        BoundPrior(
            name="alpha1_CPE_range",
            constraint_type="bound",
            description="CPE exponent α₁ typically 0.7-1.0",
            weight=2.0,
            parameter="alpha1",
            lower=0.6,
            upper=1.0,
        ),
        BoundPrior(
            name="alpha2_CPE_range",
            constraint_type="bound",
            description="CPE exponent α₂ typically 0.7-1.0",
            weight=2.0,
            parameter="alpha2",
            lower=0.6,
            upper=1.0,
        ),
        # --- Time constant ordering ---
        # τ₁ = R₁Q₁^(1/α₁) < τ₂ = R₂Q₂^(1/α₂)
        # SEI process is faster than charge-transfer
        OrderingPrior(
            name="R1_lt_R2",
            constraint_type="ordering",
            description="SEI resistance (R₁) typically < charge-transfer (R₂)",
            weight=1.0,
            smaller="R1",
            larger="R2",
        ),
        # --- Total resistance ---
        SumPrior(
            name="total_R_bound",
            constraint_type="sum",
            description="Total DC resistance (Rs+R1+R2) typically < 2 Ω",
            weight=0.5,
            parameters=("Rs", "R1", "R2"),
            expected_range=(1e-4, 2.0),
        ),
    )


def pemfc_priors() -> tuple[PhysicsPrior, ...]:
    """Physics priors for PEM fuel cell EIS.

    Returns
    -------
    tuple[PhysicsPrior, ...]
        PEMFC-specific prior constraints.
    """
    return (
        BoundPrior(
            name="Rs_membrane",
            constraint_type="bound",
            description="Membrane resistance typically 10-200 mΩ·cm²",
            weight=1.0,
            parameter="Rs",
            lower=0.01,
            upper=0.2,
        ),
        BoundPrior(
            name="Rct_cathode",
            constraint_type="bound",
            description="Cathode charge-transfer typically dominant",
            weight=1.0,
            parameter="Rct",
            lower=0.01,
            upper=10.0,
        ),
        BoundPrior(
            name="alpha_CPE",
            constraint_type="bound",
            description="CPE exponent for PEMFC typically 0.75-0.95",
            weight=2.0,
            parameter="alpha",
            lower=0.7,
            upper=0.95,
        ),
    )

def lithium_ion_seis_priors() -> tuple[PhysicsPrior, ...]:
    """Standard physics priors for lithium-ion battery SEIS model.

    These encode accepted physical constraints for SEIS micro-macro models,
    such as porosity bounds, positive diffusion vs kinetics, etc.
    """
    return (
        # Porosities must sum to < 1.0 (some solid material must exist)
        SumPrior(
            name="Porosity bounds (neg)",
            constraint_type="sum",
            description="Sum of porosities in negative electrode must be < 1.0",
            parameters=("epse_neg", "epsf_neg"),
            expected_range=(0.0, 0.99)
        ),
        SumPrior(
            name="Porosity bounds (pos)",
            constraint_type="sum",
            description="Sum of porosities in positive electrode must be < 1.0",
            parameters=("epse_pos", "epsf_pos"),
            expected_range=(0.0, 0.99)
        ),
        # Transport vs Kinetics (usually diffusion activation energy is modest)
        BoundPrior("Ea_Ds_neg", "bound", "Activation energy", 1.0, "Ea_Ds_neg", lower=1000.0, upper=100000.0),
        BoundPrior("Ea_Ds_pos", "bound", "Activation energy", 1.0, "Ea_Ds_pos", lower=1000.0, upper=100000.0),
        BoundPrior("Ea_k_neg", "bound", "Activation energy", 1.0, "Ea_k_neg", lower=1000.0, upper=100000.0),
        BoundPrior("Ea_k_pos", "bound", "Activation energy", 1.0, "Ea_k_pos", lower=1000.0, upper=100000.0),
        # Tortuosity Bruggeman > 1.0
        BoundPrior("brug_neg", "bound", "Bruggeman exponent > 1.0", 1.0, "brug_neg", lower=1.0, upper=5.0),
        BoundPrior("brug_pos", "bound", "Bruggeman exponent > 1.0", 1.0, "brug_pos", lower=1.0, upper=5.0),
        BoundPrior("brug_sep", "bound", "Bruggeman exponent > 1.0", 1.0, "brug_sep", lower=1.0, upper=5.0),
        # Contact resistance > 0
        BoundPrior("R_contact", "bound", "Contact resistance >= 0", 1.0, "R_contact", lower=0.0),
    )
