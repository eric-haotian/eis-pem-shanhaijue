# learning AI at www.haotianblog.com
"""Arrhenius reparameterization for the SEIS model (Strategies 1+2).

The Gen-1 SEIS model uses the *centered* Arrhenius form

    k(T) = k_0 * exp(-Ea / R * (1/T - 1/T_0)),   T_0 = 298.15 K,

so ``k_0`` already **is** the rate at 298.15 K — naively applying the
textbook substitution ``k_Tref = k_0 * exp(-Ea/R/T_ref)`` on top of it
would *break* the centering and make the collinearity worse.

The residual problem is subtler: the measured information is not
symmetric in temperature (cold spectra carry different relative
structure than hot ones), so the *information centroid* of ``1/T`` sits
away from ``1/T_0`` and the stacked Jacobian columns of ``(k_0, Ea)``
still correlate at rho = -0.79 .. -0.89 on the 3x3 grid.  The correct
generalization is to re-center each pair at its information-centroid
temperature ``T_c``:

    k_c = k_0 * exp(-Ea / R * (1/T_c - 1/T_0))        (rate at T_c)

In search (log10) coordinates this is a shear, and the decorrelating
shift has the closed form

    1/T_c - 1/T_0 = - <J_k, J_Ea> / (<J_k, J_k> * ln(10) * Ea / R)

where the inner products are Jacobian-column Grams (available from the
data Fisher matrix) and Ea is the reference activation energy.  With
``T_c = T_0`` the transformation is the identity, which is also the
honest fallback when no Jacobian information is available.

Identifiability claims made in the new coordinates apply to the rate at
``T_c`` — the pipeline reports both coordinate systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .parameters import ParameterSpec

from .priors import GaussianParameterPrior, PriorSet

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

#: (base coefficient, activation energy) pairs sharing an Arrhenius law.
ARRHENIUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("Ds_neg_0", "Ea_Ds_neg"),
    ("Ds_pos_0", "Ea_Ds_pos"),
    ("k_neg_0", "Ea_k_neg"),
    ("k_pos_0", "Ea_k_pos"),
    ("kappa_0", "Ea_kappa"),
)

T_REF = 298.15  # K — the Gen-1 model's own centering temperature T_0
R_GAS = 8.314   # J/mol/K


def _default_ea_values() -> dict[str, float]:
    """Literature default activation energies from the Gen-1 spec table.

    These are exactly the values the Gen-1 ``StackedSEISModel`` uses when
    an Ea is not among the fitted parameters, which keeps subset
    coordinates consistent with full-set coordinates.
    """

    from .seis_model import all_seis_parameter_specs

    return {
        spec.name: float(spec.initial_value)
        for spec in all_seis_parameter_specs()
        if spec.name.startswith("Ea_")
    }


_DEFAULT_EA = _default_ea_values()


def optimal_reference_temperatures(
    data_fim: FloatArray,
    specs: Sequence[ParameterSpec],
    theta_reference: NDArray[np.floating],
    pairs: Sequence[tuple[str, str]] = ARRHENIUS_PAIRS,
    max_shift_per_kelvin: float = 40.0,
) -> dict[str, float]:
    """Information-centroid reference temperature per Arrhenius pair.

    ``data_fim`` may be any positive multiple of ``J^T J`` in search
    space (the noise scaling cancels in the ratio).  The decorrelating
    inverse-temperature shift is clamped to ``T_0 +- max_shift_per_kelvin``
    to keep the induced bound/prior scaling mild.
    """

    names = [spec.name for spec in specs]
    theta = np.asarray(theta_reference, dtype=float)
    temperatures: dict[str, float] = {}
    for base, ea in pairs:
        if base not in names or ea not in names:
            continue
        i, j = names.index(base), names.index(ea)
        gram_kk = float(data_fim[i, i])
        gram_ke = float(data_fim[i, j])
        if gram_kk <= 0:
            temperatures[base] = T_REF
            continue
        ea_value = float(theta[j])
        # In search space s = log10, the recentring is the shear
        # s_k0 = s_kc + Ea/(R ln10) * shift, so the transformed Ea column
        # is e + k * (Ea/R) * shift (the ln10 factors cancel through
        # dEa/ds_Ea = Ea ln10).  Zeroing <k, e_new> gives:
        shift = -gram_ke / (gram_kk * ea_value / R_GAS)
        t_c = 1.0 / (1.0 / T_REF + shift)
        t_c = float(np.clip(t_c, T_REF - max_shift_per_kelvin, T_REF + max_shift_per_kelvin))
        temperatures[base] = t_c
    return temperatures


@dataclass(frozen=True)
class ArrheniusReparameterizer:
    """Bidirectional (k_0, Ea) <-> (k_c, Ea) transformation.

    ``reference_temperatures`` maps each base-coefficient name to its
    T_c; pairs absent from the mapping (or with T_c == T_0) pass through
    unchanged.  The transformation acts only on pairs whose *both*
    members appear in the parameter set being transformed.
    """

    reference_temperatures: Mapping[str, float] = field(default_factory=dict)
    pairs: tuple[tuple[str, str], ...] = ARRHENIUS_PAIRS

    def _active_pairs(
        self, names: Sequence[str]
    ) -> list[tuple[int, int | None, float, float]]:
        """(base index, Ea index or None, shift, static Ea) per pair.

        When the base coefficient is present but its activation energy
        is not (a subset fit with Ea frozen at the literature value),
        the transformation still applies with the *static* default Ea —
        otherwise subset coordinates would silently diverge from the
        full-set coordinates.
        """

        name_list = list(names)
        active = []
        for base, ea in self.pairs:
            t_c = float(self.reference_temperatures.get(base, T_REF))
            if base not in name_list or np.isclose(t_c, T_REF):
                continue
            shift = 1.0 / t_c - 1.0 / T_REF
            ea_index = name_list.index(ea) if ea in name_list else None
            active.append((name_list.index(base), ea_index, shift,
                           _DEFAULT_EA[ea]))
        return active

    # -- vector transforms ------------------------------------------------

    def from_original(
        self, names: Sequence[str], theta_original: NDArray[np.floating]
    ) -> FloatArray:
        """(k_0, Ea) -> (k_c, Ea):  k_c = k_0 * exp(-Ea/R * (1/Tc - 1/T0))."""

        theta = np.asarray(theta_original, dtype=float).copy()
        for i, j, shift, ea_default in self._active_pairs(names):
            ea = theta[j] if j is not None else ea_default
            theta[i] = theta[i] * np.exp(-ea / R_GAS * shift)
        return theta

    def to_original(
        self, names: Sequence[str], theta_reparam: NDArray[np.floating]
    ) -> FloatArray:
        """(k_c, Ea) -> (k_0, Ea):  k_0 = k_c * exp(+Ea/R * (1/Tc - 1/T0))."""

        theta = np.asarray(theta_reparam, dtype=float).copy()
        for i, j, shift, ea_default in self._active_pairs(names):
            ea = theta[j] if j is not None else ea_default
            theta[i] = theta[i] * np.exp(ea / R_GAS * shift)
        return theta

    def to_original_jax(self, names: Sequence[str], theta_reparam: Any) -> Any:
        """JAX-traceable inverse coordinate transform."""

        import jax.numpy as jnp

        theta = jnp.asarray(theta_reparam, dtype=jnp.float64)
        for i, j, shift, ea_default in self._active_pairs(names):
            ea = theta[j] if j is not None else ea_default
            theta = theta.at[i].set(theta[i] * jnp.exp(ea / R_GAS * shift))
        return theta

    # -- spec / prior transforms ------------------------------------------

    def reparameterize(
        self,
        original_specs: Sequence[ParameterSpec],
        original_theta: NDArray[np.floating],
    ) -> tuple[list[ParameterSpec], FloatArray]:
        """Specs and theta in (k_c, Ea) coordinates.

        Parameter names are preserved (the base coefficient now means
        "rate at T_c"); bounds and initial values are scaled by the
        Arrhenius factor evaluated at the pair's reference Ea, which is
        exact for the initial value and a mild rigid shift of the
        (log-space) bounds.
        """

        specs = list(original_specs)
        names = [spec.name for spec in specs]
        theta = self.from_original(names, original_theta)
        new_specs: list[ParameterSpec] = []
        factors = self._factors(specs)
        for spec, value in zip(specs, theta, strict=True):
            factor = factors.get(spec.name, 1.0)
            if factor == 1.0:
                new_specs.append(spec)
                continue
            lower, upper = spec.bounds
            new_specs.append(
                ParameterSpec(
                    spec.name,
                    float(np.clip(spec.initial_value * factor, lower * factor, upper * factor)),
                    (lower * factor, upper * factor),
                    spec.unit,
                    spec.log_transform,
                )
            )
        return new_specs, theta

    def transform_prior_set(
        self, prior_set: PriorSet, specs: Sequence[ParameterSpec]
    ) -> PriorSet:
        """Move prior means to (k_c, Ea) coordinates; widths unchanged.

        Log-space widths (decades) are invariant under the rigid shift;
        the Ea prior itself is untouched.
        """

        factors = self._factors(specs)
        priors = []
        for prior in prior_set.priors:
            factor = factors.get(prior.name, 1.0)
            if factor == 1.0:
                priors.append(prior)
            else:
                priors.append(
                    GaussianParameterPrior(
                        name=prior.name,
                        mean_physical=prior.mean_physical * factor,
                        sigma_search=prior.sigma_search,
                        source=prior.source + "+recentred",
                    )
                )
        return PriorSet(priors=tuple(priors))

    def _factors(self, specs: Sequence[ParameterSpec]) -> dict[str, float]:
        """Arrhenius scale factor per base name, at the reference Ea."""

        names = [spec.name for spec in specs]
        factors: dict[str, float] = {}
        for base, ea in self.pairs:
            t_c = float(self.reference_temperatures.get(base, T_REF))
            if base not in names or np.isclose(t_c, T_REF):
                continue
            shift = 1.0 / t_c - 1.0 / T_REF
            factors[base] = float(np.exp(-_DEFAULT_EA[ea] / R_GAS * shift))
        return factors

    # -- model wrapper -----------------------------------------------------

    def wrap_model(self, original_model: Any) -> "ReparameterizedSEISModel":
        return ReparameterizedSEISModel(
            original_model=original_model, reparameterizer=self
        )

    def coordinate_descriptions(
        self, names: Sequence[str]
    ) -> dict[str, str]:
        """Human-readable meaning of every transformed coordinate."""

        described = {}
        for base, ea in self.pairs:
            t_c = float(self.reference_temperatures.get(base, T_REF))
            if base in names and ea in names and not np.isclose(t_c, T_REF):
                described[base] = f"rate at T_c = {t_c:.2f} K (was: rate at 298.15 K)"
        return described


@dataclass(frozen=True)
class ReparameterizedSEISModel:
    """ForwardModel wrapper: accepts (k_c, Ea), simulates with (k_0, Ea)."""

    original_model: Any
    reparameterizer: ArrheniusReparameterizer

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.original_model.parameter_names)

    @property
    def conditions(self):  # pass-through for StackedSEISModel introspection
        return getattr(self.original_model, "conditions", None)

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        theta_original = self.reparameterizer.to_original(
            self.parameter_names, theta
        )
        return self.original_model.simulate(freq_hz, theta_original)

    def simulate_jax(self, freq_hz: Any, theta: Any) -> Any:
        simulator = getattr(self.original_model, "simulate_jax", None)
        if simulator is None:
            raise AttributeError("original model does not expose a JAX simulation path")
        theta_original = self.reparameterizer.to_original_jax(
            self.parameter_names, theta
        )
        return simulator(freq_hz, theta_original)

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return None
