# learning AI at www.haotianblog.com
"""Python SEIS/DFN-like forward model based on SEIS-Toolbox-LIB equations."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .parameters import ParameterSpec

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
DEFAULT_SELECTED_PARAMETER_NAMES = (
    "Ds_neg_0",
    "rs_neg",
    "k_neg_0",
    "rou_sei_neg_0",
)
SEIS_COMPONENT_CHANNELS = ("cell", "neg", "pos", "sep")
LOW_DIMENSIONAL_SEIS_TEMPERATURES_K = (278.15, 298.15, 318.15)
LOW_DIMENSIONAL_SEIS_SOCS = (0.2, 0.5, 0.8)


def default_seis_theta() -> FloatArray:
    """Return the four upstream negative-electrode parameter defaults."""

    return np.array([1.2e-14, 2.0e-6, 5.031e-11, 1.4025e5], dtype=float)


def recommended_low_dimensional_seis_conditions() -> tuple[tuple[float, float], ...]:
    """Return the default operating grid for four-parameter SEIS recovery.

    Single full-cell spectra are not sufficiently informative for the selected
    negative-electrode SEIS parameters under realistic noise. The 3x3
    temperature/SOC grid keeps the problem low-dimensional while adding the
    orthogonal excitation needed to reduce parameter confounding.
    """

    return tuple(
        (temperature_k, soc)
        for temperature_k in LOW_DIMENSIONAL_SEIS_TEMPERATURES_K
        for soc in LOW_DIMENSIONAL_SEIS_SOCS
    )


def stacked_seis_frequency_grid(
    freq_hz_per_condition: NDArray[np.floating],
    conditions: tuple[tuple[float, float], ...],
) -> FloatArray:
    """Repeat one frequency grid once per operating condition."""

    frequencies = np.asarray(freq_hz_per_condition, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("freq_hz_per_condition must be a non-empty 1D array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("freq_hz_per_condition values must be finite and positive")
    if not conditions:
        raise ValueError("conditions cannot be empty")
    return np.tile(frequencies, len(conditions))


def stacked_seis_condition_context(
    freq_hz_per_condition: NDArray[np.floating],
    conditions: tuple[tuple[float, float], ...],
) -> dict[str, FloatArray]:
    """Build per-point condition columns for a stacked SEIS dataset."""

    frequencies = np.asarray(freq_hz_per_condition, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("freq_hz_per_condition must be a non-empty 1D array")
    if not conditions:
        raise ValueError("conditions cannot be empty")
    for temperature_k, soc in conditions:
        if not np.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("temperature_k must be finite and positive")
        if not np.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("soc must be between zero and one")
    return {
        "temperature_K": np.repeat(
            [temperature_k for temperature_k, _ in conditions], frequencies.size
        ),
        "SOC": np.repeat([soc for _, soc in conditions], frequencies.size),
    }


def default_seis_parameter_specs() -> list[ParameterSpec]:
    """Return the four physical parameter specifications selected for fitting."""

    return [
        ParameterSpec("Ds_neg_0", 1.2e-14, (1e-15, 1e-13), "m^2/s", True),
        ParameterSpec("rs_neg", 2.0e-6, (0.5e-6, 10e-6), "m", True),
        ParameterSpec(
            "k_neg_0",
            5.031e-11,
            (1e-12, 1e-9),
            "m^(1+3*alpha_a_neg)*mol^(-alpha_a_neg)/s",
            True,
        ),
        ParameterSpec("rou_sei_neg_0", 1.4025e5, (1e4, 1e6), "ohm*m", True),
    ]


def all_seis_parameter_specs() -> list[ParameterSpec]:
    """Return every independent scalar physical input in the SEIS calculation."""
    global _ALL_SEIS_SPECS
    if _ALL_SEIS_SPECS is not None:
        return list(_ALL_SEIS_SPECS)
    defaults = _base_seis_parameters()
    specs = [
        ParameterSpec("alpha_a_neg", defaults["alpha_a_neg"], (0.2, 0.8), "-", False),
        ParameterSpec("alpha_a_pos", defaults["alpha_a_pos"], (0.2, 0.8), "-", False),
        ParameterSpec("sigma_neg", defaults["sigma_neg"], (0.1, 100.0), "S/m", True),
        ParameterSpec("sigma_pos", defaults["sigma_pos"], (0.1, 100.0), "S/m", True),
        ParameterSpec("epse_neg", defaults["epse_neg"], (0.2, 0.7), "-", False),
        ParameterSpec("epse_pos", defaults["epse_pos"], (0.2, 0.7), "-", False),
        ParameterSpec("epse_sep", defaults["epse_sep"], (0.3, 0.9), "-", False),
        ParameterSpec("epsf_neg", defaults["epsf_neg"], (0.005, 0.1), "-", False),
        ParameterSpec("epsf_pos", defaults["epsf_pos"], (0.005, 0.1), "-", False),
        ParameterSpec("brug_neg", defaults["brug_neg"], (1.0, 4.5), "-", False),
        ParameterSpec("brug_pos", defaults["brug_pos"], (1.0, 4.5), "-", False),
        ParameterSpec("brug_sep", defaults["brug_sep"], (1.0, 4.5), "-", False),
        ParameterSpec("Ds_neg_0", defaults["Ds_neg_0"], (1e-16, 1e-12), "m^2/s", True),
        ParameterSpec("Ds_pos_0", defaults["Ds_pos_0"], (1e-16, 1e-12), "m^2/s", True),
        ParameterSpec("rs_neg", defaults["rs_neg"], (0.2e-6, 20e-6), "m", True),
        ParameterSpec("rs_pos", defaults["rs_pos"], (0.2e-6, 20e-6), "m", True),
        ParameterSpec("L_neg", defaults["L_neg"], (20e-6, 200e-6), "m", True),
        ParameterSpec("L_pos", defaults["L_pos"], (20e-6, 200e-6), "m", True),
        ParameterSpec("L_sep", defaults["L_sep"], (5e-6, 80e-6), "m", True),
        ParameterSpec(
            "k_neg_0",
            defaults["k_neg_0"],
            (1e-13, 1e-8),
            "m^(1+3*alpha_a_neg)*mol^(-alpha_a_neg)/s",
            True,
        ),
        ParameterSpec(
            "k_pos_0",
            defaults["k_pos_0"],
            (1e-13, 1e-8),
            "m^(1+3*alpha_a_pos)*mol^(-alpha_a_pos)/s",
            True,
        ),
        ParameterSpec(
            "Cdl_neg",
            defaults["Cdl_neg"],
            (1e-3, 10.0),
            "S*s^alpha_dl_neg/m^2",
            True,
        ),
        ParameterSpec(
            "Cdl_pos",
            defaults["Cdl_pos"],
            (1e-3, 10.0),
            "S*s^alpha_dl_pos/m^2",
            True,
        ),
        ParameterSpec("alpha_dl_neg", defaults["alpha_dl_neg"], (0.5, 1.0), "-", False),
        ParameterSpec("alpha_dl_pos", defaults["alpha_dl_pos"], (0.5, 1.0), "-", False),
        ParameterSpec(
            "rou_sei_neg_0", defaults["rou_sei_neg_0"], (1e3, 1e7), "ohm*m", True
        ),
        ParameterSpec(
            "epse_sei_neg", defaults["epse_sei_neg"], (1e-12, 1e-8), "F/m", True
        ),
        ParameterSpec(
            "rou_sei_pos_0", defaults["rou_sei_pos_0"], (1e2, 1e6), "ohm*m", True
        ),
        ParameterSpec(
            "epse_sei_pos", defaults["epse_sei_pos"], (1e-12, 1e-8), "F/m", True
        ),
        ParameterSpec("kappa_0", defaults["kappa_0"], (0.05, 10.0), "S/m", True),
        ParameterSpec("dlnf_ce", defaults["dlnf_ce"], (0.05, 5.0), "-", True),
        ParameterSpec("t_plus", defaults["t_plus"], (0.1, 0.7), "-", False),
        ParameterSpec("ce_0", defaults["ce_0"], (100.0, 3000.0), "mol/m^3", True),
        ParameterSpec("s0_neg", defaults["s0_neg"], (0.001, 0.15), "-", False),
        ParameterSpec("s100_neg", defaults["s100_neg"], (0.65, 0.98), "-", False),
        ParameterSpec("s0_pos", defaults["s0_pos"], (0.80, 0.999), "-", False),
        ParameterSpec("s100_pos", defaults["s100_pos"], (0.25, 0.75), "-", False),
        ParameterSpec(
            "cs_max_neg", defaults["cs_max_neg"], (5e3, 1e5), "mol/m^3", True
        ),
        ParameterSpec(
            "cs_max_pos", defaults["cs_max_pos"], (5e3, 1e5), "mol/m^3", True
        ),
        ParameterSpec("Ea_Ds_neg", defaults["Ea_Ds_neg"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_Ds_pos", defaults["Ea_Ds_pos"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_k_neg", defaults["Ea_k_neg"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_k_pos", defaults["Ea_k_pos"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_De", defaults["Ea_De"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_kappa", defaults["Ea_kappa"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("Ea_rou_sei", defaults["Ea_rou_sei"], (1e3, 1e5), "J/mol", True),
        ParameterSpec("R_contact", defaults["R_contact"], (1e-4, 0.1), "ohm*m^2", True),
        ParameterSpec("L_ind", defaults["L_ind"], (1e-10, 1e-6), "H*m^2", True),
    ]
    _ALL_SEIS_SPECS = tuple(specs)
    return specs


# Module-level cache — ParameterSpec is frozen so this is safe.
_ALL_SEIS_SPECS: tuple[ParameterSpec, ...] | None = None
_ALL_SEIS_NAMES: frozenset[str] | None = None


def _all_seis_parameter_names() -> frozenset[str]:
    """Return cached set of valid SEIS parameter names."""
    global _ALL_SEIS_NAMES
    if _ALL_SEIS_NAMES is None:
        _ALL_SEIS_NAMES = frozenset(s.name for s in all_seis_parameter_specs())
    return _ALL_SEIS_NAMES



@dataclass(frozen=True)
class SEISModel:
    """Full-cell DFN-like static EIS model with selected fitted parameters.

    The model ports the calculation path used by the upstream MATLAB functions
    ``Parameters_update``, ``Model_particle_calculate`` and
    ``Model_DFN_calculate``. The returned full-cell impedance is area-specific
    impedance in ohm*m^2.
    """

    temperature_k: float = 298.15
    soc: float = 1.0
    parameter_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES

    impedance_unit = "ohm*m^2"

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("temperature_k must be finite and positive")
        if not np.isfinite(self.soc) or not 0 <= self.soc <= 1:
            raise ValueError("soc must be between zero and one")
        known_names = _all_seis_parameter_names()
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            raise ValueError("parameter_names must be unique and non-empty")
        unknown_names = set(self.parameter_names).difference(known_names)
        if unknown_names:
            raise ValueError(f"unknown SEIS parameters: {sorted(unknown_names)}")

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        frequencies = np.asarray(freq_hz, dtype=float)
        parameters = np.asarray(theta, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("freq_hz must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("freq_hz values must be finite and positive")
        if parameters.shape != (len(self.parameter_names),):
            raise ValueError(
                f"theta must contain values for {list(self.parameter_names)}"
            )
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
            raise ValueError("SEIS fitted parameters must be finite and positive")
        p = _updated_parameters(
            dict(zip(self.parameter_names, parameters, strict=True)),
            self.temperature_k,
            self.soc,
        )
        return _dfn_cell_impedance(p, frequencies)


@dataclass(frozen=True)
class SEISComponentModel:
    """Return one toolbox-defined DFN component impedance spectrum."""

    temperature_k: float = 298.15
    soc: float = 1.0
    response_channel: str = "cell"
    parameter_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES

    impedance_unit = "ohm*m^2"

    def __post_init__(self) -> None:
        if self.response_channel not in SEIS_COMPONENT_CHANNELS:
            raise ValueError(
                f"response_channel must be one of {list(SEIS_COMPONENT_CHANNELS)}"
            )
        if not np.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("temperature_k must be finite and positive")
        if not np.isfinite(self.soc) or not 0 <= self.soc <= 1:
            raise ValueError("soc must be between zero and one")
        known_names = _all_seis_parameter_names()
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            raise ValueError("parameter_names must be unique and non-empty")
        unknown_names = set(self.parameter_names).difference(known_names)
        if unknown_names:
            raise ValueError(f"unknown SEIS parameters: {sorted(unknown_names)}")

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        return self.simulate_components(freq_hz, theta)[self.response_channel]

    def simulate_components(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> dict[str, ComplexArray]:
        """Return all component spectra computed in the same DFN solve."""

        frequencies = np.asarray(freq_hz, dtype=float)
        parameters = np.asarray(theta, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("freq_hz must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("freq_hz values must be finite and positive")
        if parameters.shape != (len(self.parameter_names),):
            raise ValueError(
                f"theta must contain values for {list(self.parameter_names)}"
            )
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
            raise ValueError("SEIS fitted parameters must be finite and positive")
        p = _updated_parameters(
            dict(zip(self.parameter_names, parameters, strict=True)),
            self.temperature_k,
            self.soc,
        )
        return _dfn_component_impedances(p, frequencies)


@dataclass(frozen=True)
class StackedSEISModel:
    """Apply one shared SEIS parameter vector to several T/SOC spectra."""

    conditions: tuple[tuple[float, float], ...]
    parameter_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES
    compute_backend: str = "numpy"

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("conditions cannot be empty")
        if self.compute_backend not in {"numpy", "jax", "jax-cpu", "jax-gpu"}:
            raise ValueError(
                "compute_backend must be one of 'numpy', 'jax', 'jax-cpu', "
                "or 'jax-gpu'"
            )
        known_names = _all_seis_parameter_names()
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            raise ValueError("parameter_names must be unique and non-empty")
        unknown_names = set(self.parameter_names).difference(known_names)
        if unknown_names:
            raise ValueError(f"unknown SEIS parameters: {sorted(unknown_names)}")
        for temperature_k, soc in self.conditions:
            if not np.isfinite(temperature_k) or temperature_k <= 0:
                raise ValueError("temperature_k must be finite and positive")
            if not np.isfinite(soc) or not 0 <= soc <= 1:
                raise ValueError("soc must be between zero and one")

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        frequencies = np.asarray(freq_hz, dtype=float)
        if frequencies.ndim != 1 or frequencies.size % len(self.conditions) != 0:
            raise ValueError("freq_hz must contain one equal-size block per condition")
        parameters = np.asarray(theta, dtype=float)
        if parameters.shape != (len(self.parameter_names),):
            raise ValueError(
                f"theta must contain values for {list(self.parameter_names)}"
            )
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
            raise ValueError("SEIS fitted parameters must be finite and positive")
        overrides = dict(zip(self.parameter_names, parameters, strict=True))
        blocks = np.split(frequencies, len(self.conditions))
        return np.concatenate(
            [
                _dfn_cell_impedance(
                    _updated_parameters(overrides, temperature_k, soc),
                    block,
                )
                for block, (temperature_k, soc) in zip(
                    blocks, self.conditions, strict=True
                )
            ]
        )

    def simulate_jax(self, freq_hz: Any, theta: Any) -> Any:
        """Return the stacked spectrum as a JAX array.

        Validation remains in :meth:`simulate`; this trace-friendly entry point
        is used by JAX JIT/autodiff inside the optimizer.
        """

        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError as exc:  # pragma: no cover - optional backend
            raise ModuleNotFoundError(
                "JAX backend requested; install the optional 'gpu' dependencies"
            ) from exc

        jax.config.update("jax_enable_x64", True)
        frequencies = jnp.asarray(freq_hz, dtype=jnp.float64).reshape(
            (len(self.conditions), -1)
        )
        parameters = jnp.asarray(theta, dtype=jnp.float64)
        conditions = jnp.asarray(self.conditions, dtype=jnp.float64)
        names = self.parameter_names

        def one_condition(block: Any, condition: Any) -> Any:
            overrides = dict(zip(names, parameters, strict=True))
            updated = _updated_parameters(
                overrides,
                condition[0],
                condition[1],
                array_namespace=jnp,
            )
            return _dfn_cell_impedance(
                updated, block, array_namespace=jnp, validate=False
            )

        return jax.vmap(one_condition)(frequencies, conditions).reshape(-1)


@dataclass(frozen=True)
class DecoupledStackedSEISModel:
    """Stack toolbox-defined regional impedance channels over operating points."""

    conditions: tuple[tuple[float, float], ...]
    response_channels: tuple[str, ...] = SEIS_COMPONENT_CHANNELS
    parameter_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("conditions cannot be empty")
        if not self.response_channels or len(set(self.response_channels)) != len(
            self.response_channels
        ):
            raise ValueError("response_channels must be unique and non-empty")
        for channel in self.response_channels:
            if channel not in SEIS_COMPONENT_CHANNELS:
                raise ValueError(
                    f"response_channels must be drawn from {list(SEIS_COMPONENT_CHANNELS)}"
                )
        known_names = _all_seis_parameter_names()
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            raise ValueError("parameter_names must be unique and non-empty")
        unknown_names = set(self.parameter_names).difference(known_names)
        if unknown_names:
            raise ValueError(f"unknown SEIS parameters: {sorted(unknown_names)}")
        for temperature_k, soc in self.conditions:
            if not np.isfinite(temperature_k) or temperature_k <= 0:
                raise ValueError("temperature_k must be finite and positive")
            if not np.isfinite(soc) or not 0 <= soc <= 1:
                raise ValueError("soc must be between zero and one")

    @property
    def observation_block_count(self) -> int:
        return len(self.conditions) * len(self.response_channels)

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        frequencies = np.asarray(freq_hz, dtype=float)
        if (
            frequencies.ndim != 1
            or frequencies.size % self.observation_block_count != 0
        ):
            raise ValueError(
                "freq_hz must contain one equal-size block per condition/channel"
            )
        parameters = np.asarray(theta, dtype=float)
        if parameters.shape != (len(self.parameter_names),):
            raise ValueError(
                f"theta must contain values for {list(self.parameter_names)}"
            )
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
            raise ValueError("SEIS fitted parameters must be finite and positive")
        overrides = dict(zip(self.parameter_names, parameters, strict=True))
        blocks = np.split(frequencies, self.observation_block_count)
        block_index = 0
        predictions = []
        for temperature_k, soc in self.conditions:
            condition_blocks = blocks[
                block_index : block_index + len(self.response_channels)
            ]
            p = _updated_parameters(overrides, temperature_k, soc)
            if all(
                np.array_equal(condition_blocks[0], block)
                for block in condition_blocks[1:]
            ):
                components = _dfn_component_impedances(p, condition_blocks[0])
                predictions.extend(
                    components[channel] for channel in self.response_channels
                )
            else:
                for channel, block in zip(
                    self.response_channels, condition_blocks, strict=True
                ):
                    full_components = _dfn_component_impedances(p, block)
                    predictions.append(full_components[channel])
            block_index += len(self.response_channels)
        return np.concatenate(predictions)


def _base_seis_parameters() -> dict[str, float]:
    p = {
        "alpha_a_neg": 0.5,
        "alpha_a_pos": 0.5,
        "sigma_neg": 10.0,
        "sigma_pos": 3.8,
        "epse_neg": 0.485,
        "epse_pos": 0.385,
        "epse_sep": 0.724,
        "epsf_neg": 0.0326,
        "epsf_pos": 0.0250,
        "brug_neg": 1.5,
        "brug_pos": 1.5,
        "brug_sep": 1.5,
        "Ds_neg_0": 1.2e-14,
        "Ds_pos_0": 1.0e-14,
        "rs_neg": 2.0e-6,
        "rs_pos": 2.0e-6,
        "L_neg": 88e-6,
        "L_pos": 80e-6,
        "L_sep": 25e-6,
        "k_neg_0": 5.031e-11,
        "k_pos_0": 2.334e-11,
        "Cdl_neg": 0.1,
        "Cdl_pos": 0.1,
        "alpha_dl_neg": 0.9,
        "alpha_dl_pos": 0.9,
        "delta_sei_neg": 2.0e-6 / 50.0,
        "rou_sei_neg_0": 1.4025e5,
        "epse_sei_neg": 3.9216e-10,
        "rou_sei_pos_0": 1.4025e4,
        "epse_sei_pos": 3.9216e-10,
        "kappa_0": 1.2049,
        "dlnf_ce": 1.1319,
        "t_plus": 0.38,
        "ce_0": 1000.0,
        "F": 96487.0,
        "R": 8.314,
        "T_0": 298.15,
        "s0_neg": 0.01429,
        "s100_neg": 0.85510,
        "s0_pos": 0.99174,
        "s100_pos": 0.49950,
        "cs_max_neg": 30555.0,
        "cs_max_pos": 51554.0,
        "Ea_Ds_neg": 42.77e3,
        "Ea_Ds_pos": 18.55e3,
        "Ea_k_neg": 39.57e3,
        "Ea_k_pos": 37.48e3,
        "Ea_De": 37.04e3,
        "Ea_kappa": 34.70e3,
        "Ea_rou_sei": 33.26e3,
        "R_contact": 0.01,
        "L_ind": 1e-8,
    }
    p["De_0"] = (
        2.0
        * p["R"]
        * p["T_0"]
        / p["F"] ** 2
        / p["ce_0"]
        * (1.0 + p["dlnf_ce"])
        * p["t_plus"]
        * (1.0 - p["t_plus"])
        * p["kappa_0"]
    )
    return p


def _updated_parameters(
    parameter_overrides: Mapping[str, float],
    temperature_k: float,
    soc: float,
    *,
    array_namespace: Any = np,
) -> dict[str, float]:
    xp = array_namespace
    p = _base_seis_parameters()
    if xp is np:
        p.update({name: float(value) for name, value in parameter_overrides.items()})
    else:
        p.update(parameter_overrides)

    # Enforce physical dependencies and literature constants
    p["delta_sei_neg"] = p["rs_neg"] / 50.0
    p["delta_sei_pos"] = p["rs_pos"] / 50.0
    p["De_0"] = (
        2.0
        * p["R"]
        * p["T_0"]
        / p["F"] ** 2
        / p["ce_0"]
        * (1.0 + p["dlnf_ce"])
        * p["t_plus"]
        * (1.0 - p["t_plus"])
        * p["kappa_0"]
    )

    p["alpha_c_neg"] = 1.0 - p["alpha_a_neg"]
    p["alpha_c_pos"] = 1.0 - p["alpha_a_pos"]
    p["T"] = temperature_k
    p["SOC"] = soc

    def transport_arrhenius(activation: float | np.ndarray) -> np.ndarray:
        return xp.exp(-activation / p["R"] * (1.0 / p["T"] - 1.0 / p["T_0"]))

    def resistivity_arrhenius(activation: float | np.ndarray) -> np.ndarray:
        return xp.exp(activation / p["R"] * (1.0 / p["T"] - 1.0 / p["T_0"]))

    p["Ds_neg"] = p["Ds_neg_0"] * transport_arrhenius(p["Ea_Ds_neg"])
    p["Ds_pos"] = p["Ds_pos_0"] * transport_arrhenius(p["Ea_Ds_pos"])
    p["k_neg"] = p["k_neg_0"] * transport_arrhenius(p["Ea_k_neg"])
    p["k_pos"] = p["k_pos_0"] * transport_arrhenius(p["Ea_k_pos"])
    p["De"] = p["De_0"] * transport_arrhenius(p["Ea_De"])
    p["kappa"] = p["kappa_0"] * transport_arrhenius(p["Ea_kappa"])
    p["rou_sei_neg"] = p["rou_sei_neg_0"] * resistivity_arrhenius(
        p["Ea_rou_sei"]
    )
    p["rou_sei_pos"] = p["rou_sei_pos_0"] * resistivity_arrhenius(
        p["Ea_rou_sei"]
    )
    p["Rsei_neg"] = (
        p["rou_sei_neg"]
        * (p["rs_neg"] * p["delta_sei_neg"])
        / (p["rs_neg"] + p["delta_sei_neg"])
    )
    p["Rsei_pos"] = (
        p["rou_sei_pos"]
        * (p["rs_pos"] * p["delta_sei_pos"])
        / (p["rs_pos"] + p["delta_sei_pos"])
    )
    p["Csei_neg"] = (
        p["epse_sei_neg"]
        * (p["rs_neg"] + p["delta_sei_neg"])
        / (p["rs_neg"] * p["delta_sei_neg"])
    )
    p["Csei_pos"] = (
        p["epse_sei_pos"]
        * (p["rs_pos"] + p["delta_sei_pos"])
        / (p["rs_pos"] * p["delta_sei_pos"])
    )

    stoich_neg = p["s0_neg"] + soc * (p["s100_neg"] - p["s0_neg"])
    stoich_pos = p["s0_pos"] + soc * (p["s100_pos"] - p["s0_pos"])
    p["dUdc_neg"] = (
        _ocp_derivative(
            _u_neg, _dudt_neg, stoich_neg, p["T"] - p["T_0"],
            array_namespace=xp,
        )
        / p["cs_max_neg"]
    )
    p["dUdc_pos"] = (
        _ocp_derivative(
            _u_pos, _dudt_pos, stoich_pos, p["T"] - p["T_0"],
            array_namespace=xp,
        )
        / p["cs_max_pos"]
    )
    p["cs0_neg"] = p["cs_max_neg"] * stoich_neg
    p["cs0_pos"] = p["cs_max_pos"] * stoich_pos
    p["i0_neg"] = (
        p["F"]
        * p["k_neg"]
        * p["ce_0"] ** p["alpha_a_neg"]
        * (p["cs_max_neg"] - p["cs0_neg"]) ** p["alpha_a_neg"]
        * p["cs0_neg"] ** p["alpha_c_neg"]
    )
    p["i0_pos"] = (
        p["F"]
        * p["k_pos"]
        * p["ce_0"] ** p["alpha_a_pos"]
        * (p["cs_max_pos"] - p["cs0_pos"]) ** p["alpha_a_pos"]
        * p["cs0_pos"] ** p["alpha_c_pos"]
    )
    p["epss_neg"] = 1.0 - p["epse_neg"] - p["epsf_neg"]
    p["epss_pos"] = 1.0 - p["epse_pos"] - p["epsf_pos"]
    p["as_neg"] = 3.0 * p["epss_neg"] / p["rs_neg"]
    p["as_pos"] = 3.0 * p["epss_pos"] / p["rs_pos"]
    for region in ("neg", "pos", "sep"):
        p[f"De_eff_{region}"] = p["De"] * p[f"epse_{region}"] ** p[f"brug_{region}"]
        p[f"kappa_eff_{region}"] = (
            p["kappa"] * p[f"epse_{region}"] ** p[f"brug_{region}"]
        )
        p[f"kappa_D_eff_{region}"] = (
            2.0
            * p["R"]
            * p["T"]
            / p["F"]
            * p[f"kappa_eff_{region}"]
            * (p["t_plus"] - 1.0)
            * (1.0 + p["dlnf_ce"])
        )
    p["sigma_eff_neg"] = p["sigma_neg"] * p["epss_neg"]
    p["sigma_eff_pos"] = p["sigma_pos"] * p["epss_pos"]
    p["Rct_neg"] = p["R"] * p["T"] / (p["i0_neg"] * p["F"])
    p["Rct_pos"] = p["R"] * p["T"] / (p["i0_pos"] * p["F"])
    p["Rdiff_neg"] = -p["rs_neg"] / (p["Ds_neg"] * p["F"]) * p["dUdc_neg"]
    p["Rdiff_pos"] = -p["rs_pos"] / (p["Ds_pos"] * p["F"]) * p["dUdc_pos"]
    return p


def _ocp_derivative(
    base: Any,
    thermal: Any,
    stoich: float,
    delta_t: float,
    *,
    array_namespace: Any = np,
) -> float:
    xp = array_namespace
    step = 1e-30
    value = base(stoich + 1j * step, array_namespace=xp) + delta_t * thermal(
        stoich + 1j * step, array_namespace=xp
    )
    derivative = xp.imag(value) / step
    return float(derivative) if xp is np else derivative


def _u_neg(theta: Any, *, array_namespace: Any = np) -> Any:
    xp = array_namespace
    return (
        0.7222
        + 0.1387 * theta
        + 0.029 * theta**0.5
        - 0.0172 / theta
        + 0.0019 / theta**1.5
        + 0.2808 * xp.exp(0.9 - 15.0 * theta)
        - 0.7984 * xp.exp(0.4465 * theta - 0.4108)
    )


def _u_pos(theta: Any, *, array_namespace: Any = np) -> Any:
    return (
        -4.656
        + 88.669 * theta**2
        - 401.119 * theta**4
        + 342.909 * theta**6
        - 462.471 * theta**8
        + 433.434 * theta**10
    ) / (
        -1.0
        + 18.933 * theta**2
        - 79.532 * theta**4
        + 37.311 * theta**6
        - 73.083 * theta**8
        + 95.96 * theta**10
    )


def _dudt_neg(theta: Any, *, array_namespace: Any = np) -> Any:
    return (
        0.001
        * (
            0.005269056
            + 3.299265709 * theta
            - 91.79325798 * theta**2
            + 1004.911008 * theta**3
            - 5812.278127 * theta**4
            + 19329.7549 * theta**5
            - 37147.8947 * theta**6
            + 38379.18127 * theta**7
            - 16515.05308 * theta**8
        )
        / (
            1.0
            - 48.09287227 * theta
            + 1017.234804 * theta**2
            - 10481.80419 * theta**3
            + 59431.3 * theta**4
            - 195881.6488 * theta**5
            + 374577.3152 * theta**6
            - 385821.1607 * theta**7
            + 165705.8597 * theta**8
        )
    )


def _dudt_pos(theta: Any, *, array_namespace: Any = np) -> Any:
    return (
        -0.001
        * (
            0.199521039
            - 0.928373822 * theta
            + 1.364550689000003 * theta**2
            - 0.6115448939999998 * theta**3
        )
        / (
            1.0
            - 5.661479886999997 * theta
            + 11.47636191 * theta**2
            - 9.82431213599998 * theta**3
            + 3.048755063 * theta**4
        )
    )


def _csch(values: ComplexArray, *, array_namespace: Any = np) -> ComplexArray:
    xp = array_namespace
    if xp is not np:
        values = xp.asarray(values, dtype=complex)
        non_negative = values.real >= 0
        positive_branch = 2.0 * xp.exp(-values) / (1.0 - xp.exp(-2.0 * values))
        negative_branch = -2.0 * xp.exp(values) / (1.0 - xp.exp(2.0 * values))
        return xp.where(non_negative, positive_branch, negative_branch)
    values = np.asarray(values, dtype=complex)
    result = np.empty_like(values)
    non_negative = values.real >= 0
    positive_values = values[non_negative]
    negative_values = values[~non_negative]
    result[non_negative] = (
        2.0 * np.exp(-positive_values) / (1.0 - np.exp(-2.0 * positive_values))
    )
    result[~non_negative] = (
        -2.0 * np.exp(negative_values) / (1.0 - np.exp(2.0 * negative_values))
    )
    return result


def _dfn_component_impedances(
    p: dict[str, float],
    freq_hz: FloatArray,
    include_regional_components: bool = True,
    *,
    array_namespace: Any = np,
    validate: bool = True,
) -> dict[str, ComplexArray]:
    xp = array_namespace
    s = 1j * 2.0 * xp.pi * freq_hz
    error_context = (
        np.errstate(over="ignore", divide="ignore", invalid="ignore")
        if xp is np
        else nullcontext()
    )
    with error_context:
        root_particle_neg = xp.sqrt(s * p["rs_neg"] ** 2 / p["Ds_neg"])
        root_particle_pos = xp.sqrt(s * p["rs_pos"] ** 2 / p["Ds_pos"])
        ys_neg = (root_particle_neg - xp.tanh(root_particle_neg)) / xp.tanh(
            root_particle_neg
        )
        ys_pos = (root_particle_pos - xp.tanh(root_particle_pos)) / xp.tanh(
            root_particle_pos
        )
        zd_neg = p["Rdiff_neg"] / ys_neg
        zd_pos = p["Rdiff_pos"] / ys_pos
        omega = 2.0 * xp.pi * freq_hz
        y_dl_neg = p["Cdl_neg"] * (1j * omega) ** p["alpha_dl_neg"]
        y_dl_pos = p["Cdl_pos"] * (1j * omega) ** p["alpha_dl_pos"]
        zf_neg = 1.0 / (y_dl_neg + 1.0 / (p["Rct_neg"] + zd_neg))
        zf_pos = 1.0 / (y_dl_pos + 1.0 / (p["Rct_pos"] + zd_pos))
        zint_neg = 1.0 / (s * p["Csei_neg"] + 1.0 / (p["Rsei_neg"] + zf_neg))
        zint_pos = 1.0 / (s * p["Csei_pos"] + 1.0 / (p["Rsei_pos"] + zf_pos))

        s_neg = p["epse_neg"] * p["L_neg"] ** 2 / p["De_eff_neg"] * s
        s_pos = p["epse_pos"] * p["L_pos"] ** 2 / p["De_eff_pos"] * s
        s_sep = p["epse_sep"] * p["L_sep"] ** 2 / p["De_eff_sep"] * s
        pi_i_neg = (
            -p["kappa_D_eff_neg"]
            / p["kappa_eff_neg"]
            / p["ce_0"]
            * (1.0 - p["t_plus"])
            * p["as_neg"]
            * p["L_neg"] ** 2
            / p["F"]
            / p["De_eff_neg"]
            / zint_neg
        )
        pi_i_pos = (
            -p["kappa_D_eff_pos"]
            / p["kappa_eff_pos"]
            / p["ce_0"]
            * (1.0 - p["t_plus"])
            * p["as_pos"]
            * p["L_pos"] ** 2
            / p["F"]
            / p["De_eff_pos"]
            / zint_pos
        )
        pi_ii_neg = (
            p["as_neg"]
            * p["L_neg"] ** 2
            * (1.0 + p["sigma_eff_neg"] / p["kappa_eff_neg"])
            / p["sigma_eff_neg"]
            / zint_neg
        )
        pi_ii_pos = (
            p["as_pos"]
            * p["L_pos"] ** 2
            * (1.0 + p["sigma_eff_pos"] / p["kappa_eff_pos"])
            / p["sigma_eff_pos"]
            / zint_pos
        )
        pi_iii_sep = -p["kappa_D_eff_sep"] / p["ce_0"] / p["De_eff_sep"] / p["F"]
        disc_neg = xp.sqrt(
            s_neg**2
            + 2.0 * s_neg * (pi_i_neg - pi_ii_neg)
            + (pi_i_neg + pi_ii_neg) ** 2
        )
        disc_pos = xp.sqrt(
            s_pos**2
            + 2.0 * s_pos * (pi_i_pos - pi_ii_pos)
            + (pi_i_pos + pi_ii_pos) ** 2
        )
        lam_i_neg = 0.5 * (s_neg + pi_i_neg + pi_ii_neg + disc_neg)
        lam_ii_neg = 0.5 * (s_neg + pi_i_neg + pi_ii_neg - disc_neg)
        lam_i_pos = 0.5 * (s_pos + pi_i_pos + pi_ii_pos + disc_pos)
        lam_ii_pos = 0.5 * (s_pos + pi_i_pos + pi_ii_pos - disc_pos)
        root_sep = xp.sqrt(s_sep)
        root_i_neg, root_ii_neg = xp.sqrt(lam_i_neg), xp.sqrt(lam_ii_neg)
        root_i_pos, root_ii_pos = xp.sqrt(lam_i_pos), xp.sqrt(lam_ii_pos)
        delta_neg = lam_i_neg - lam_ii_neg
        delta_pos = lam_i_pos - lam_ii_pos
        lambda_iii_sep = (
            p["L_sep"] / p["De_eff_sep"] / root_sep
            * _csch(root_sep, array_namespace=xp)
        )
        lambda_i_neg = (
            -p["L_neg"] ** 3
            * p["as_neg"]
            * (1.0 - p["t_plus"])
            / p["F"]
            / zint_neg
            / p["De_eff_neg"]
            / p["sigma_eff_neg"]
            / delta_neg
            * (
                _csch(root_ii_neg, array_namespace=xp) / root_ii_neg
                - _csch(root_i_neg, array_namespace=xp) / root_i_neg
                + p["sigma_eff_neg"]
                / p["kappa_eff_neg"]
                * (
                    1.0 / root_ii_neg / xp.tanh(root_ii_neg)
                    - 1.0 / root_i_neg / xp.tanh(root_i_neg)
                )
            )
        )
        lambda_ii_neg = p["L_neg"] / p["De_eff_neg"] / delta_neg * (
            (s_neg + pi_i_neg - lam_ii_neg) / root_i_neg / xp.tanh(root_i_neg)
            - (s_neg + pi_i_neg - lam_i_neg) / root_ii_neg / xp.tanh(root_ii_neg)
        ) + p["L_sep"] / p["De_eff_sep"] / root_sep / xp.tanh(root_sep)
        lambda_i_pos = (
            -p["L_pos"] ** 3
            * p["as_pos"]
            * (1.0 - p["t_plus"])
            / p["F"]
            / zint_pos
            / p["De_eff_pos"]
            / p["sigma_eff_pos"]
            / delta_pos
            * (
                _csch(root_ii_pos, array_namespace=xp) / root_ii_pos
                - _csch(root_i_pos, array_namespace=xp) / root_i_pos
                + p["sigma_eff_pos"]
                / p["kappa_eff_pos"]
                * (
                    1.0 / root_ii_pos / xp.tanh(root_ii_pos)
                    - 1.0 / root_i_pos / xp.tanh(root_i_pos)
                )
            )
        )
        lambda_ii_pos = p["L_pos"] / p["De_eff_pos"] / delta_pos * (
            (s_pos + pi_i_pos - lam_ii_pos) / root_i_pos / xp.tanh(root_i_pos)
            - (s_pos + pi_i_pos - lam_i_pos) / root_ii_pos / xp.tanh(root_ii_pos)
        ) + p["L_sep"] / p["De_eff_sep"] / root_sep / xp.tanh(root_sep)
        zeta_neg = (lambda_i_neg + lambda_i_pos / lambda_ii_pos * lambda_iii_sep) / (
            lambda_ii_neg - lambda_iii_sep**2 / lambda_ii_pos
        )
        zeta_pos = (lambda_i_pos + lambda_i_neg / lambda_ii_neg * lambda_iii_sep) / (
            lambda_ii_pos - lambda_iii_sep**2 / lambda_ii_neg
        )
        interface_neg_ii = (
            p["sigma_eff_neg"] / p["kappa_eff_neg"]
            - (s_neg + pi_i_neg - lam_ii_neg)
            * p["sigma_eff_neg"]
            / p["L_neg"] ** 2
            / p["as_neg"]
            / (1.0 - p["t_plus"])
            * p["F"]
            * zint_neg
            * zeta_neg
        )
        interface_neg_i = (
            p["sigma_eff_neg"] / p["kappa_eff_neg"]
            - (s_neg + pi_i_neg - lam_i_neg)
            * p["sigma_eff_neg"]
            / p["L_neg"] ** 2
            / p["as_neg"]
            / (1.0 - p["t_plus"])
            * p["F"]
            * zint_neg
            * zeta_neg
        )
        b_i_neg = (
            pi_i_neg / root_i_neg / delta_neg / xp.tanh(root_i_neg)
            + pi_i_neg / root_i_neg / delta_neg
            * _csch(root_i_neg, array_namespace=xp) * interface_neg_ii
        )
        b_iii_neg = (
            -pi_i_neg / root_ii_neg / delta_neg / xp.tanh(root_ii_neg)
            - pi_i_neg / root_ii_neg / delta_neg
            * _csch(root_ii_neg, array_namespace=xp) * interface_neg_i
        )
        interface_pos_ii = (
            p["sigma_eff_pos"] / p["kappa_eff_pos"]
            - (s_pos + pi_i_pos - lam_ii_pos)
            * p["sigma_eff_pos"]
            / p["L_pos"] ** 2
            / p["as_pos"]
            / (1.0 - p["t_plus"])
            * p["F"]
            * zint_pos
            * zeta_pos
        )
        interface_pos_i = (
            p["sigma_eff_pos"] / p["kappa_eff_pos"]
            - (s_pos + pi_i_pos - lam_i_pos)
            * p["sigma_eff_pos"]
            / p["L_pos"] ** 2
            / p["as_pos"]
            / (1.0 - p["t_plus"])
            * p["F"]
            * zint_pos
            * zeta_pos
        )
        b_ii_pos = pi_i_pos / root_i_pos / delta_pos * interface_pos_ii
        b_i_pos = -pi_i_pos / root_i_pos / delta_pos * _csch(
            root_i_pos, array_namespace=xp
        ) - b_ii_pos / xp.tanh(root_i_pos)
        b_iv_pos = -pi_i_pos / root_ii_pos / delta_pos * interface_pos_i
        b_iii_pos = pi_i_pos / root_ii_pos / delta_pos * _csch(
            root_ii_pos, array_namespace=xp
        ) - b_iv_pos / xp.tanh(root_ii_pos)
        b_i_sep = (
            pi_iii_sep * p["F"] / root_sep
            * _csch(root_sep, array_namespace=xp) * zeta_pos
            - pi_iii_sep * p["F"] / root_sep / xp.tanh(root_sep) * zeta_neg
        )
        chi_i_neg = (
            pi_i_neg / root_i_neg / delta_neg
            * _csch(root_i_neg, array_namespace=xp)
            + pi_i_neg / root_i_neg / delta_neg / xp.tanh(root_i_neg)
            * interface_neg_ii
        )
        chi_ii_neg = (
            -pi_i_neg / root_ii_neg / delta_neg
            * _csch(root_ii_neg, array_namespace=xp)
            - pi_i_neg
            / root_ii_neg
            / delta_neg
            / xp.tanh(root_ii_neg)
            * interface_neg_i
        )
        chi_i_pos = -pi_i_pos / root_i_pos / delta_pos / xp.tanh(
            root_i_pos
        ) - b_ii_pos * _csch(root_i_pos, array_namespace=xp)
        chi_ii_pos = pi_i_pos / root_ii_pos / delta_pos / xp.tanh(
            root_ii_pos
        ) - b_iv_pos * _csch(root_ii_pos, array_namespace=xp)
        chi_iii_sep = (
            pi_iii_sep * p["F"] / root_sep / xp.tanh(root_sep) * zeta_pos
            - pi_iii_sep * p["F"] / root_sep
            * _csch(root_sep, array_namespace=xp) * zeta_neg
        )
        h_i_neg = -p["L_neg"] / (p["sigma_eff_neg"] + p["kappa_eff_neg"])
        h_ii_neg = (
            -p["L_neg"]
            / p["sigma_eff_neg"]
            / pi_i_neg
            * ((s_neg - lam_i_neg) * chi_i_neg + (s_neg - lam_ii_neg) * chi_ii_neg)
            + p["L_neg"] ** 3
            * p["as_neg"]
            / p["sigma_eff_neg"] ** 2
            / zint_neg
            / pi_i_neg
            * (
                (s_neg - lam_i_neg) / lam_i_neg * chi_i_neg
                + (s_neg - lam_ii_neg) / lam_ii_neg * chi_ii_neg
            )
            - h_i_neg
            + p["L_sep"] / p["kappa_eff_sep"] * (1.0 + b_i_sep - chi_iii_sep)
        )
        h_i_pos = -p["L_pos"] / (p["sigma_eff_pos"] + p["kappa_eff_pos"])
        h_ii_pos = -p["L_pos"] / p["sigma_eff_pos"] / pi_i_pos * (
            (s_pos - lam_i_pos) * b_i_pos + (s_pos - lam_ii_pos) * b_iii_pos
        ) + p["L_pos"] ** 3 * p["as_pos"] / p[
            "sigma_eff_pos"
        ] ** 2 / zint_pos / pi_i_pos * (
            (s_pos - lam_i_pos) / lam_i_pos * b_i_pos
            + (s_pos - lam_ii_pos) / lam_ii_pos * b_iii_pos
        )
        z_cell = (
            p["L_pos"] ** 3
            * p["as_pos"]
            / p["sigma_eff_pos"] ** 2
            / zint_pos
            / pi_i_pos
            * (
                (s_pos - lam_i_pos) / lam_i_pos * chi_i_pos
                + (s_pos - lam_ii_pos) / lam_ii_pos * chi_ii_pos
            )
            - (h_i_pos + h_ii_pos)
            - p["L_neg"] ** 3
            * p["as_neg"]
            / p["sigma_eff_neg"] ** 2
            / zint_neg
            / pi_i_neg
            * (
                (s_neg - lam_i_neg) / lam_i_neg * b_i_neg
                + (s_neg - lam_ii_neg) / lam_ii_neg * b_iii_neg
            )
            + h_ii_neg
        )
        if include_regional_components:
            z_neg = (
                p["L_neg"] / (p["sigma_eff_neg"] + p["kappa_eff_neg"])
                - p["L_neg"]
                / p["sigma_eff_neg"]
                / pi_i_neg
                * ((s_neg - lam_i_neg) * chi_i_neg + (s_neg - lam_ii_neg) * chi_ii_neg)
                + p["L_neg"] ** 3
                * p["as_neg"]
                / p["sigma_eff_neg"] ** 2
                / zint_neg
                / pi_i_neg
                * (
                    (s_neg - lam_i_neg) / lam_i_neg * (chi_i_neg - b_i_neg)
                    + (s_neg - lam_ii_neg) / lam_ii_neg * (chi_ii_neg - b_iii_neg)
                )
            )
            z_pos = (
                p["L_pos"] / (p["sigma_eff_pos"] + p["kappa_eff_pos"])
                + p["L_pos"]
                / p["sigma_eff_pos"]
                / pi_i_pos
                * ((s_pos - lam_i_pos) * b_i_pos + (s_pos - lam_ii_pos) * b_iii_pos)
                + p["L_pos"] ** 3
                * p["as_pos"]
                / p["sigma_eff_pos"] ** 2
                / zint_pos
                / pi_i_pos
                * (
                    (s_pos - lam_i_pos) / lam_i_pos * (chi_i_pos - b_i_pos)
                    + (s_pos - lam_ii_pos) / lam_ii_pos * (chi_ii_pos - b_iii_pos)
                )
            )
            z_sep = p["L_sep"] / p["kappa_eff_sep"] + p["L_sep"] / p[
                "kappa_eff_sep"
            ] * pi_iii_sep * p["F"] / root_sep * (
                _csch(root_sep, array_namespace=xp) - 1.0 / xp.tanh(root_sep)
            ) * (
                zeta_neg + zeta_pos
            )
    omega = 2.0 * xp.pi * freq_hz
    z_cell_new = z_cell + p["R_contact"] + 1j * omega * p["L_ind"]
    impedances = {"cell": xp.asarray(z_cell_new, dtype=complex)}
    if include_regional_components:
        z_neg_new = z_neg + 0.5 * p["R_contact"] + 0.5 * 1j * omega * p["L_ind"]
        z_pos_new = z_pos + 0.5 * p["R_contact"] + 0.5 * 1j * omega * p["L_ind"]
        z_sep_new = z_sep
        impedances.update(
            {
                "neg": xp.asarray(z_neg_new, dtype=complex),
                "pos": xp.asarray(z_pos_new, dtype=complex),
                "sep": xp.asarray(z_sep_new, dtype=complex),
            }
        )
    if validate:
        for channel, impedance in impedances.items():
            if not np.all(np.isfinite(impedance.real)) or not np.all(
                np.isfinite(impedance.imag)
            ):
                raise FloatingPointError(
                    f"SEIS {channel} impedance calculation returned non-finite values"
                )
    return impedances


def _dfn_cell_impedance(
    p: dict[str, float],
    freq_hz: FloatArray,
    *,
    array_namespace: Any = np,
    validate: bool = True,
) -> ComplexArray:
    return _dfn_component_impedances(
        p,
        freq_hz,
        include_regional_components=False,
        array_namespace=array_namespace,
        validate=validate,
    )["cell"]
