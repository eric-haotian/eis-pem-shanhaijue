# learning AI at www.haotianblog.com
"""Generation-2 ECM library: higher-order, physically explicit models.

Extends the six Generation-1 models (which remain available unchanged
through the combined registry) with:

* **ZARC-chain models** (``ZARC1`` .. ``ZARC3``) — the Gen-1 RCPE chains
  reparameterized as (R, τ, α) Cole-Cole arcs, removing Q-α collinearity.
* **Finite diffusion models** (``ZARC2_FLW``, ``ZARC2_FSW``,
  ``ZARC3_FLW``) — transmissive / reflective bounded diffusion instead of
  the semi-infinite Warburg, matching battery low-frequency behaviour.
* **Fractional diffusion** (``ZARC2_fFLW``) — anomalous-transport
  exponent γ for disordered electrodes.
* **Gerischer** (``ZARC1_G``) — coupled reaction-diffusion.
* **Transmission-line models** (``TLMB``, ``TLM_CT``, ``ZARC_SEI_TLM``)
  — porous-electrode physics with explicit ionic pore resistance, and an
  SEI film + porous electrode stack that separates film and
  charge-transfer processes.

All models implement the Generation-1 ``ForwardModel`` protocol
(``simulate``, ``guess_from_drt``) plus the ECM conventions
(``name``, ``circuit_string``, ``parameter_specs``, ``n_params``,
``parameter_names``, ``default_theta``) so the entire Gen-1
infrastructure (optimizers, diagnostics, selectors) works unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .ecm_library import all_ecm_models as _gen1_models
from .parameters import ParameterSpec

from .elements import (
    z_gerischer,
    z_inductor,
    z_tlm_blocking,
    z_tlm_faradaic,
    z_warburg_open,
    z_warburg_short,
    z_warburg_short_fractional,
    z_zarc,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Shared parameter-spec factories (consistent bounds across the library)
# ---------------------------------------------------------------------------


def _spec_rs() -> ParameterSpec:
    return ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True)


def _spec_l() -> ParameterSpec:
    return ParameterSpec("L", 1e-7, (1e-10, 1e-4), "H", log_transform=True)


def _spec_r(name: str, initial: float = 0.05) -> ParameterSpec:
    return ParameterSpec(name, initial, (1e-6, 10.0), "ohm", log_transform=True)


def _spec_tau(name: str, initial: float = 1e-2) -> ParameterSpec:
    return ParameterSpec(name, initial, (1e-8, 1e4), "s", log_transform=True)


def _spec_alpha(name: str, initial: float = 0.85) -> ParameterSpec:
    return ParameterSpec(name, initial, (0.5, 1.0), "1", log_transform=False)


def _spec_rd(name: str = "Rd", initial: float = 0.05) -> ParameterSpec:
    return ParameterSpec(name, initial, (1e-6, 100.0), "ohm", log_transform=True)


def _spec_taud(name: str = "tau_d", initial: float = 50.0) -> ParameterSpec:
    return ParameterSpec(name, initial, (1e-3, 1e6), "s", log_transform=True)


def _spec_gamma(name: str = "gamma", initial: float = 0.5) -> ParameterSpec:
    return ParameterSpec(name, initial, (0.3, 0.6), "1", log_transform=False)


def _spec_q(name: str, initial: float = 1.0) -> ParameterSpec:
    return ParameterSpec(name, initial, (1e-6, 1e6), "F*s^(a-1)", log_transform=True)


def _zarc_specs(index: int, tau_initial: float) -> tuple[ParameterSpec, ...]:
    return (
        _spec_r(f"R{index}", 0.05 / index),
        _spec_tau(f"tau{index}", tau_initial),
        _spec_alpha(f"alpha{index}", 0.90 - 0.05 * (index - 1)),
    )


# ---------------------------------------------------------------------------
# Base class with the shared ECM protocol plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ECMBaseV2:
    """Shared protocol plumbing for Generation-2 ECM models."""

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:  # pragma: no cover
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return len(self.parameter_specs)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def _validate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> tuple[FloatArray, dict[str, float]]:
        f = np.asarray(freq_hz, dtype=float)
        t = np.asarray(theta, dtype=float)
        if f.ndim != 1 or f.size == 0:
            raise ValueError("freq_hz must be a non-empty 1-D array")
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            raise ValueError("freq_hz must be finite and positive")
        if t.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},), got {t.shape}")
        if not np.all(np.isfinite(t)):
            raise ValueError("theta must be finite")
        return f, dict(zip(self.parameter_names, t, strict=True))

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return None


# ---------------------------------------------------------------------------
# DRT-guess helpers
# ---------------------------------------------------------------------------


def _sorted_drt_peaks(drt_result: Any) -> list[tuple[float, float]]:
    """Return (resistance, tau) pairs sorted by tau ascending."""

    if drt_result is None or drt_result.n_peaks == 0:
        return []
    idx = np.argsort(drt_result.peak_taus)
    return [
        (
            max(float(drt_result.peak_resistances[i]), 1e-6),
            max(float(drt_result.peak_taus[i]), 1e-8),
        )
        for i in idx
    ]


def _assign_arcs(drt_result: Any, n_arcs: int) -> list[tuple[float, float]] | None:
    """Distribute DRT peaks over ``n_arcs`` ZARC arcs.

    When fewer peaks than arcs are visible, the largest peak is split
    into fast/slow halves so every arc gets a physically plausible
    starting point.  Returns ``None`` when the DRT saw no peaks at all.
    """

    peaks = _sorted_drt_peaks(drt_result)
    if not peaks:
        return None
    while len(peaks) < n_arcs:
        # Split the largest-resistance peak into two staggered halves.
        j = int(np.argmax([r for r, _ in peaks]))
        r, tau = peaks[j]
        peaks[j] = (r * 0.5, tau / 3.0)
        peaks.insert(j + 1, (r * 0.5, tau * 3.0))
        peaks.sort(key=lambda p: p[1])
    if len(peaks) > n_arcs:
        # Keep the n_arcs largest peaks, preserving tau order.
        keep = sorted(
            sorted(range(len(peaks)), key=lambda i: peaks[i][0], reverse=True)[:n_arcs]
        )
        peaks = [peaks[i] for i in keep]
    return peaks


def _zarc_guess(drt_result: Any, n_arcs: int) -> dict[str, float] | None:
    arcs = _assign_arcs(drt_result, n_arcs)
    if arcs is None:
        return None
    guess: dict[str, float] = {"Rs": max(float(drt_result.r_inf), 1e-6)}
    for k, (r, tau) in enumerate(arcs, start=1):
        guess[f"R{k}"] = r
        guess[f"tau{k}"] = tau
        guess[f"alpha{k}"] = 0.90 - 0.05 * (k - 1)
    return guess


def _diffusion_guess(drt_result: Any) -> dict[str, float]:
    """Heuristic FLW/FSW starting values from the DRT tail."""

    peaks = _sorted_drt_peaks(drt_result)
    slowest_tau = peaks[-1][1] if peaks else 1.0
    total_r = max(float(drt_result.total_resistance()), 1e-6)
    return {"Rd": 0.5 * total_r, "tau_d": max(20.0 * slowest_tau, 1e-2)}


# ---------------------------------------------------------------------------
# ZARC-chain models (reparameterized RCPE chains)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zarc1Model(ECMBaseV2):
    """Rs + ZARC — single depressed arc, (R, τ, α) parameterization."""

    name: str = "ZARC1"
    circuit_string: str = "Rs + ZARC1(R1,tau1,alpha1)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (_spec_rs(), *_zarc_specs(1, 1e-2))

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"] + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"]), dtype=complex
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return _zarc_guess(drt_result, 1)


@dataclass(frozen=True)
class Zarc2Model(ECMBaseV2):
    """Rs + 2 ZARCs — SEI + charge transfer, orthogonal parameterization."""

    name: str = "ZARC2"
    circuit_string: str = "Rs + ZARC1 + ZARC2"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (_spec_rs(), *_zarc_specs(1, 1e-3), *_zarc_specs(2, 1e-1))

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return _zarc_guess(drt_result, 2)


@dataclass(frozen=True)
class Zarc3Model(ECMBaseV2):
    """Rs + 3 ZARCs — SEI + charge transfer + solid-state arc."""

    name: str = "ZARC3"
    circuit_string: str = "Rs + ZARC1 + ZARC2 + ZARC3"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            *_zarc_specs(1, 1e-4),
            *_zarc_specs(2, 1e-2),
            *_zarc_specs(3, 1.0),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"])
            + z_zarc(omega, p["R3"], p["tau3"], p["alpha3"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return _zarc_guess(drt_result, 3)


# ---------------------------------------------------------------------------
# Finite / fractional diffusion models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zarc2FLWModel(ECMBaseV2):
    """Rs + L + 2 ZARCs + finite-length Warburg (transmissive boundary)."""

    name: str = "ZARC2_FLW"
    circuit_string: str = "Rs + L + ZARC1 + ZARC2 + FLW(Rd,tau_d)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            *_zarc_specs(1, 1e-3),
            *_zarc_specs(2, 1e-1),
            _spec_rd(),
            _spec_taud(),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"])
            + z_warburg_short(omega, p["Rd"], p["tau_d"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guess = _zarc_guess(drt_result, 2)
        if guess is None:
            return None
        guess.update(_diffusion_guess(drt_result))
        guess["L"] = 1e-7
        return guess


@dataclass(frozen=True)
class Zarc2FSWModel(ECMBaseV2):
    """Rs + L + 2 ZARCs + finite-space Warburg (reflective boundary)."""

    name: str = "ZARC2_FSW"
    circuit_string: str = "Rs + L + ZARC1 + ZARC2 + FSW(Rd,tau_d)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            *_zarc_specs(1, 1e-3),
            *_zarc_specs(2, 1e-1),
            _spec_rd(),
            _spec_taud(),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"])
            + z_warburg_open(omega, p["Rd"], p["tau_d"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guess = _zarc_guess(drt_result, 2)
        if guess is None:
            return None
        guess.update(_diffusion_guess(drt_result))
        guess["L"] = 1e-7
        return guess


@dataclass(frozen=True)
class Zarc3FLWModel(ECMBaseV2):
    """Rs + L + 3 ZARCs + finite-length Warburg — full high-order model."""

    name: str = "ZARC3_FLW"
    circuit_string: str = "Rs + L + ZARC1 + ZARC2 + ZARC3 + FLW(Rd,tau_d)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            *_zarc_specs(1, 1e-4),
            *_zarc_specs(2, 1e-2),
            *_zarc_specs(3, 1.0),
            _spec_rd(),
            _spec_taud(),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"])
            + z_zarc(omega, p["R3"], p["tau3"], p["alpha3"])
            + z_warburg_short(omega, p["Rd"], p["tau_d"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guess = _zarc_guess(drt_result, 3)
        if guess is None:
            return None
        guess.update(_diffusion_guess(drt_result))
        guess["L"] = 1e-7
        return guess


@dataclass(frozen=True)
class Zarc2FractionalFLWModel(ECMBaseV2):
    """Rs + L + 2 ZARCs + fractional finite-length Warburg (exponent γ)."""

    name: str = "ZARC2_fFLW"
    circuit_string: str = "Rs + L + ZARC1 + ZARC2 + fFLW(Rd,tau_d,gamma)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            *_zarc_specs(1, 1e-3),
            *_zarc_specs(2, 1e-1),
            _spec_rd(),
            _spec_taud(),
            _spec_gamma(),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_zarc(omega, p["R2"], p["tau2"], p["alpha2"])
            + z_warburg_short_fractional(omega, p["Rd"], p["tau_d"], p["gamma"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guess = _zarc_guess(drt_result, 2)
        if guess is None:
            return None
        guess.update(_diffusion_guess(drt_result))
        guess["L"] = 1e-7
        guess["gamma"] = 0.5
        return guess


@dataclass(frozen=True)
class Zarc1GerischerModel(ECMBaseV2):
    """Rs + ZARC + Gerischer — coupled reaction-diffusion tail."""

    name: str = "ZARC1_G"
    circuit_string: str = "Rs + ZARC1 + G(Rg,tau_g)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            *_zarc_specs(1, 1e-2),
            _spec_rd("Rg", 0.05),
            _spec_taud("tau_g", 1.0),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_zarc(omega, p["R1"], p["tau1"], p["alpha1"])
            + z_gerischer(omega, p["Rg"], p["tau_g"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        guess = _zarc_guess(drt_result, 1)
        if guess is None:
            return None
        diffusion = _diffusion_guess(drt_result)
        guess["Rg"] = diffusion["Rd"]
        guess["tau_g"] = diffusion["tau_d"] / 20.0
        return guess


# ---------------------------------------------------------------------------
# Transmission-line (porous electrode) models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TLMBlockingModel(ECMBaseV2):
    """Rs + L + blocking-electrode transmission line (no charge transfer)."""

    name: str = "TLMB"
    circuit_string: str = "Rs + L + TLM(R_ion, CPE_dl)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            _spec_r("Rion", 0.02),
            _spec_q("Qdl", 10.0),
            _spec_alpha("alpha", 0.90),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_tlm_blocking(omega, p["Rion"], p["Qdl"], p["alpha"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        peaks = _sorted_drt_peaks(drt_result)
        if not peaks:
            return None
        r_total = max(float(drt_result.total_resistance()), 1e-6)
        tau_main = peaks[-1][1]
        alpha = 0.90
        return {
            "Rs": max(float(drt_result.r_inf), 1e-6),
            "L": 1e-7,
            "Rion": 3.0 * r_total,
            "Qdl": (tau_main**alpha) / r_total,
            "alpha": alpha,
        }


@dataclass(frozen=True)
class TLMChargeTransferModel(ECMBaseV2):
    """Rs + L + faradaic TLM (distributed R_ct ‖ CPE along the pore)."""

    name: str = "TLM_CT"
    circuit_string: str = "Rs + L + TLM(R_ion, Rct||CPE_dl)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            _spec_r("Rion", 0.02),
            _spec_r("Rct", 0.05),
            _spec_q("Qdl", 1.0),
            _spec_alpha("alpha", 0.90),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_tlm_faradaic(omega, p["Rion"], p["Rct"], p["Qdl"], p["alpha"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        peaks = _sorted_drt_peaks(drt_result)
        if not peaks:
            return None
        r_main, tau_main = max(peaks, key=lambda p: p[0])
        alpha = 0.90
        return {
            "Rs": max(float(drt_result.r_inf), 1e-6),
            "L": 1e-7,
            "Rion": max(0.5 * r_main, 1e-6),
            "Rct": r_main,
            "Qdl": (tau_main**alpha) / r_main,
            "alpha": alpha,
        }


@dataclass(frozen=True)
class ZarcSEITLMModel(ECMBaseV2):
    """Rs + L + ZARC(SEI film) + faradaic TLM (porous electrode).

    Explicitly separates the surface-film (SEI/CEI) arc from the
    distributed charge-transfer response of the porous electrode.
    """

    name: str = "ZARC_SEI_TLM"
    circuit_string: str = "Rs + L + ZARC_sei + TLM(R_ion, Rct||CPE_dl)"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            _spec_rs(),
            _spec_l(),
            _spec_r("Rsei", 0.01),
            _spec_tau("tau_sei", 1e-4),
            _spec_alpha("alpha_sei", 0.90),
            _spec_r("Rion", 0.02),
            _spec_r("Rct", 0.05),
            _spec_q("Qdl", 1.0),
            _spec_alpha("alpha", 0.85),
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, p = self._validate(freq_hz, theta)
        omega = 2.0 * np.pi * f
        return np.asarray(
            p["Rs"]
            + z_inductor(omega, p["L"])
            + z_zarc(omega, p["Rsei"], p["tau_sei"], p["alpha_sei"])
            + z_tlm_faradaic(omega, p["Rion"], p["Rct"], p["Qdl"], p["alpha"]),
            dtype=complex,
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        arcs = _assign_arcs(drt_result, 2)
        if arcs is None:
            return None
        (r_fast, tau_fast), (r_slow, tau_slow) = arcs
        alpha = 0.85
        return {
            "Rs": max(float(drt_result.r_inf), 1e-6),
            "L": 1e-7,
            "Rsei": r_fast,
            "tau_sei": tau_fast,
            "alpha_sei": 0.90,
            "Rion": max(0.5 * r_slow, 1e-6),
            "Rct": r_slow,
            "Qdl": (tau_slow**alpha) / r_slow,
            "alpha": alpha,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def new_v2_models() -> tuple[Any, ...]:
    """Models introduced in Generation 2, ordered by complexity."""

    return (
        Zarc1Model(),
        Zarc1GerischerModel(),
        TLMBlockingModel(),
        TLMChargeTransferModel(),
        Zarc2Model(),
        ZarcSEITLMModel(),
        Zarc2FLWModel(),
        Zarc2FSWModel(),
        Zarc3Model(),
        Zarc2FractionalFLWModel(),
        Zarc3FLWModel(),
    )


def all_ecm_models_v2(include_gen1: bool = True) -> tuple[Any, ...]:
    """Combined Generation-1 + Generation-2 model registry."""

    models: list[Any] = list(_gen1_models()) if include_gen1 else []
    models.extend(new_v2_models())
    return tuple(models)


def ecm_model_by_name_v2(name: str) -> Any:
    """Look up any Gen-1 or Gen-2 model by short name."""

    for model in all_ecm_models_v2():
        if model.name == name:
            return model
    available = [m.name for m in all_ecm_models_v2()]
    raise ValueError(f"Unknown ECM model '{name}'. Available: {available}")


def suggest_models_v2(
    n_peaks: int,
    has_diffusion_tail: bool = False,
    has_inductive: bool = False,
    include_gen1_baselines: bool = True,
) -> tuple[Any, ...]:
    """Suggest Gen-2 candidates from DRT features.

    Two lessons from benchmarking shape this pool:

    * **Simple baselines are always offered.**  A single depressed arc
      (low α) frequently splits into several DRT peaks, so the peak
      count alone must never exclude the low-order chain the
      information criterion needs as a reference.
    * **Bounded diffusion is always offered once two processes are
      visible.**  A finite-length Warburg with a large τ_d appears in
      the DRT as a slow *peak*, not as a rising tail, so gating the FLW
      candidates on ``has_diffusion_tail`` alone systematically hides
      the true model.  The effective-dof AICc is the arbiter instead.
    """

    candidates: list[Any] = [Zarc1Model(), Zarc2Model()]

    if n_peaks <= 1:
        if has_diffusion_tail:
            candidates.append(Zarc1GerischerModel())
            candidates.append(Zarc2FLWModel())
        if has_inductive:
            candidates.append(TLMChargeTransferModel())
    elif n_peaks == 2:
        candidates.append(ZarcSEITLMModel())
        candidates.append(Zarc2FLWModel())
        candidates.append(Zarc3Model())
        if has_diffusion_tail:
            candidates.append(Zarc2FSWModel())
            candidates.append(Zarc2FractionalFLWModel())
    else:
        candidates.append(Zarc3Model())
        candidates.append(ZarcSEITLMModel())
        candidates.append(Zarc2FLWModel())
        candidates.append(Zarc3FLWModel())
        if has_diffusion_tail:
            candidates.append(Zarc2FSWModel())

    if include_gen1_baselines:
        from .ecm_library import suggest_models_from_peaks

        for gen1 in suggest_models_from_peaks(n_peaks, has_diffusion_tail, has_inductive):
            if gen1.name not in {c.name for c in candidates}:
                candidates.append(gen1)

    return tuple(candidates)
