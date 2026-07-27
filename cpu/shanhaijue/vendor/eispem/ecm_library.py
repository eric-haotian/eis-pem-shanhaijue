# learning AI at www.haotianblog.com
"""Equivalent Circuit Model (ECM) library for EIS parameter identification.

Provides a family of common circuit models with increasing complexity,
all implementing the :class:`~eis_pem.forward_models.ForwardModel` protocol.
Each model carries its own :class:`~eis_pem.parameters.ParameterSpec` list
so the full identification pipeline (selection → optimization → diagnostics)
works out-of-the-box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .parameters import ParameterSpec

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Impedance element helpers
# ---------------------------------------------------------------------------


def _z_cpe(omega: FloatArray, q: float, alpha: float) -> ComplexArray:
    """CPE impedance: Z = 1 / (Q (jω)^α)."""
    return 1.0 / (q * (1j * omega) ** alpha)


def _z_warburg_semi_inf(omega: FloatArray, aw: float) -> ComplexArray:
    """Semi-infinite Warburg: Z = A_W / √(jω)."""
    return aw / np.sqrt(1j * omega)


def _z_warburg_finite(omega: FloatArray, rd: float, tau_d: float) -> ComplexArray:
    """Finite-length Warburg: Z = R_d · tanh(√(jωτ)) / √(jωτ)."""
    s = np.sqrt(1j * omega * tau_d)
    return rd * np.tanh(s) / s


def _z_rc_parallel(omega: FloatArray, r: float, c: float) -> ComplexArray:
    """Parallel RC: Z = R / (1 + jωRC)."""
    return r / (1.0 + 1j * omega * r * c)


def _z_rcpe_parallel(omega: FloatArray, r: float, q: float, alpha: float) -> ComplexArray:
    """Parallel R‖CPE: Z = R / (1 + R·Q·(jω)^α)."""
    return r / (1.0 + r * q * (1j * omega) ** alpha)


def _validate_eis_inputs(
    freq_hz: NDArray[np.floating], theta: NDArray[np.floating], n_params: int
) -> tuple[FloatArray, FloatArray]:
    """Validate and cast EIS model inputs."""
    f = np.asarray(freq_hz, dtype=float)
    t = np.asarray(theta, dtype=float)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("freq_hz must be a non-empty 1-D array")
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("freq_hz must be finite and positive")
    if t.shape != (n_params,):
        raise ValueError(f"theta must have shape ({n_params},), got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("theta must be finite")
    return f, t


# ---------------------------------------------------------------------------
# ECM models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleRCModel:
    """Rs + R₁‖C₁ — simplest single-arc model.

    Parameters: Rs, R1, C1
    """

    name: str = "1RC"
    circuit_string: str = "Rs + R1||C1"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("R1", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("C1", 1.0, (1e-4, 1e4), "F", log_transform=True),
        )

    @property
    def n_params(self) -> int:
        return 3

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        r_idx = np.argmax(drt_result.peak_resistances)
        rs = float(drt_result.r_inf)
        r1 = float(drt_result.peak_resistances[r_idx])
        tau = float(drt_result.peak_taus[r_idx])
        return {"Rs": max(rs, 1e-6), "R1": max(r1, 1e-6), "C1": tau / max(r1, 1e-6)}

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs, r1, c1 = t
        omega = 2.0 * np.pi * f
        return np.asarray(rs + _z_rc_parallel(omega, r1, c1), dtype=complex)


@dataclass(frozen=True)
class SingleRCPEModel:
    """Rs + R₁‖CPE₁ — single depressed semicircle.

    Parameters: Rs, R1, Q1, alpha1
    """

    name: str = "1RCPE"
    circuit_string: str = "Rs + R1||CPE1"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("R1", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q1", 1.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha1", 0.85, (0.5, 1.0), "1", log_transform=False),
        )

    @property
    def n_params(self) -> int:
        return 4

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        r_idx = np.argmax(drt_result.peak_resistances)
        rs = float(drt_result.r_inf)
        r1 = float(drt_result.peak_resistances[r_idx])
        tau = float(drt_result.peak_taus[r_idx])
        alpha = 0.85
        q1 = (tau ** alpha) / max(r1, 1e-6)
        return {"Rs": max(rs, 1e-6), "R1": max(r1, 1e-6), "Q1": q1, "alpha1": alpha}

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs, r1, q1, alpha1 = t
        omega = 2.0 * np.pi * f
        return np.asarray(rs + _z_rcpe_parallel(omega, r1, q1, alpha1), dtype=complex)


@dataclass(frozen=True)
class TwoRCPEModel:
    """Rs + R₁‖CPE₁ + R₂‖CPE₂ — SEI + charge transfer.

    Parameters: Rs, R1, Q1, alpha1, R2, Q2, alpha2
    """

    name: str = "2RCPE"
    circuit_string: str = "Rs + R1||CPE1 + R2||CPE2"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("R1", 0.01, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q1", 10.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha1", 0.85, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("R2", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q2", 1.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha2", 0.80, (0.5, 1.0), "1", log_transform=False),
        )

    @property
    def n_params(self) -> int:
        return 7

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        rs = max(float(drt_result.r_inf), 1e-6)

        if drt_result.n_peaks == 1:
            # Single visible arc — split into two with different time constants
            r_total = max(float(drt_result.peak_resistances[0]), 1e-6)
            tau_main = float(drt_result.peak_taus[0])
            # Arc 1: smaller, faster (SEI-like), ~30% of total R
            r1 = max(r_total * 0.3, 1e-6)
            t1 = tau_main / 5.0  # faster time constant
            a1 = 0.90
            q1 = (t1 ** a1) / r1
            # Arc 2: larger, slower (charge transfer), ~70% of total R
            r2 = max(r_total * 0.7, 1e-6)
            t2 = tau_main * 2.0  # slower time constant
            a2 = 0.80
            q2 = (t2 ** a2) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": a1,
                "R2": r2, "Q2": q2, "alpha2": a2,
            }

        idx = np.argsort(drt_result.peak_taus)
        r1 = max(float(drt_result.peak_resistances[idx[0]]), 1e-6)
        t1 = float(drt_result.peak_taus[idx[0]])
        r2 = max(float(drt_result.peak_resistances[idx[1]]), 1e-6)
        t2 = float(drt_result.peak_taus[idx[1]])
        a1, a2 = 0.90, 0.80  # separate alphas: high-freq arc more ideal
        q1 = (t1 ** a1) / r1
        q2 = (t2 ** a2) / r2

        return {
            "Rs": rs,
            "R1": r1, "Q1": q1, "alpha1": a1,
            "R2": r2, "Q2": q2, "alpha2": a2,
        }

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs, r1, q1, a1, r2, q2, a2 = t
        omega = 2.0 * np.pi * f
        return np.asarray(
            rs
            + _z_rcpe_parallel(omega, r1, q1, a1)
            + _z_rcpe_parallel(omega, r2, q2, a2),
            dtype=complex,
        )


@dataclass(frozen=True)
class RandlesWarburgModel:
    """Rs + (Rct‖CPE + W) — charge transfer with semi-infinite diffusion.

    The Warburg element is in series with the parallel Rct‖CPE.

    Parameters: Rs, Rct, Qdl, alpha, Aw, L
    """

    name: str = "Randles_W"
    circuit_string: str = "Rs + Rct||(CPE + W) + L"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("Rct", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Qdl", 1.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha", 0.85, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("Aw", 0.01, (1e-8, 10.0), "ohm*s^(-1/2)", log_transform=True),
            ParameterSpec("L", 1e-7, (1e-10, 1e-4), "H", log_transform=True),
        )

    @property
    def n_params(self) -> int:
        return 6

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        r_idx = np.argmax(drt_result.peak_resistances)
        rs = max(float(drt_result.r_inf), 1e-6)
        rct = max(float(drt_result.peak_resistances[r_idx]), 1e-6)
        tau = float(drt_result.peak_taus[r_idx])
        alpha = 0.85
        qdl = (tau ** alpha) / rct
        return {
            "Rs": rs, "Rct": rct, "Qdl": qdl, "alpha": alpha,
            "Aw": 0.01, "L": 1e-7
        }

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs, rct, qdl, alpha, aw, l_ind = t
        omega = 2.0 * np.pi * f
        z_cpe = _z_cpe(omega, qdl, alpha)
        z_w = _z_warburg_semi_inf(omega, aw)
        # Rct in parallel with (CPE + W in series with CPE? No — standard Randles:
        # Rct‖(CPE), then Warburg in series with that parallel combination
        # Actually the standard Randles has W inside the parallel: Z = Rs + jωL + 1/(1/Rct + 1/Zw + Y_CPE)
        # where Y_CPE = Q(jω)^α. Let's use the common form:
        # Z = Rs + jωL + 1 / (1/Rct + 1/(Zw) + Q(jω)^α)  -- NO, this is unusual.
        # Standard Randles: Z = Rs + jωL + Rct/(1 + Rct*Q*(jω)^α) + Warburg
        # More precisely: the faradaic impedance = Rct + Zw, in parallel with Cdl/CPE:
        # Z_faradaic = Rct + Zw
        # Z_parallel = 1 / (1/Z_faradaic + Y_CPE) = Z_faradaic / (1 + Z_faradaic * Y_CPE)
        z_faradaic = rct + z_w
        y_cpe = qdl * (1j * omega) ** alpha
        z_parallel = z_faradaic / (1.0 + z_faradaic * y_cpe)
        return np.asarray(rs + 1j * omega * l_ind + z_parallel, dtype=complex)


@dataclass(frozen=True)
class TwoRCPEWarburgModel:
    """Rs + R₁‖CPE₁ + R₂‖CPE₂ + W — SEI + charge transfer + diffusion.

    Parameters: Rs, R1, Q1, alpha1, R2, Q2, alpha2, Aw, L
    """

    name: str = "2RCPE_W"
    circuit_string: str = "Rs + R1||CPE1 + R2||(CPE2 + W) + L"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("R1", 0.005, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q1", 10.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha1", 0.85, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("R2", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q2", 1.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha2", 0.80, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("Aw", 0.01, (1e-8, 10.0), "ohm*s^(-1/2)", log_transform=True),
            ParameterSpec("L", 1e-7, (1e-10, 1e-4), "H", log_transform=True),
        )

    @property
    def n_params(self) -> int:
        return 9

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        rs = max(float(drt_result.r_inf), 1e-6)
        alpha = 0.85

        if drt_result.n_peaks == 1:
            r_val = max(float(drt_result.peak_resistances[0]) / 2.0, 1e-6)
            tau_val = float(drt_result.peak_taus[0])
            q_val = (tau_val ** alpha) / r_val
            return {
                "Rs": rs,
                "R1": r_val, "Q1": q_val, "alpha1": alpha,
                "R2": r_val, "Q2": q_val, "alpha2": alpha,
                "Aw": 0.01, "L": 1e-7
            }

        idx = np.argsort(drt_result.peak_taus)
        r1 = max(float(drt_result.peak_resistances[idx[0]]), 1e-6)
        t1 = float(drt_result.peak_taus[idx[0]])
        r2 = max(float(drt_result.peak_resistances[idx[1]]), 1e-6)
        t2 = float(drt_result.peak_taus[idx[1]])
        q1 = (t1 ** alpha) / r1
        q2 = (t2 ** alpha) / r2

        return {
            "Rs": rs,
            "R1": r1, "Q1": q1, "alpha1": alpha,
            "R2": r2, "Q2": q2, "alpha2": alpha,
            "Aw": 0.01, "L": 1e-7
        }

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs, r1, q1, a1, r2, q2, a2, aw, l_ind = t
        omega = 2.0 * np.pi * f
        # First arc: SEI layer (R1‖CPE1)
        z1 = _z_rcpe_parallel(omega, r1, q1, a1)
        # Second arc with Warburg: faradaic = R2 + W, in parallel with CPE2
        z_faradaic = r2 + _z_warburg_semi_inf(omega, aw)
        y_cpe2 = q2 * (1j * omega) ** a2
        z2 = z_faradaic / (1.0 + z_faradaic * y_cpe2)
        return np.asarray(rs + 1j * omega * l_ind + z1 + z2, dtype=complex)


@dataclass(frozen=True)
class ThreeRCPEModel:
    """Rs + R₁‖CPE₁ + R₂‖CPE₂ + R₃‖CPE₃ — three-arc model.

    Suitable for cells with distinct SEI, charge-transfer, and bulk processes.

    Parameters: Rs, R1, Q1, alpha1, R2, Q2, alpha2, R3, Q3, alpha3
    """

    name: str = "3RCPE"
    circuit_string: str = "Rs + R1||CPE1 + R2||CPE2 + R3||CPE3"

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("Rs", 0.01, (1e-6, 1.0), "ohm", log_transform=True),
            ParameterSpec("R1", 0.005, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q1", 50.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha1", 0.90, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("R2", 0.02, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q2", 5.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha2", 0.85, (0.5, 1.0), "1", log_transform=False),
            ParameterSpec("R3", 0.05, (1e-6, 10.0), "ohm", log_transform=True),
            ParameterSpec("Q3", 1.0, (1e-6, 1e6), "F*s^(a-1)", log_transform=True),
            ParameterSpec("alpha3", 0.80, (0.5, 1.0), "1", log_transform=False),
        )

    @property
    def n_params(self) -> int:
        return 10

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.parameter_specs)

    def default_theta(self) -> FloatArray:
        return np.array([s.initial_value for s in self.parameter_specs])

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        if drt_result.n_peaks == 0:
            return None
        rs = max(float(drt_result.r_inf), 1e-6)
        alpha = 0.85

        idx = np.argsort(drt_result.peak_taus)

        if drt_result.n_peaks == 1:
            r_val = max(float(drt_result.peak_resistances[0]) / 3.0, 1e-6)
            tau_val = float(drt_result.peak_taus[0])
            q_val = (tau_val ** alpha) / r_val
            return {
                "Rs": rs,
                "R1": r_val, "Q1": q_val, "alpha1": alpha,
                "R2": r_val, "Q2": q_val, "alpha2": alpha,
                "R3": r_val, "Q3": q_val, "alpha3": alpha,
            }
        elif drt_result.n_peaks == 2:
            r1 = max(float(drt_result.peak_resistances[idx[0]]), 1e-6)
            t1 = float(drt_result.peak_taus[idx[0]])
            r2 = max(float(drt_result.peak_resistances[idx[1]]) / 2.0, 1e-6)
            t2 = float(drt_result.peak_taus[idx[1]])
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** alpha) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": alpha,
                "R3": r2, "Q3": q2, "alpha3": alpha,
            }
        else:
            r1 = max(float(drt_result.peak_resistances[idx[0]]), 1e-6)
            t1 = float(drt_result.peak_taus[idx[0]])
            r2 = max(float(drt_result.peak_resistances[idx[1]]), 1e-6)
            t2 = float(drt_result.peak_taus[idx[1]])
            r3 = max(float(drt_result.peak_resistances[idx[2]]), 1e-6)
            t3 = float(drt_result.peak_taus[idx[2]])
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** alpha) / r2
            q3 = (t3 ** alpha) / r3
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": alpha,
                "R3": r3, "Q3": q3, "alpha3": alpha,
            }

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> ComplexArray:
        f, t = _validate_eis_inputs(freq_hz, theta, self.n_params)
        rs = t[0]
        r1, q1, a1 = t[1], t[2], t[3]
        r2, q2, a2 = t[4], t[5], t[6]
        r3, q3, a3 = t[7], t[8], t[9]
        omega = 2.0 * np.pi * f
        return np.asarray(
            rs
            + _z_rcpe_parallel(omega, r1, q1, a1)
            + _z_rcpe_parallel(omega, r2, q2, a2)
            + _z_rcpe_parallel(omega, r3, q3, a3),
            dtype=complex,
        )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Union type for all ECM models
ECMModel = (
    SingleRCModel
    | SingleRCPEModel
    | TwoRCPEModel
    | RandlesWarburgModel
    | TwoRCPEWarburgModel
    | ThreeRCPEModel
)


def all_ecm_models() -> tuple[ECMModel, ...]:
    """Return all available ECM models ordered by complexity."""
    return (
        SingleRCModel(),
        SingleRCPEModel(),
        TwoRCPEModel(),
        RandlesWarburgModel(),
        TwoRCPEWarburgModel(),
        ThreeRCPEModel(),
    )


def ecm_model_by_name(name: str) -> ECMModel:
    """Look up an ECM model by its short name."""
    for model in all_ecm_models():
        if model.name == name:
            return model
    available = [m.name for m in all_ecm_models()]
    raise ValueError(f"Unknown ECM model '{name}'. Available: {available}")


def suggest_models_from_peaks(
    n_peaks: int,
    has_diffusion_tail: bool = False,
    has_inductive: bool = False,
) -> tuple[ECMModel, ...]:
    """Suggest candidate ECM models based on DRT peak count and features.

    Parameters
    ----------
    n_peaks
        Number of peaks detected in the DRT spectrum.
    has_diffusion_tail
        Whether a low-frequency diffusion tail was detected.
    has_inductive
        Whether high-frequency inductive behavior was detected.

    Returns
    -------
    tuple[ECMModel, ...]
        Candidate models ordered by increasing complexity.
    """
    candidates: list[ECMModel] = []

    if n_peaks <= 1:
        candidates.append(SingleRCModel())
        candidates.append(SingleRCPEModel())
        if has_diffusion_tail:
            candidates.append(RandlesWarburgModel())
    elif n_peaks == 2:
        candidates.append(SingleRCPEModel())  # maybe merged arcs
        candidates.append(TwoRCPEModel())
        if has_diffusion_tail:
            candidates.append(RandlesWarburgModel())
            candidates.append(TwoRCPEWarburgModel())
    else:  # 3+
        candidates.append(TwoRCPEModel())
        candidates.append(ThreeRCPEModel())
        if has_diffusion_tail:
            candidates.append(TwoRCPEWarburgModel())

    return tuple(candidates)
