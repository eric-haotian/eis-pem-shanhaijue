# learning AI at www.haotianblog.com
"""Forward simulation models for electrochemical impedance spectra."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class ForwardModel(Protocol):
    """Interface implemented by any model usable by the PEM workflow."""

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        """Return complex impedance values at each supplied frequency."""

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        """Return an informed initial guess from DRT peaks.

        If not supported, returns None.
        """

class RandlesModel:
    """Five-parameter Randles-like impedance model with CPE and Inductance."""

    parameter_names = ("Rs", "Rct", "Qdl", "alpha", "L")

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        frequencies = np.asarray(freq_hz, dtype=float)
        parameters = np.asarray(theta, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("freq_hz must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("freq_hz values must be finite and positive")
        if parameters.shape != (5,):
            raise ValueError("theta must contain [Rs, Rct, Qdl, alpha, L]")
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
            raise ValueError("Randles parameters must be finite and positive")

        rs, rct, qdl, alpha, l_ind = parameters
        omega = 2.0 * np.pi * frequencies
        cpe_term = (1j * omega * rct * qdl) ** alpha
        return np.asarray(
            rs + 1j * omega * l_ind + rct / (1.0 + cpe_term), dtype=complex
        )

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return None
