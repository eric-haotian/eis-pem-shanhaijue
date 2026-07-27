# learning AI at www.haotianblog.com
"""Prediction error costs for complex EIS observations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dataset import EISDataset
from .forward_models import ForwardModel


@dataclass
class EISPredictionErrorCost:
    """Squared real/imaginary residual cost with optional scaling."""

    dataset: EISDataset
    model: ForwardModel
    weights: NDArray[np.floating] | None = None
    relative: bool = False
    modulus_weighting: bool = False
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != self.dataset.freq_hz.shape:
                raise ValueError(
                    "weights must have the same shape as dataset frequencies"
                )
            if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
                raise ValueError("weights must be finite and non-negative")
        if self.modulus_weighting and self.weights is None:
            self.weights = 1.0 / np.maximum(np.abs(self.dataset.z_obs), self.eps)

    def residuals(self, theta: NDArray[np.floating]) -> NDArray[np.complex128]:
        """Return complex residual contributions used in the scalar cost."""

        residuals = self.dataset.z_obs - self.model.simulate(
            self.dataset.freq_hz, theta
        )
        if self.relative:
            scale = np.maximum(np.abs(self.dataset.z_obs), self.eps)
            residuals = residuals / scale
        if self.weights is not None:
            residuals = residuals * np.sqrt(self.weights)
        return np.asarray(residuals, dtype=complex)

    def __call__(self, theta: NDArray[np.floating]) -> float:
        residuals = self.residuals(theta)
        return float(np.sum(residuals.real**2 + residuals.imag**2))
