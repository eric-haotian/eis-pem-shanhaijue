# learning AI at www.haotianblog.com
"""Parameter metadata and optimisation-space transformations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Specification of one physical model parameter."""

    name: str
    initial_value: float
    bounds: tuple[float, float]
    unit: str
    log_transform: bool = False

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not self.name:
            raise ValueError("parameter name cannot be empty")
        if not np.all(np.isfinite([lower, upper, self.initial_value])):
            raise ValueError(f"{self.name} values and bounds must be finite")
        if lower >= upper:
            raise ValueError(f"{self.name} lower bound must be below upper bound")
        if not lower <= self.initial_value <= upper:
            raise ValueError(f"{self.name} initial value must be within its bounds")
        if self.log_transform and lower <= 0:
            raise ValueError(f"{self.name} log transformed bounds must be positive")

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        """Bounds in the search space used by the optimizer."""

        if self.log_transform:
            return (float(np.log10(self.bounds[0])), float(np.log10(self.bounds[1])))
        return self.bounds

    def to_optimization(self, physical_value: float) -> float:
        """Map one physical parameter value to optimiser coordinates."""

        if not np.isfinite(physical_value):
            raise ValueError(f"{self.name} physical value must be finite")
        if self.log_transform:
            if physical_value <= 0:
                raise ValueError(f"{self.name} log transformed value must be positive")
            return float(np.log10(physical_value))
        return float(physical_value)

    def from_optimization(self, optimized_value: float) -> float:
        """Map one optimiser coordinate back to a physical parameter value."""

        if not np.isfinite(optimized_value):
            raise ValueError(f"{self.name} optimized value must be finite")
        if self.log_transform:
            return float(10.0**optimized_value)
        return float(optimized_value)


def encode_parameter_vector(
    specs: Sequence[ParameterSpec],
    physical_values: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Map a physical parameter vector to optimizer coordinates."""

    spec_tuple = tuple(specs)
    values = np.asarray(physical_values, dtype=float)
    if values.shape != (len(spec_tuple),):
        raise ValueError("physical_values must match parameter_specs length")
    if not np.all(np.isfinite(values)):
        raise ValueError("physical_values must be finite")
    encoded = values.copy()
    log_mask = np.asarray([spec.log_transform for spec in spec_tuple], dtype=bool)
    if np.any(log_mask) and np.any(encoded[log_mask] <= 0):
        bad = [spec_tuple[i].name for i in np.flatnonzero(log_mask & (encoded <= 0))]
        raise ValueError(f"log-transformed parameters must be positive: {bad}")
    encoded[log_mask] = np.log10(encoded[log_mask])
    return encoded


def decode_parameter_vector(
    specs: Sequence[ParameterSpec],
    optimization_values: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Map an optimizer-coordinate vector back to physical parameters."""

    spec_tuple = tuple(specs)
    values = np.asarray(optimization_values, dtype=float)
    if values.shape != (len(spec_tuple),):
        raise ValueError("optimization_values must match parameter_specs length")
    if not np.all(np.isfinite(values)):
        raise ValueError("optimization_values must be finite")
    decoded = values.copy()
    log_mask = np.asarray([spec.log_transform for spec in spec_tuple], dtype=bool)
    decoded[log_mask] = 10.0 ** decoded[log_mask]
    return decoded


def optimization_bounds_array(
    specs: Sequence[ParameterSpec],
) -> NDArray[np.float64]:
    """Return an ``(n, 2)`` optimizer-space bounds array."""

    return np.asarray([spec.optimization_bounds for spec in specs], dtype=float)


def initial_optimization_vector(
    specs: Sequence[ParameterSpec],
) -> NDArray[np.float64]:
    """Return initial parameter values in optimizer coordinates."""

    return encode_parameter_vector(
        specs,
        np.asarray([spec.initial_value for spec in specs], dtype=float),
    )
