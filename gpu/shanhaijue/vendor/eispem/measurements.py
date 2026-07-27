# learning AI at www.haotianblog.com
"""Independent scalar parameter observations for joint PEM identification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .parameters import ParameterSpec

FloatArray = NDArray[np.float64]


@dataclass
class ParameterMeasurementDataset:
    """Auxiliary physical measurements associated with named model parameters."""

    parameter_names: tuple[str, ...]
    observed_values: FloatArray
    units: tuple[str, ...]
    relative_noise_std: FloatArray
    true_values: FloatArray | None = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.observed_values = np.asarray(self.observed_values, dtype=float)
        self.relative_noise_std = np.asarray(self.relative_noise_std, dtype=float)
        if self.true_values is not None:
            self.true_values = np.asarray(self.true_values, dtype=float)
        count = len(self.parameter_names)
        if not self.parameter_names:
            raise ValueError("measurement parameter_names cannot be empty")
        if len(self.units) != count:
            raise ValueError("measurement units must match parameter_names")
        if self.observed_values.shape != (count,) or self.relative_noise_std.shape != (
            count,
        ):
            raise ValueError("measurement arrays must match parameter_names")
        if self.true_values is not None and self.true_values.shape != (count,):
            raise ValueError("measurement true_values must match parameter_names")
        if not np.all(np.isfinite(self.observed_values)) or np.any(
            self.observed_values <= 0
        ):
            raise ValueError("measurement observed_values must be finite and positive")
        if not np.all(np.isfinite(self.relative_noise_std)) or np.any(
            self.relative_noise_std < 0
        ):
            raise ValueError("measurement relative_noise_std must be non-negative")
        if self.true_values is not None and (
            not np.all(np.isfinite(self.true_values)) or np.any(self.true_values <= 0)
        ):
            raise ValueError("measurement true_values must be finite and positive")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def relative_residuals(
        self, model_parameter_names: tuple[str, ...], theta: NDArray[np.floating]
    ) -> FloatArray:
        """Return dimensionless residuals for named scalar observations."""

        parameters = np.asarray(theta, dtype=float)
        if parameters.shape != (len(model_parameter_names),):
            raise ValueError("theta must match model_parameter_names")
        index = {name: position for position, name in enumerate(model_parameter_names)}
        missing = set(self.parameter_names).difference(index)
        if missing:
            raise ValueError(
                f"measured parameters not found in model: {sorted(missing)}"
            )
        predicted = np.asarray(
            [parameters[index[name]] for name in self.parameter_names], dtype=float
        )
        scale = np.maximum(np.abs(self.observed_values), self.eps)
        return (self.observed_values - predicted) / scale

    def to_frame(self) -> pd.DataFrame:
        """Return the scalar calibration observations for audit/export."""

        data: dict[str, object] = {
            "parameter": self.parameter_names,
            "observed_value": self.observed_values,
            "unit": self.units,
            "relative_noise_std": self.relative_noise_std,
        }
        if self.true_values is not None:
            data["true_value"] = self.true_values
        return pd.DataFrame(data)

    def export_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)
        return output_path


def generate_synthetic_parameter_measurements(
    parameter_specs: Sequence[ParameterSpec],
    measured_names: Sequence[str],
    theta_true: NDArray[np.floating],
    noise_level: float = 0.005,
    seed: int = 142,
    replicates: int = 1,
) -> ParameterMeasurementDataset:
    """Generate independent relative-noise scalar measurements for selected inputs."""

    specs = tuple(parameter_specs)
    names = tuple(measured_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("measured_names must be unique and non-empty")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    values = np.asarray(theta_true, dtype=float)
    if values.shape != (len(specs),) or not np.all(np.isfinite(values)):
        raise ValueError("theta_true must be finite and match parameter_specs")
    specification = {spec.name: spec for spec in specs}
    index = {spec.name: position for position, spec in enumerate(specs)}
    missing = set(names).difference(specification)
    if missing:
        raise ValueError(f"unknown measured parameters: {sorted(missing)}")
    observation_names = tuple(name for name in names for _ in range(replicates))
    true_values = np.asarray(
        [values[index[name]] for name in observation_names], dtype=float
    )
    rng = np.random.default_rng(seed)
    observed_values = true_values * (
        1.0 + noise_level * rng.normal(size=len(observation_names))
    )
    return ParameterMeasurementDataset(
        parameter_names=observation_names,
        observed_values=observed_values,
        units=tuple(specification[name].unit for name in observation_names),
        relative_noise_std=np.full(len(observation_names), noise_level, dtype=float),
        true_values=true_values,
    )
