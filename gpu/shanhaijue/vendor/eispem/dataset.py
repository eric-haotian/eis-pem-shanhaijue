# learning AI at www.haotianblog.com
"""EIS observation containers and deterministic synthetic data generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .forward_models import ForwardModel

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class EISDataset:
    """Frequency-domain EIS observations and optional generating truth."""

    freq_hz: FloatArray
    z_obs: ComplexArray
    z_true: ComplexArray | None = None
    context: dict[str, FloatArray] | None = None

    def __post_init__(self) -> None:
        self.freq_hz = np.asarray(self.freq_hz, dtype=float)
        self.z_obs = np.asarray(self.z_obs, dtype=complex)
        if self.z_true is not None:
            self.z_true = np.asarray(self.z_true, dtype=complex)
        if self.context is not None:
            self.context = {
                name: np.asarray(values, dtype=float)
                for name, values in self.context.items()
            }

        if self.freq_hz.ndim != 1 or self.freq_hz.size == 0:
            raise ValueError("freq_hz must be a non-empty one-dimensional array")
        if self.z_obs.shape != self.freq_hz.shape:
            raise ValueError("z_obs must have the same shape as freq_hz")
        if self.z_true is not None and self.z_true.shape != self.freq_hz.shape:
            raise ValueError("z_true must have the same shape as freq_hz")
        if not np.all(np.isfinite(self.freq_hz)) or np.any(self.freq_hz <= 0):
            raise ValueError("freq_hz values must be finite and positive")
        if not _complex_values_are_finite(self.z_obs):
            raise ValueError("z_obs values must be finite")
        if self.z_true is not None and not _complex_values_are_finite(self.z_true):
            raise ValueError("z_true values must be finite")
        if self.context is not None:
            for name, values in self.context.items():
                if values.shape != self.freq_hz.shape:
                    raise ValueError(f"context column {name} must match freq_hz shape")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"context column {name} values must be finite")

    def to_frame(self, impedance_unit: str = "ohm") -> pd.DataFrame:
        """Return observations with the documented CSV column contract."""

        unit = _csv_unit_suffix(impedance_unit)
        data: dict[str, FloatArray] = {}
        if self.context is not None:
            data.update(self.context)
        data.update(
            {
                "freq_Hz": self.freq_hz,
                f"Zreal_{unit}": self.z_obs.real,
                f"Zimag_{unit}": self.z_obs.imag,
            }
        )
        if self.z_true is not None:
            data[f"Zreal_true_{unit}"] = self.z_true.real
            data[f"Zimag_true_{unit}"] = self.z_true.imag
        return pd.DataFrame(data)

    def to_csv(self, path: str | Path, impedance_unit: str = "ohm") -> Path:
        """Write observations to CSV and return the resolved output path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame(impedance_unit=impedance_unit).to_csv(output_path, index=False)
        return output_path

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        impedance_unit: str = "ohm",
        context_columns: tuple[str, ...] = (),
    ) -> EISDataset:
        """Read an observation CSV following the public column names."""

        frame = pd.read_csv(path)
        unit = _csv_unit_suffix(impedance_unit)
        required = {"freq_Hz", f"Zreal_{unit}", f"Zimag_{unit}"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing required EIS CSV columns: {sorted(missing)}")
        true_columns = {f"Zreal_true_{unit}", f"Zimag_true_{unit}"}
        present_true_columns = true_columns.intersection(frame.columns)
        if present_true_columns and present_true_columns != true_columns:
            raise ValueError("both true impedance columns must be provided together")

        z_obs = (
            frame[f"Zreal_{unit}"].to_numpy() + 1j * frame[f"Zimag_{unit}"].to_numpy()
        )
        z_true = None
        if present_true_columns == true_columns:
            z_true = (
                frame[f"Zreal_true_{unit}"].to_numpy()
                + 1j * frame[f"Zimag_true_{unit}"].to_numpy()
            )
        missing_context = set(context_columns).difference(frame.columns)
        if missing_context:
            raise ValueError(f"missing context CSV columns: {sorted(missing_context)}")
        context = (
            {name: frame[name].to_numpy() for name in context_columns}
            if context_columns
            else None
        )
        return cls(frame["freq_Hz"].to_numpy(), z_obs, z_true, context=context)


def generate_synthetic_dataset(
    model: ForwardModel,
    freq_hz: NDArray[np.floating],
    theta_true: NDArray[np.floating],
    noise_level: float = 0.005,
    seed: int = 42,
) -> EISDataset:
    """Generate complex EIS observations with pointwise relative noise."""

    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")
    frequencies = np.asarray(freq_hz, dtype=float)
    z_true = model.simulate(frequencies, np.asarray(theta_true, dtype=float))
    rng = np.random.default_rng(seed)
    sigma = noise_level * np.abs(z_true)
    noise = sigma * (
        rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)
    )
    return EISDataset(freq_hz=frequencies, z_obs=z_true + noise, z_true=z_true)


def _complex_values_are_finite(values: ComplexArray) -> bool:
    return bool(np.all(np.isfinite(values.real)) and np.all(np.isfinite(values.imag)))


def _csv_unit_suffix(unit: str) -> str:
    if not unit or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for char in unit
    ):
        raise ValueError("impedance_unit must be a non-empty CSV-safe suffix")
    return unit
