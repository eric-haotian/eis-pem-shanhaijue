# learning AI at www.haotianblog.com
"""Structured identification result output."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset

if TYPE_CHECKING:
    from .parameters import ParameterSpec
    from .robust import ParameterSelection


@dataclass
class IdentificationResult:
    """Estimated parameters, fitted impedance, residuals and optimiser details."""

    theta_best: dict[str, float]
    final_cost: float
    z_fit: NDArray[np.complex128]
    residuals: NDArray[np.complex128]
    dataset: EISDataset
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z_fit = np.asarray(self.z_fit, dtype=complex)
        self.residuals = np.asarray(self.residuals, dtype=complex)
        if self.z_fit.shape != self.dataset.freq_hz.shape:
            raise ValueError("z_fit must have the same shape as the dataset")
        if self.residuals.shape != self.dataset.freq_hz.shape:
            raise ValueError("residuals must have the same shape as the dataset")
        if not np.isfinite(self.final_cost) or self.final_cost < 0:
            raise ValueError("final_cost must be finite and non-negative")

    def to_frame(self, impedance_unit: str = "ohm") -> pd.DataFrame:
        """Return the documented fitted spectrum and residual table."""

        if not impedance_unit or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for char in impedance_unit
        ):
            raise ValueError("impedance_unit must be a non-empty CSV-safe suffix")
        data: dict[str, NDArray[np.floating]] = {}
        if self.dataset.context is not None:
            data.update(self.dataset.context)
        data.update(
            {
                "freq_Hz": self.dataset.freq_hz,
                f"Zreal_obs_{impedance_unit}": self.dataset.z_obs.real,
                f"Zimag_obs_{impedance_unit}": self.dataset.z_obs.imag,
                f"Zreal_fit_{impedance_unit}": self.z_fit.real,
                f"Zimag_fit_{impedance_unit}": self.z_fit.imag,
            }
        )
        if self.dataset.z_true is not None:
            data[f"Zreal_true_{impedance_unit}"] = self.dataset.z_true.real
            data[f"Zimag_true_{impedance_unit}"] = self.dataset.z_true.imag
        data[f"residual_real_{impedance_unit}"] = self.residuals.real
        data[f"residual_imag_{impedance_unit}"] = self.residuals.imag
        return pd.DataFrame(data)

    def export_fit_csv(self, path: str | Path, impedance_unit: str = "ohm") -> Path:
        """Write the fitted spectrum table to disk."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame(impedance_unit=impedance_unit).to_csv(output_path, index=False)
        return output_path

    def export_parameters_csv(
        self,
        path: str | Path,
        parameter_specs: Sequence[ParameterSpec],
        theta_true: NDArray[np.floating] | None = None,
        selection: ParameterSelection | None = None,
    ) -> Path:
        """Write estimated parameter values and optional synthetic errors."""

        specs = tuple(parameter_specs)
        if (
            selection is not None
            and tuple(spec.name for spec in specs) != selection.parameter_names
        ):
            raise ValueError("selection parameter order must match parameter_specs")
        identified_values = []
        for spec in specs:
            if spec.name in self.theta_best:
                identified_values.append(self.theta_best[spec.name])
            elif selection is not None and spec.name in selection.fixed_values:
                identified_values.append(selection.fixed_values[spec.name])
            else:
                raise ValueError(
                    f"no identified or fixed value available for {spec.name}"
                )
        data: dict[str, list[Any]] = {
            "parameter": [spec.name for spec in specs],
            "initial_value": [spec.initial_value for spec in specs],
            "identified_value": identified_values,
            "unit": [spec.unit for spec in specs],
        }
        if selection is not None:
            data["status"] = [selection.statuses[spec.name] for spec in specs]
            data["reason"] = [selection.reasons[spec.name] for spec in specs]
            data["reference_value"] = [
                selection.reference_values[spec.name] for spec in specs
            ]
            data["ci95_relative"] = [
                selection.ci95_relative[spec.name] for spec in specs
            ]
        if theta_true is not None:
            true_values = np.asarray(theta_true, dtype=float)
            if true_values.shape != (len(specs),):
                raise ValueError("theta_true must match parameter_specs length")
            identified = np.asarray(data["identified_value"], dtype=float)
            data["true_value"] = list(true_values)
            errors = np.abs(identified - true_values) / np.abs(true_values)
            if selection is not None:
                errors = np.where(
                    [
                        selection.statuses[spec.name] == "fixed_identifiability"
                        for spec in specs
                    ],
                    np.nan,
                    errors,
                )
            data["relative_error"] = list(errors)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(output_path, index=False)
        return output_path
