# learning AI at www.haotianblog.com
"""Pre-fit data quality assessment for experimental EIS data.

Provides noise estimation, inductive-behavior detection, low-frequency
drift analysis, Kramers–Kronig residual validation, and composite
quality scoring prior to parameter identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _estimate_noise_from_smoothness(freq_hz: FloatArray, z_obs: ComplexArray) -> float:
    """Estimate measurement noise from adjacent-point impedance smoothness.

    For frequency-sorted data the impedance should evolve smoothly on a
    logarithmic frequency axis.  The normalised adjacent-point difference

        δ_i = |Z[i+1] - Z[i]| / |Z[i]| / |log10(f[i+1]/f[i])|

    should be small for clean data.  The *median* of δ_i provides a
    robust estimate that is insensitive to a few outliers.

    Parameters
    ----------
    freq_hz : FloatArray
        Positive frequencies in Hertz (need not be sorted).
    z_obs : ComplexArray
        Observed complex impedance.

    Returns
    -------
    float
        Estimated relative noise level (dimensionless, ≥ 0).
    """
    if freq_hz.size < 3:
        return 0.0

    order = np.argsort(freq_hz)
    f_sorted = freq_hz[order]
    z_sorted = z_obs[order]

    dz = np.abs(np.diff(z_sorted))
    z_mag = np.abs(z_sorted[:-1])
    z_mag = np.maximum(z_mag, 1e-30)  # avoid division by zero

    log_spacing = np.abs(np.diff(np.log10(f_sorted)))
    log_spacing = np.maximum(log_spacing, 1e-30)

    normalised = dz / z_mag / log_spacing
    return float(np.median(normalised))


def _lin_kk_validate(
    freq_hz: FloatArray,
    z_obs: ComplexArray,
    n_rc: int | None = None,
) -> FloatArray:
    """Kramers–Kronig validation via the Lin-KK method.

    Places *n_rc* Voigt elements with logarithmically spaced time
    constants across the measured frequency band, together with an ohmic
    resistance *R₀* and an inductive term *L*.  The model impedance is

        Z(ω) = R₀ + jωL + Σ_k  R_k / (1 + jωτ_k)

    A *real* linear least-squares problem is solved by stacking the real
    and imaginary rows.

    Parameters
    ----------
    freq_hz : FloatArray
        Positive frequencies (Hz).
    z_obs : ComplexArray
        Measured complex impedance.
    n_rc : int or None
        Number of Voigt (RC) elements.  Defaults to
        ``max(5, 3 × ceil(n_decades))``.

    Returns
    -------
    FloatArray
        Per-point relative residuals  |Z_fit - Z_obs| / |Z_obs|.
    """
    n_pts = freq_hz.size
    if n_pts < 3:
        return np.zeros(n_pts, dtype=np.float64)

    omega = 2.0 * np.pi * freq_hz  # (n_pts,)

    # Determine number of RC elements
    n_decades = np.log10(freq_hz.max() / freq_hz.min())
    if n_rc is None:
        n_rc = max(5, int(np.ceil(3.0 * n_decades)))
    n_rc = min(n_rc, n_pts - 2)  # keep system overdetermined
    n_rc = max(n_rc, 1)

    # Logarithmically spaced time constants spanning the frequency band
    tau = np.logspace(
        np.log10(1.0 / (2.0 * np.pi * freq_hz.max())),
        np.log10(1.0 / (2.0 * np.pi * freq_hz.min())),
        n_rc,
    )  # (n_rc,)

    # --- Build complex design matrix (n_pts × n_cols) ---
    # Columns: [R₀ (1), L (jω), R_1/(1+jωτ_1), ..., R_{n_rc}/(1+jωτ_{n_rc})]
    n_cols = 2 + n_rc
    A_complex = np.empty((n_pts, n_cols), dtype=np.complex128)
    A_complex[:, 0] = 1.0                     # R₀
    A_complex[:, 1] = 1j * omega              # jωL
    for k in range(n_rc):
        A_complex[:, 2 + k] = 1.0 / (1.0 + 1j * omega * tau[k])

    # Stack real and imaginary for real-valued lstsq
    A_real = np.vstack([A_complex.real, A_complex.imag])  # (2·n_pts, n_cols)
    b_real = np.concatenate([z_obs.real, z_obs.imag])     # (2·n_pts,)

    coeffs, _, _, _ = np.linalg.lstsq(A_real, b_real, rcond=None)

    z_fit = A_complex @ coeffs
    residuals = np.abs(z_fit - z_obs) / np.maximum(np.abs(z_obs), 1e-30)
    return residuals.astype(np.float64)


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataQualityReport:
    """Pre-fit assessment of EIS measurement quality.

    Attributes
    ----------
    total_points : int
        Total number of frequency points in the dataset.
    valid_points : int
        Number of points passing all quality checks.
    outlier_indices : NDArray[np.intp]
        Indices of detected statistical outliers.
    inductive_indices : NDArray[np.intp]
        Indices where Im(Z) > 0 (inductive behaviour).
    low_freq_drift_indices : NDArray[np.intp]
        Indices exhibiting low-frequency drift.
    kk_residuals : FloatArray | None
        Per-point relative KK residuals (None if validation skipped).
    kk_violation_indices : NDArray[np.intp]
        Indices where KK residual exceeds 2 % of |Z|.
    estimated_noise_level : float
        Global noise estimate (relative, dimensionless).
    frequency_weights : FloatArray
        Per-point recommended weights for fitting (0–1).
    recommended_freq_range : tuple[float, float]
        (f_low, f_high) in Hz for the clean frequency band.
    quality_score : float
        Composite quality metric in [0, 1].
    warnings : tuple[str, ...]
        Human-readable warning strings.
    """

    total_points: int
    valid_points: int
    outlier_indices: NDArray[np.intp]
    inductive_indices: NDArray[np.intp]
    low_freq_drift_indices: NDArray[np.intp]
    kk_residuals: FloatArray | None
    kk_violation_indices: NDArray[np.intp]
    estimated_noise_level: float
    frequency_weights: FloatArray
    recommended_freq_range: tuple[float, float]
    quality_score: float  # 0–1
    warnings: tuple[str, ...]

    # ---- DataFrame / CSV helpers ----

    def to_frame(self) -> pd.DataFrame:
        """Return a per-point quality summary as a :class:`~pandas.DataFrame`.

        Columns include the point index, whether the point is flagged as
        an outlier / inductive / drifting, the KK residual (if
        available), and the frequency weight.
        """
        n = self.total_points
        idx = np.arange(n)

        is_outlier = np.isin(idx, self.outlier_indices)
        is_inductive = np.isin(idx, self.inductive_indices)
        is_drift = np.isin(idx, self.low_freq_drift_indices)
        is_kk_violation = np.isin(idx, self.kk_violation_indices)

        data: dict[str, object] = {
            "point_index": idx,
            "is_outlier": is_outlier,
            "is_inductive": is_inductive,
            "is_low_freq_drift": is_drift,
            "is_kk_violation": is_kk_violation,
            "frequency_weight": self.frequency_weights,
        }
        if self.kk_residuals is not None:
            data["kk_residual"] = self.kk_residuals

        return pd.DataFrame(data)

    def export_csv(self, path: str | Path) -> Path:
        """Write the per-point quality table to CSV.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        Path
            Resolved output path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)
        return output_path

    def summary(self) -> str:
        """Return a human-readable text summary of the quality report."""
        lines = [
            "EIS Data Quality Report",
            "=" * 40,
            f"Total points:            {self.total_points}",
            f"Valid points:            {self.valid_points}",
            f"Outliers:                {self.outlier_indices.size}",
            f"Inductive points:        {self.inductive_indices.size}",
            f"Low-frequency drift:     {self.low_freq_drift_indices.size}",
            f"KK violations:           {self.kk_violation_indices.size}",
            f"Estimated noise level:   {self.estimated_noise_level:.4f}"
            f"  ({self.estimated_noise_level * 100:.2f} %)",
            f"Recommended freq range:  {self.recommended_freq_range[0]:.4g}"
            f" – {self.recommended_freq_range[1]:.4g} Hz",
            f"Quality score:           {self.quality_score:.3f}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assess_data_quality(
    dataset: EISDataset,
    *,
    kk_threshold: float = 0.02,
    noise_threshold: float = 0.02,
    drift_multiplier: float = 3.0,
    run_kk: bool = True,
    n_rc: int | None = None,
) -> DataQualityReport:
    """Run all pre-fit quality checks on an :class:`EISDataset`.

    Parameters
    ----------
    dataset : EISDataset
        The impedance data to evaluate.
    kk_threshold : float, optional
        Relative KK-residual threshold for flagging a point (default 0.02).
    noise_threshold : float, optional
        Noise-level threshold for generating a warning (default 0.02).
    drift_multiplier : float, optional
        Multiple of estimated noise level used as the drift detection
        threshold (default 3.0).
    run_kk : bool, optional
        Whether to run Lin-KK validation (default ``True``).
    n_rc : int | None, optional
        Number of RC elements for the KK test; *None* for automatic.

    Returns
    -------
    DataQualityReport
        Comprehensive quality assessment.

    Raises
    ------
    ValueError
        If *kk_threshold*, *noise_threshold*, or *drift_multiplier* are
        not finite and positive.
    """
    # ---- Input validation ------------------------------------------------
    if not np.isfinite(kk_threshold) or kk_threshold <= 0:
        raise ValueError("kk_threshold must be finite and positive")
    if not np.isfinite(noise_threshold) or noise_threshold <= 0:
        raise ValueError("noise_threshold must be finite and positive")
    if not np.isfinite(drift_multiplier) or drift_multiplier <= 0:
        raise ValueError("drift_multiplier must be finite and positive")
    if n_rc is not None and (n_rc < 1 or not np.isfinite(n_rc)):
        raise ValueError("n_rc must be a positive integer or None")

    freq_hz = dataset.freq_hz
    z_obs = dataset.z_obs
    n_pts = freq_hz.size
    kk_residuals = None

    # ---- (a) Noise level estimation --------------------------------------
    if dataset.z_true is not None:
        residuals = np.abs(z_obs - dataset.z_true) / np.maximum(
            np.abs(dataset.z_true), 1e-30
        )
        noise_level = float(np.median(residuals))
    elif run_kk and n_pts >= 3:
        # Run KK early to use its residuals for highly accurate noise estimation
        kk_residuals = _lin_kk_validate(freq_hz, z_obs, n_rc=n_rc)
        noise_level = float(np.median(kk_residuals))
    else:
        noise_level = _estimate_noise_from_smoothness(freq_hz, z_obs)

    # ---- (b) Inductive behaviour detection --------------------------------
    inductive_mask = z_obs.imag > 0
    inductive_indices = np.where(inductive_mask)[0].astype(np.intp)

    # ---- (c) Low-frequency drift detection --------------------------------
    order = np.argsort(freq_hz)
    f_sorted = freq_hz[order]
    z_sorted = z_obs[order]

    drift_threshold = max(drift_multiplier * noise_level, 1e-6)
    drift_mask_sorted = np.zeros(n_pts, dtype=bool)

    if n_pts >= 3:
        # Check non-monotonic Re(Z) at the low-frequency end
        re_sorted = z_sorted.real
        for i in range(1, n_pts - 1):
            # Interpolate expected value from neighbours
            t = (np.log10(f_sorted[i]) - np.log10(f_sorted[i - 1])) / max(
                np.log10(f_sorted[i + 1]) - np.log10(f_sorted[i - 1]), 1e-30
            )
            z_interp = z_sorted[i - 1] + t * (z_sorted[i + 1] - z_sorted[i - 1])
            deviation = np.abs(z_sorted[i] - z_interp) / max(
                np.abs(z_sorted[i]), 1e-30
            )
            if deviation > drift_threshold:
                drift_mask_sorted[i] = True

        # Also flag if Re(Z) is non-monotonic in the lowest-frequency quarter
        low_quarter = max(n_pts // 4, 2)
        re_low = re_sorted[:low_quarter]
        for i in range(1, len(re_low)):
            # Re(Z) typically increases or stays flat toward low f for
            # typical electrochemical systems.  Flag reversals.
            if re_low[i] < re_low[i - 1] - drift_threshold * np.abs(re_low[i - 1]):
                drift_mask_sorted[i] = True

    # Map back to original indices
    low_freq_drift_indices_sorted = np.where(drift_mask_sorted)[0]
    low_freq_drift_indices = order[low_freq_drift_indices_sorted].astype(np.intp)

    # ---- (d) Kramers-Kronig residuals ------------------------------------
    if run_kk and n_pts >= 3:
        if kk_residuals is None:
            kk_residuals = _lin_kk_validate(freq_hz, z_obs, n_rc=n_rc)
        kk_violation_mask = kk_residuals > kk_threshold
        kk_violation_indices = np.where(kk_violation_mask)[0].astype(np.intp)
    else:
        kk_violation_indices = np.array([], dtype=np.intp)
        kk_residuals = None

    # ---- Outlier detection (union of severe flags) -----------------------
    outlier_set: set[int] = set()
    # Points with very large KK residuals (> 3× threshold) are outliers
    if kk_residuals is not None:
        severe_kk = np.where(kk_residuals > 3.0 * kk_threshold)[0]
        outlier_set.update(severe_kk.tolist())
    # Points that are both inductive *and* have drift are clear outliers
    outlier_set.update(
        set(inductive_indices.tolist()) & set(low_freq_drift_indices.tolist())
    )
    outlier_indices = np.array(sorted(outlier_set), dtype=np.intp)

    # ---- (e) Recommended frequency range ---------------------------------
    # All bad indices (in original order)
    bad_indices: set[int] = set()
    bad_indices.update(outlier_indices.tolist())
    bad_indices.update(inductive_indices.tolist())
    bad_indices.update(low_freq_drift_indices.tolist())
    if kk_residuals is not None:
        bad_indices.update(kk_violation_indices.tolist())

    good_mask = np.ones(n_pts, dtype=bool)
    for idx in bad_indices:
        good_mask[idx] = False

    if np.any(good_mask):
        good_freqs = freq_hz[good_mask]
        recommended_freq_range = (float(good_freqs.min()), float(good_freqs.max()))
    else:
        # Fallback: full range
        recommended_freq_range = (float(freq_hz.min()), float(freq_hz.max()))

    # ---- (h) Frequency weights -------------------------------------------
    weights = np.ones(n_pts, dtype=np.float64)
    # Zero weight for flagged points
    for idx in bad_indices:
        weights[idx] = 0.0

    # Taper weights at frequency-band edges (cosine roll-off over 10 % of
    # the recommended band on each side in log-frequency space)
    if np.any(good_mask):
        log_f = np.log10(freq_hz)
        log_low = np.log10(recommended_freq_range[0])
        log_high = np.log10(recommended_freq_range[1])
        band_width = log_high - log_low
        if band_width > 0:
            taper = 0.1 * band_width
            for i in range(n_pts):
                if weights[i] > 0:
                    if log_f[i] < log_low + taper:
                        t = (log_f[i] - log_low) / max(taper, 1e-30)
                        weights[i] *= float(
                            0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
                        )
                    elif log_f[i] > log_high - taper:
                        t = (log_high - log_f[i]) / max(taper, 1e-30)
                        weights[i] *= float(
                            0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
                        )

    valid_points = int(np.sum(good_mask))

    # ---- (f) Quality score -----------------------------------------------
    valid_frac = valid_points / max(n_pts, 1)
    noise_term = max(0.0, 1.0 - noise_level / 0.05)
    median_kk = (
        float(np.median(kk_residuals)) if kk_residuals is not None else 0.0
    )
    kk_term = max(0.0, 1.0 - median_kk / 0.02)
    log_range_full = np.log10(freq_hz.max() / freq_hz.min()) if n_pts > 1 else 1.0
    log_range_good = (
        np.log10(recommended_freq_range[1] / recommended_freq_range[0])
        if recommended_freq_range[1] > recommended_freq_range[0]
        else 0.0
    )
    coverage = log_range_good / max(log_range_full, 1e-30)
    coverage = min(coverage, 1.0)

    quality_score = float(
        0.4 * valid_frac
        + 0.3 * noise_term
        + 0.2 * kk_term
        + 0.1 * coverage
    )
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    # ---- (g) Warnings ----------------------------------------------------
    warnings_list: list[str] = []

    if inductive_indices.size > 0:
        ind_freqs = freq_hz[inductive_indices]
        min_ind = float(ind_freqs.min())
        label = _format_freq(min_ind)
        warnings_list.append(
            f"High-frequency inductive behavior detected above {label} "
            f"({inductive_indices.size} point"
            f"{'s' if inductive_indices.size != 1 else ''})"
        )

    if low_freq_drift_indices.size > 0:
        drift_freqs = freq_hz[low_freq_drift_indices]
        max_drift = float(drift_freqs.max())
        label = _format_freq(max_drift)
        warnings_list.append(
            f"Low-frequency drift detected below {label} "
            f"({low_freq_drift_indices.size} point"
            f"{'s' if low_freq_drift_indices.size != 1 else ''})"
        )

    if noise_level > noise_threshold:
        warnings_list.append(
            f"Estimated noise level ({noise_level * 100:.1f}%) exceeds "
            f"typical threshold ({noise_threshold * 100:.0f}%)"
        )

    if kk_residuals is not None and kk_violation_indices.size > 0:
        med_kk_pct = median_kk * 100
        warnings_list.append(
            f"Kramers-Kronig residuals exceed {kk_threshold * 100:.0f}% at "
            f"{kk_violation_indices.size} point"
            f"{'s' if kk_violation_indices.size != 1 else ''} "
            f"(median KK residual: {med_kk_pct:.2f}%)"
        )

    if outlier_indices.size > 0:
        warnings_list.append(
            f"Detected {outlier_indices.size} severe outlier"
            f"{'s' if outlier_indices.size != 1 else ''}"
        )

    if valid_frac < 0.5:
        warnings_list.append(
            f"Only {valid_frac * 100:.0f}% of points passed quality checks; "
            f"consider re-measuring"
        )

    return DataQualityReport(
        total_points=n_pts,
        valid_points=valid_points,
        outlier_indices=outlier_indices,
        inductive_indices=inductive_indices,
        low_freq_drift_indices=low_freq_drift_indices,
        kk_residuals=kk_residuals,
        kk_violation_indices=kk_violation_indices,
        estimated_noise_level=noise_level,
        frequency_weights=weights,
        recommended_freq_range=recommended_freq_range,
        quality_score=quality_score,
        warnings=tuple(warnings_list),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _format_freq(freq_hz: float) -> str:
    """Format a frequency value with an appropriate SI prefix."""
    if freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.1f} MHz"
    if freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.1f} kHz"
    if freq_hz >= 1.0:
        return f"{freq_hz:.2f} Hz"
    if freq_hz >= 1e-3:
        return f"{freq_hz * 1e3:.2f} mHz"
    return f"{freq_hz:.2e} Hz"
