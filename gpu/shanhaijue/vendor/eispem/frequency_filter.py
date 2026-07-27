# learning AI at www.haotianblog.com
"""Physics-aware frequency band selection for EIS data.

Provides tools to classify EIS frequency points into quality bands,
detect inductive tails, low-frequency drift, outliers, and validate
Kramers-Kronig consistency via a simplified Lin-KK test.  The result
is a per-point quality weight that downstream fitting routines can use
to down-weight or exclude unreliable measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset

if TYPE_CHECKING:
    pass  # reserved for future protocol imports

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrequencyBandConfig:
    """Configurable thresholds for physics-aware frequency filtering.

    Parameters
    ----------
    low_mid_boundary : float
        Frequency in Hz separating the low and mid bands.
    mid_high_boundary : float
        Frequency in Hz separating the mid and high bands.
    max_inductive_residual : float
        Relative residual threshold for inductive tail detection.
    max_low_freq_scatter : float
        Scatter threshold for low-frequency drift detection.
    enable_kramers_kronig : bool
        Whether to run the simplified Lin-KK validation.
    kk_residual_threshold : float
        Per-point Lin-KK residual threshold (fraction).
    min_points_per_decade : int
        Minimum acceptable frequency density (informational).
    """

    low_mid_boundary: float = 1.0
    mid_high_boundary: float = 1000.0
    max_inductive_residual: float = 0.15
    max_low_freq_scatter: float = 0.20
    enable_kramers_kronig: bool = True
    kk_residual_threshold: float = 0.02
    min_points_per_decade: int = 3

    def __post_init__(self) -> None:
        if not np.isfinite(self.low_mid_boundary) or self.low_mid_boundary <= 0:
            raise ValueError("low_mid_boundary must be finite and positive")
        if not np.isfinite(self.mid_high_boundary) or self.mid_high_boundary <= 0:
            raise ValueError("mid_high_boundary must be finite and positive")
        if self.low_mid_boundary >= self.mid_high_boundary:
            raise ValueError(
                "low_mid_boundary must be strictly less than mid_high_boundary"
            )
        for name, value in (
            ("max_inductive_residual", self.max_inductive_residual),
            ("max_low_freq_scatter", self.max_low_freq_scatter),
            ("kk_residual_threshold", self.kk_residual_threshold),
        ):
            if not np.isfinite(value) or value <= 0 or value >= 1:
                raise ValueError(f"{name} must be in the open interval (0, 1)")
        if self.min_points_per_decade < 1:
            raise ValueError("min_points_per_decade must be at least 1")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrequencyFilterResult:
    """Stores per-point filtering decisions and quality weights.

    Parameters
    ----------
    freq_hz : FloatArray
        Original frequencies (Hz).
    valid_mask : NDArray[np.bool_]
        ``True`` for points that should be kept.
    weights : FloatArray
        Per-point quality weights in [0, 1].
    band_labels : tuple[str, ...]
        ``'low'``, ``'mid'``, or ``'high'`` per frequency point.
    inductive_mask : NDArray[np.bool_]
        ``True`` where inductive behaviour was detected.
    drift_mask : NDArray[np.bool_]
        ``True`` where low-frequency drift was detected.
    kk_residuals : FloatArray | None
        Per-point Lin-KK residuals (``None`` when KK is disabled).
    outlier_mask : NDArray[np.bool_]
        ``True`` for detected outlier points.
    n_removed : int
        Total number of points removed.
    removal_reasons : dict[str, int]
        Count of removed points per reason category.
    """

    freq_hz: FloatArray
    valid_mask: NDArray[np.bool_]
    weights: FloatArray
    band_labels: tuple[str, ...]
    inductive_mask: NDArray[np.bool_]
    drift_mask: NDArray[np.bool_]
    kk_residuals: FloatArray | None
    outlier_mask: NDArray[np.bool_]
    n_removed: int
    removal_reasons: dict[str, int]

    # -- convenience properties -------------------------------------------

    @property
    def valid_freq_hz(self) -> FloatArray:
        """Frequencies retained after filtering."""
        return self.freq_hz[self.valid_mask]

    @property
    def valid_fraction(self) -> float:
        """Fraction of original points that are valid."""
        if self.valid_mask.size == 0:
            return 0.0
        return float(np.sum(self.valid_mask) / len(self.valid_mask))

    def to_frame(self) -> pd.DataFrame:
        """Return a :class:`~pandas.DataFrame` with all per-point info."""
        data: dict[str, object] = {
            "freq_Hz": self.freq_hz,
            "band": list(self.band_labels),
            "valid": self.valid_mask,
            "weight": self.weights,
            "inductive": self.inductive_mask,
            "drift": self.drift_mask,
            "outlier": self.outlier_mask,
        }
        if self.kk_residuals is not None:
            data["kk_residual"] = self.kk_residuals
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrequencyBandAnalyzer:
    """Physics-aware frequency band analysis for EIS datasets.

    Fields
    ------
    config : FrequencyBandConfig
        Configurable thresholds (defaults are sensible for typical PEM cells).
    """

    config: FrequencyBandConfig = field(default_factory=FrequencyBandConfig)

    def analyze(self, dataset: EISDataset) -> FrequencyFilterResult:
        """Classify, validate and weight every frequency point in *dataset*.

        Parameters
        ----------
        dataset : EISDataset
            An EIS observation set (must contain ``freq_hz`` and ``z_obs``).

        Returns
        -------
        FrequencyFilterResult
            Per-point masks, weights, and summary statistics.
        """
        if not isinstance(dataset, EISDataset):
            raise TypeError("dataset must be an EISDataset instance")

        freq = dataset.freq_hz
        z_obs = dataset.z_obs
        n = freq.size
        cfg = self.config

        # (a) Band labelling -----------------------------------------------
        band_labels = _classify_bands(freq, cfg.low_mid_boundary, cfg.mid_high_boundary)

        # (b) Inductive tail detection -------------------------------------
        inductive_mask = _detect_inductive_tail(freq, z_obs, band_labels, cfg)

        # (c) Low-frequency drift detection --------------------------------
        drift_mask = _detect_low_freq_drift(freq, z_obs, band_labels, cfg)

        # (d) Outlier detection (Nyquist curvature) ------------------------
        outlier_mask = _detect_outliers(z_obs)

        # (e) Kramers-Kronig validation (Lin-KK) --------------------------
        kk_residuals: FloatArray | None = None
        kk_fail_mask = np.zeros(n, dtype=bool)
        if cfg.enable_kramers_kronig and n >= 4:
            kk_residuals = _lin_kk_residuals(freq, z_obs)
            kk_fail_mask = kk_residuals > cfg.kk_residual_threshold

        # (f) Weight computation -------------------------------------------
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            label = band_labels[i]
            if label == "high" and not inductive_mask[i]:
                weights[i] *= 0.7
            elif label == "low" and not drift_mask[i]:
                weights[i] *= 0.8
            # mid band keeps weight 1.0

            if kk_residuals is not None and not kk_fail_mask[i]:
                # Down-weight points approaching the KK threshold
                ratio = kk_residuals[i] / cfg.kk_residual_threshold
                if ratio > 0.5:
                    weights[i] *= max(0.1, 1.0 - ratio)

            if kk_fail_mask[i]:
                weights[i] *= max(0.1, 1.0 - kk_residuals[i] / cfg.kk_residual_threshold)

            if outlier_mask[i] or inductive_mask[i] or drift_mask[i]:
                weights[i] = 0.0

        # (g) Valid mask ----------------------------------------------------
        valid_mask = (
            (weights > 0)
            & ~outlier_mask
            & ~inductive_mask
            & ~drift_mask
        )

        # Summary -----------------------------------------------------------
        n_removed = int(np.sum(~valid_mask))
        removal_reasons: dict[str, int] = {
            "inductive": int(np.sum(inductive_mask & ~valid_mask)),
            "drift": int(np.sum(drift_mask & ~valid_mask)),
            "outlier": int(np.sum(outlier_mask & ~valid_mask)),
        }
        if kk_residuals is not None:
            # KK failures that are *not already* counted under other masks
            kk_only = kk_fail_mask & ~inductive_mask & ~drift_mask & ~outlier_mask & ~valid_mask
            removal_reasons["kk_fail"] = int(np.sum(kk_only))

        return FrequencyFilterResult(
            freq_hz=freq,
            valid_mask=valid_mask,
            weights=weights,
            band_labels=tuple(band_labels),
            inductive_mask=inductive_mask,
            drift_mask=drift_mask,
            kk_residuals=kk_residuals,
            outlier_mask=outlier_mask,
            n_removed=n_removed,
            removal_reasons=removal_reasons,
        )


def analyze_stacked_frequency_bands(
    dataset: EISDataset,
    conditions: tuple[tuple[float, float], ...],
    config: FrequencyBandConfig | None = None,
) -> FrequencyFilterResult:
    """Run frequency-quality analysis independently for each condition block."""

    if not conditions:
        raise ValueError("conditions cannot be empty")
    if dataset.freq_hz.size % len(conditions) != 0:
        raise ValueError("dataset must contain one equal-size block per condition")
    if len(conditions) == 1:
        return FrequencyBandAnalyzer(config or FrequencyBandConfig()).analyze(dataset)

    analyzer = FrequencyBandAnalyzer(config or FrequencyBandConfig())
    freq_blocks = np.split(dataset.freq_hz, len(conditions))
    obs_blocks = np.split(dataset.z_obs, len(conditions))
    results = [
        analyzer.analyze(EISDataset(freq_hz=freq, z_obs=z_obs))
        for freq, z_obs in zip(freq_blocks, obs_blocks, strict=True)
    ]
    kk_present = all(result.kk_residuals is not None for result in results)
    reasons: dict[str, int] = {}
    for result in results:
        for reason, count in result.removal_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count

    return FrequencyFilterResult(
        freq_hz=dataset.freq_hz,
        valid_mask=np.concatenate([result.valid_mask for result in results]),
        weights=np.concatenate([result.weights for result in results]),
        band_labels=tuple(
            label for result in results for label in result.band_labels
        ),
        inductive_mask=np.concatenate([result.inductive_mask for result in results]),
        drift_mask=np.concatenate([result.drift_mask for result in results]),
        kk_residuals=(
            np.concatenate([result.kk_residuals for result in results])
            if kk_present
            else None
        ),
        outlier_mask=np.concatenate([result.outlier_mask for result in results]),
        n_removed=sum(result.n_removed for result in results),
        removal_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Helper functions (public API)
# ---------------------------------------------------------------------------

def filter_dataset(
    dataset: EISDataset, result: FrequencyFilterResult
) -> EISDataset:
    """Return a new :class:`EISDataset` containing only valid frequency points.

    Parameters
    ----------
    dataset : EISDataset
        The original dataset.
    result : FrequencyFilterResult
        A filtering result compatible with *dataset*.

    Returns
    -------
    EISDataset
        A copy restricted to valid points.

    Raises
    ------
    ValueError
        If no valid points remain or shapes are incompatible.
    """
    _validate_filter_args(dataset, result)
    mask = result.valid_mask
    if not np.any(mask):
        raise ValueError("No valid frequency points remain after filtering")

    context: dict[str, FloatArray] | None = None
    if dataset.context is not None:
        context = {k: v[mask] for k, v in dataset.context.items()}

    return EISDataset(
        freq_hz=dataset.freq_hz[mask],
        z_obs=dataset.z_obs[mask],
        z_true=dataset.z_true[mask] if dataset.z_true is not None else None,
        context=context,
    )


def weighted_dataset(
    dataset: EISDataset, result: FrequencyFilterResult
) -> tuple[EISDataset, FloatArray]:
    """Return the filtered dataset together with its non-zero weights.

    Parameters
    ----------
    dataset : EISDataset
        The original dataset.
    result : FrequencyFilterResult
        A filtering result compatible with *dataset*.

    Returns
    -------
    tuple[EISDataset, FloatArray]
        ``(filtered_dataset, weights)`` where *weights* are the non-zero
        quality weights corresponding to the retained points.

    Raises
    ------
    ValueError
        If no valid points remain or shapes are incompatible.
    """
    filtered = filter_dataset(dataset, result)
    valid_weights = result.weights[result.valid_mask]
    return filtered, valid_weights


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_filter_args(
    dataset: EISDataset, result: FrequencyFilterResult
) -> None:
    if dataset.freq_hz.shape != result.freq_hz.shape:
        raise ValueError(
            "dataset and result frequency arrays must have the same shape"
        )
    if not np.allclose(dataset.freq_hz, result.freq_hz):
        raise ValueError(
            "dataset and result frequency arrays must contain the same values"
        )


def _classify_bands(
    freq: FloatArray, low_mid: float, mid_high: float
) -> list[str]:
    """Assign each frequency to 'low', 'mid', or 'high'."""
    labels: list[str] = []
    for f in freq:
        if f < low_mid:
            labels.append("low")
        elif f <= mid_high:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


def _detect_inductive_tail(
    freq: FloatArray,
    z_obs: ComplexArray,
    band_labels: list[str],
    cfg: FrequencyBandConfig,
) -> NDArray[np.bool_]:
    """Mark inductive-behaviour points (Im(Z) > 0) in the high-freq band.

    Once consecutive high-frequency points with positive imaginary part are
    found, every point from the first such point to the end of the high-freq
    band is flagged.  An additional slope check (|Im(Z)| growing with
    frequency) supplements the detection.
    """
    n = freq.size
    mask = np.zeros(n, dtype=bool)
    imag = z_obs.imag

    # Sort indices by ascending frequency for slope analysis
    order = np.argsort(freq)

    # --- Consecutive positive-imaginary detection in high band ---
    first_inductive: int | None = None
    for idx in order:
        if band_labels[idx] == "high" and imag[idx] > 0:
            if first_inductive is None:
                first_inductive = idx
            mask[idx] = True
        elif band_labels[idx] == "high" and first_inductive is not None:
            # Still mark after first inductive point in high band
            mask[idx] = True

    # If we found a first inductive point, mark everything from there onward
    # (in frequency order) within the high-frequency band.
    if first_inductive is not None:
        found = False
        for idx in order:
            if idx == first_inductive:
                found = True
            if found and band_labels[idx] == "high":
                mask[idx] = True

    # --- Inductive slope check across all bands ---
    # If |Im(Z)| is growing with frequency for consecutive points where
    # Im(Z) > 0, mark them (indicative of inductance tail).
    sorted_imag = imag[order]
    sorted_freq = freq[order]
    for k in range(1, n):
        if sorted_imag[k] > 0 and sorted_imag[k - 1] > 0:
            if np.abs(sorted_imag[k]) > np.abs(sorted_imag[k - 1]):
                mask[order[k]] = True
                mask[order[k - 1]] = True

    return mask


def _detect_low_freq_drift(
    freq: FloatArray,
    z_obs: ComplexArray,
    band_labels: list[str],
    cfg: FrequencyBandConfig,
) -> NDArray[np.bool_]:
    """Detect scatter and non-monotonic drift in the low-frequency band."""
    n = freq.size
    mask = np.zeros(n, dtype=bool)

    # Identify low-band indices sorted by frequency
    low_indices = np.array(
        [i for i in range(n) if band_labels[i] == "low"], dtype=int
    )
    if low_indices.size < 3:
        return mask

    order = low_indices[np.argsort(freq[low_indices])]

    # --- Smoothness check: 3-point deviation from linear interpolation ---
    for k in range(1, len(order) - 1):
        i0, i1, i2 = order[k - 1], order[k], order[k + 1]
        f0, f1, f2 = freq[i0], freq[i1], freq[i2]

        # Linear interpolation weight
        denom = f2 - f0
        if denom == 0:
            continue
        t = (f1 - f0) / denom
        z_interp = z_obs[i0] + t * (z_obs[i2] - z_obs[i0])

        z_interp_mag = np.abs(z_interp)
        if z_interp_mag < 1e-30:
            continue

        deviation = np.abs(z_obs[i1] - z_interp) / z_interp_mag
        if deviation > cfg.max_low_freq_scatter:
            mask[i1] = True

    # --- Non-monotonic real-part behaviour (sign of drift) ---
    re_vals = z_obs[order].real
    for k in range(1, len(order)):
        # In typical EIS the real part should evolve monotonically at low freq.
        # Non-monotonic behaviour signals drift or contact artefacts.
        if k >= 2:
            d1 = re_vals[k - 1] - re_vals[k - 2]
            d2 = re_vals[k] - re_vals[k - 1]
            if d1 * d2 < 0:
                # Sign reversal in the real-part trend
                mask[order[k]] = True

    return mask


def _detect_outliers(z_obs: ComplexArray) -> NDArray[np.bool_]:
    """Detect Nyquist-plot outliers via curvature and modified Z-score.

    Uses two complementary criteria:
    1. Abrupt curvature sign change or magnitude > 5× median curvature.
    2. Modified Z-score of curvature magnitude > 3.5.
    """
    n = z_obs.size
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask

    # Work in Nyquist coordinates: (Re(Z), -Im(Z))
    x = z_obs.real.copy()
    y = -z_obs.imag.copy()

    # Local curvature via finite differences
    # κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    dx = np.diff(x)
    dy = np.diff(y)

    # Second derivatives (at interior points)
    ddx = np.diff(dx)
    ddy = np.diff(dy)

    # First derivatives at interior points (average of neighbours)
    dx_mid = 0.5 * (dx[:-1] + dx[1:])
    dy_mid = 0.5 * (dy[:-1] + dy[1:])

    denom = (dx_mid**2 + dy_mid**2) ** 1.5
    safe_denom = np.where(denom > 1e-30, denom, 1e-30)

    curvature = (dx_mid * ddy - dy_mid * ddx) / safe_denom
    abs_curvature = np.abs(curvature)

    if abs_curvature.size == 0:
        return mask

    # Criterion 1: magnitude > 5× median curvature
    median_curv = np.median(abs_curvature)
    if median_curv > 0:
        high_curv = abs_curvature > 5.0 * median_curv
        for k in range(len(high_curv)):
            if high_curv[k]:
                # Interior index k corresponds to z_obs index k + 1
                mask[k + 1] = True

    # Criterion 1b: abrupt curvature sign change
    for k in range(1, len(curvature)):
        if curvature[k] * curvature[k - 1] < 0:
            ratio = abs_curvature[k] / max(abs_curvature[k - 1], 1e-30)
            if ratio > 5.0 or 1.0 / max(ratio, 1e-30) > 5.0:
                mask[k + 1] = True

    # Criterion 2: Modified Z-score on curvature magnitude
    mad = np.median(np.abs(abs_curvature - median_curv))
    if mad > 1e-12:
        z_scores = 0.6745 * (abs_curvature - median_curv) / mad
        for k in range(len(z_scores)):
            if np.abs(z_scores[k]) > 3.5:
                mask[k + 1] = True

    return mask


def _lin_kk_residuals(
    freq: FloatArray, z_obs: ComplexArray
) -> FloatArray:
    """Simplified Lin-KK test: fit Voigt elements and return per-point residuals.

    Distributes RC (Voigt) elements logarithmically across the frequency range
    (~3 per decade) and solves for the resistances via linear least squares.
    The per-point relative residual is ``|Z_fit - Z_obs| / |Z_obs|``.
    """
    n = freq.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    omega = 2.0 * np.pi * freq

    # Distribute time constants τ_k logarithmically
    log_f_min = np.log10(np.min(freq))
    log_f_max = np.log10(np.max(freq))
    decades = max(log_f_max - log_f_min, 0.5)
    n_elements = max(int(np.round(3.0 * decades)), 2)
    # Extend slightly beyond measured range for edge stability
    tau_k = 1.0 / (
        2.0 * np.pi * np.logspace(log_f_min - 0.2, log_f_max + 0.2, n_elements)
    )

    # Build basis matrix A such that Z_model = R_inf + sum_k R_k / (1 + jωτ_k)
    # Column 0: constant (R_inf)
    # Columns 1..n_elements: Voigt element basis functions
    n_cols = 1 + n_elements
    A_real = np.zeros((n, n_cols), dtype=np.float64)
    A_imag = np.zeros((n, n_cols), dtype=np.float64)

    A_real[:, 0] = 1.0  # R_inf contributes only to real part
    A_imag[:, 0] = 0.0

    for k in range(n_elements):
        jwτ = 1j * omega * tau_k[k]
        basis = 1.0 / (1.0 + jwτ)
        A_real[:, k + 1] = basis.real
        A_imag[:, k + 1] = basis.imag

    # Stack real and imaginary parts for a real-valued least-squares solve
    A = np.vstack([A_real, A_imag])
    b = np.concatenate([z_obs.real, z_obs.imag])

    # Solve with non-negative least squares is ideal but plain lstsq suffices
    # for residual diagnostics.
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Reconstruct fitted impedance
    z_fit = (A_real @ coeffs) + 1j * (A_imag @ coeffs)

    # Per-point relative residual
    z_obs_mag = np.abs(z_obs)
    safe_mag = np.where(z_obs_mag > 1e-30, z_obs_mag, 1e-30)
    residuals = np.abs(z_fit - z_obs) / safe_mag

    return residuals.astype(np.float64)
