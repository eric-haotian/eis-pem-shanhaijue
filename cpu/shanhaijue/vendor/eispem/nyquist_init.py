# learning AI at www.haotianblog.com
"""Nyquist-plot direct feature extraction for EIS parameter initialisation.

Provides a DRT-independent fallback for generating initial parameter
guesses by analysing the Nyquist plot geometry (Re(Z) vs -Im(Z)).

This module extracts:
- Ohmic resistance (Rs) from the high-frequency intercept
- Arc count, centre frequencies, and approximate resistances from
  -Im(Z) local maxima
- Diffusion tail characteristics from low-frequency 45° behaviour
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal as _signal

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NyquistFeatures:
    """Features extracted directly from the Nyquist plot.

    Attributes
    ----------
    rs_estimate : float
        Ohmic resistance estimated from the highest-frequency Re(Z).
    n_arcs : int
        Number of semicircular arcs detected from -Im(Z) maxima.
    arc_frequencies : FloatArray
        Characteristic frequency (Hz) at each arc centre (ascending τ).
    arc_resistances : FloatArray
        Approximate resistance contribution of each arc.
    arc_taus : FloatArray
        Time constants τ = 1/(2πf_peak) for each arc.
    arc_peak_imag : FloatArray
        Peak -Im(Z) value at each arc centre.
    total_resistance : float
        Total estimated polarisation resistance (sum of arc resistances).
    has_inductive : bool
        Whether high-frequency inductive behaviour (Im(Z) > 0) was detected.
    has_diffusion_tail : bool
        Whether a low-frequency ~45° diffusion tail was detected.
    z_low_freq : complex
        Impedance at the lowest frequency.
    z_high_freq : complex
        Impedance at the highest frequency.
    """

    rs_estimate: float
    n_arcs: int
    arc_frequencies: FloatArray
    arc_resistances: FloatArray
    arc_taus: FloatArray
    arc_peak_imag: FloatArray
    total_resistance: float
    has_inductive: bool
    has_diffusion_tail: bool
    z_low_freq: complex
    z_high_freq: complex

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Nyquist Feature Extraction",
            "=" * 40,
            f"  Rs estimate:       {self.rs_estimate:.4e} Ω",
            f"  Arcs detected:     {self.n_arcs}",
            f"  Total polar. R:    {self.total_resistance:.4e} Ω",
            f"  Inductive tail:    {'Yes' if self.has_inductive else 'No'}",
            f"  Diffusion tail:    {'Yes' if self.has_diffusion_tail else 'No'}",
        ]
        for i in range(self.n_arcs):
            lines.append(
                f"  Arc {i + 1}: f={self.arc_frequencies[i]:.1f} Hz, "
                f"R≈{self.arc_resistances[i]:.4e} Ω, "
                f"τ={self.arc_taus[i]:.2e} s"
            )
        return "\n".join(lines)


def extract_nyquist_features(
    freq_hz: FloatArray,
    z_obs: ComplexArray,
    min_prominence_frac: float = 0.03,
    smoothing_window: int = 5,
) -> NyquistFeatures:
    """Extract physical features directly from Nyquist plot geometry.

    Parameters
    ----------
    freq_hz : FloatArray
        Frequencies in Hz, need not be sorted.
    z_obs : ComplexArray
        Observed complex impedance.
    min_prominence_frac : float
        Minimum relative prominence (fraction of max -Im(Z)) for arc detection.
    smoothing_window : int
        Window size for Savitzky-Golay smoothing of -Im(Z).
        Set to 0 to disable smoothing.

    Returns
    -------
    NyquistFeatures
        Extracted features.
    """
    freq = np.asarray(freq_hz, dtype=float)
    z = np.asarray(z_obs, dtype=complex)

    if freq.size < 3:
        raise ValueError("Need at least 3 frequency points")

    # Sort by frequency descending (high → low) to match EIS convention
    sort_idx = np.argsort(freq)[::-1]
    freq_sorted = freq[sort_idx]
    z_sorted = z[sort_idx]

    # --- Rs from highest frequency ---
    # Use the mean of the top few points for robustness
    n_top = max(1, min(3, freq.size // 10))
    rs = float(np.median(z_sorted[:n_top].real))

    # --- Inductive detection ---
    has_inductive = bool(np.any(z_sorted.imag > 0))

    # --- Work with -Im(Z) for arc detection ---
    neg_imag = -z_sorted.imag
    re_z = z_sorted.real

    # Mask out inductive region (Im(Z) > 0 → -Im(Z) < 0)
    capacitive_mask = neg_imag >= 0
    if np.sum(capacitive_mask) < 3:
        # All inductive — no arcs detectable
        return NyquistFeatures(
            rs_estimate=max(rs, 0.0),
            n_arcs=0,
            arc_frequencies=np.array([]),
            arc_resistances=np.array([]),
            arc_taus=np.array([]),
            arc_peak_imag=np.array([]),
            total_resistance=0.0,
            has_inductive=has_inductive,
            has_diffusion_tail=False,
            z_low_freq=z_sorted[-1],
            z_high_freq=z_sorted[0],
        )

    # --- Smooth -Im(Z) for robust peak finding ---
    if smoothing_window > 0 and neg_imag.size >= smoothing_window >= 3:
        # Use Savitzky-Golay filter for smoothing
        win = min(smoothing_window, neg_imag.size)
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            neg_imag_smooth = _signal.savgol_filter(neg_imag, win, min(2, win - 1))
        else:
            neg_imag_smooth = neg_imag.copy()
    else:
        neg_imag_smooth = neg_imag.copy()

    # --- Find arcs via peaks in -Im(Z) ---
    max_val = np.max(neg_imag_smooth[capacitive_mask])
    if max_val <= 0:
        min_prom = 0
    else:
        min_prom = min_prominence_frac * max_val

    peaks, peak_props = _signal.find_peaks(
        neg_imag_smooth,
        prominence=min_prom,
        distance=max(1, freq.size // 15),
    )

    # Filter out peaks in the inductive region
    valid_peaks = [p for p in peaks if neg_imag[p] > 0]
    peaks = np.array(valid_peaks, dtype=int)

    n_arcs = len(peaks)
    if n_arcs == 0:
        # No clear peaks — try to detect a single broad arc from the global maximum
        cap_indices = np.where(capacitive_mask)[0]
        if cap_indices.size > 0:
            global_max_idx = cap_indices[np.argmax(neg_imag_smooth[cap_indices])]
            if neg_imag[global_max_idx] > 0:
                peaks = np.array([global_max_idx])
                n_arcs = 1

    # --- Compute arc properties ---
    arc_freqs = freq_sorted[peaks] if n_arcs > 0 else np.array([])
    arc_peak_imag = neg_imag[peaks] if n_arcs > 0 else np.array([])
    arc_taus = 1.0 / (2.0 * np.pi * arc_freqs) if n_arcs > 0 else np.array([])

    # Estimate arc resistances from semicircle geometry
    # For a perfect semicircle: R ≈ 2 × peak(-Im(Z))
    # For a depressed semicircle (CPE): R ≈ 2 × peak(-Im(Z)) / sin(α π/2)
    # Use the conservative 2× estimate (assumes α≈1)
    arc_resistances = np.zeros(n_arcs)
    for i, pk in enumerate(peaks):
        # Better estimate: measure the Re(Z) span around this arc
        # Find the half-height crossing points on each side
        half_h = neg_imag[pk] / 2.0

        left = pk
        while left > 0 and neg_imag[left] > half_h:
            left -= 1
        right = pk
        while right < len(neg_imag) - 1 and neg_imag[right] > half_h:
            right += 1

        # Arc resistance ≈ Re(Z_right) - Re(Z_left)
        r_span = abs(re_z[right] - re_z[left])
        # Also use the 2× peak height estimate
        r_peak = 2.0 * neg_imag[pk]

        # Take the average of both estimates for robustness
        arc_resistances[i] = max((r_span + r_peak) / 2.0, 1e-8)

    total_r = float(np.sum(arc_resistances))

    # --- Diffusion tail detection ---
    # Check if the low-frequency end shows ~45° slope (Re and -Im growing together)
    has_diffusion = False
    n_tail = max(3, freq.size // 10)
    if freq.size > n_tail + 2:
        # Low-frequency points (end of sorted array since sorted high→low)
        tail_re = re_z[-n_tail:]
        tail_im = neg_imag[-n_tail:]
        # Check if both Re and -Im are monotonically increasing
        re_increasing = np.sum(np.diff(tail_re) > 0) > n_tail * 0.5
        im_increasing = np.sum(np.diff(tail_im) > 0) > n_tail * 0.5
        if re_increasing and im_increasing:
            # Check approximate 45° slope: d(-Im)/d(Re) ≈ 1
            d_re = tail_re[-1] - tail_re[0]
            d_im = tail_im[-1] - tail_im[0]
            if d_re > 0:
                slope = d_im / d_re
                if 0.3 < slope < 3.0:
                    has_diffusion = True

    # Sort arcs by time constant (ascending = high freq first)
    if n_arcs > 1:
        sort_arc = np.argsort(arc_taus)
        arc_freqs = arc_freqs[sort_arc]
        arc_resistances = arc_resistances[sort_arc]
        arc_taus = arc_taus[sort_arc]
        arc_peak_imag = arc_peak_imag[sort_arc]

    return NyquistFeatures(
        rs_estimate=max(rs, 0.0),
        n_arcs=n_arcs,
        arc_frequencies=arc_freqs,
        arc_resistances=arc_resistances,
        arc_taus=arc_taus,
        arc_peak_imag=arc_peak_imag,
        total_resistance=total_r,
        has_inductive=has_inductive,
        has_diffusion_tail=has_diffusion,
        z_low_freq=z_sorted[-1],
        z_high_freq=z_sorted[0],
    )


def guess_ecm_params(
    features: NyquistFeatures,
    model_name: str,
) -> dict[str, float] | None:
    """Generate initial parameter guesses for an ECM from Nyquist features.

    Parameters
    ----------
    features : NyquistFeatures
        Features extracted from the Nyquist plot.
    model_name : str
        ECM model name (e.g. '1RC', '1RCPE', '2RCPE', etc.)

    Returns
    -------
    dict[str, float] | None
        Parameter name → initial value mapping, or None if not enough data.
    """
    rs = max(features.rs_estimate, 1e-6)

    if model_name == "1RC":
        if features.n_arcs == 0:
            return None
        r1 = max(features.arc_resistances[0], 1e-6)
        tau = max(features.arc_taus[0], 1e-10)
        return {"Rs": rs, "R1": r1, "C1": tau / r1}

    elif model_name == "1RCPE":
        if features.n_arcs == 0:
            return None
        r1 = max(features.arc_resistances[0], 1e-6)
        tau = max(features.arc_taus[0], 1e-10)
        alpha = 0.85
        q1 = (tau ** alpha) / r1
        return {"Rs": rs, "R1": r1, "Q1": q1, "alpha1": alpha}

    elif model_name == "2RCPE":
        if features.n_arcs == 0:
            return None
        alpha = 0.85
        if features.n_arcs == 1:
            # Single arc — split it into two halves
            r_total = max(features.arc_resistances[0], 1e-6)
            tau = max(features.arc_taus[0], 1e-10)
            r1 = r_total * 0.3
            r2 = r_total * 0.7
            # Separate time constants by ~1 decade
            t1 = tau / 3.0
            t2 = tau * 3.0
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** alpha) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": 0.80,
            }
        else:
            r1 = max(features.arc_resistances[0], 1e-6)
            r2 = max(features.arc_resistances[1], 1e-6)
            t1 = max(features.arc_taus[0], 1e-10)
            t2 = max(features.arc_taus[1], 1e-10)
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** 0.80) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": 0.80,
            }

    elif model_name == "Randles_W":
        if features.n_arcs == 0:
            return None
        r_idx = 0 if features.n_arcs == 1 else np.argmax(features.arc_resistances)
        rct = max(features.arc_resistances[r_idx], 1e-6)
        tau = max(features.arc_taus[r_idx], 1e-10)
        alpha = 0.85
        qdl = (tau ** alpha) / rct
        return {
            "Rs": rs, "Rct": rct, "Qdl": qdl, "alpha": alpha,
            "Aw": 0.01, "L": 1e-7,
        }

    elif model_name == "2RCPE_W":
        if features.n_arcs == 0:
            return None
        alpha = 0.85
        if features.n_arcs == 1:
            r_total = max(features.arc_resistances[0], 1e-6)
            tau = max(features.arc_taus[0], 1e-10)
            r1 = r_total * 0.3
            r2 = r_total * 0.7
            t1 = tau / 3.0
            t2 = tau * 3.0
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** 0.80) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": 0.80,
                "Aw": 0.01, "L": 1e-7,
            }
        else:
            r1 = max(features.arc_resistances[0], 1e-6)
            r2 = max(features.arc_resistances[1], 1e-6)
            t1 = max(features.arc_taus[0], 1e-10)
            t2 = max(features.arc_taus[1], 1e-10)
            q1 = (t1 ** alpha) / r1
            q2 = (t2 ** 0.80) / r2
            return {
                "Rs": rs,
                "R1": r1, "Q1": q1, "alpha1": alpha,
                "R2": r2, "Q2": q2, "alpha2": 0.80,
                "Aw": 0.01, "L": 1e-7,
            }

    elif model_name == "3RCPE":
        if features.n_arcs == 0:
            return None
        alpha = 0.85
        if features.n_arcs == 1:
            r_total = max(features.arc_resistances[0], 1e-6)
            tau = max(features.arc_taus[0], 1e-10)
            r1 = r_total * 0.2
            r2 = r_total * 0.3
            r3 = r_total * 0.5
            t1 = tau / 10.0
            t2 = tau
            t3 = tau * 10.0
            return {
                "Rs": rs,
                "R1": r1, "Q1": (t1 ** 0.90) / r1, "alpha1": 0.90,
                "R2": r2, "Q2": (t2 ** alpha) / r2, "alpha2": alpha,
                "R3": r3, "Q3": (t3 ** 0.80) / r3, "alpha3": 0.80,
            }
        elif features.n_arcs == 2:
            r1 = max(features.arc_resistances[0], 1e-6)
            r2 = max(features.arc_resistances[1] * 0.5, 1e-6)
            r3 = max(features.arc_resistances[1] * 0.5, 1e-6)
            t1 = max(features.arc_taus[0], 1e-10)
            t2 = max(features.arc_taus[1] / 2.0, 1e-10)
            t3 = max(features.arc_taus[1] * 2.0, 1e-10)
            return {
                "Rs": rs,
                "R1": r1, "Q1": (t1 ** 0.90) / r1, "alpha1": 0.90,
                "R2": r2, "Q2": (t2 ** alpha) / r2, "alpha2": alpha,
                "R3": r3, "Q3": (t3 ** 0.80) / r3, "alpha3": 0.80,
            }
        else:
            r1 = max(features.arc_resistances[0], 1e-6)
            r2 = max(features.arc_resistances[1], 1e-6)
            r3 = max(features.arc_resistances[2], 1e-6)
            t1 = max(features.arc_taus[0], 1e-10)
            t2 = max(features.arc_taus[1], 1e-10)
            t3 = max(features.arc_taus[2], 1e-10)
            return {
                "Rs": rs,
                "R1": r1, "Q1": (t1 ** 0.90) / r1, "alpha1": 0.90,
                "R2": r2, "Q2": (t2 ** alpha) / r2, "alpha2": alpha,
                "R3": r3, "Q3": (t3 ** 0.80) / r3, "alpha3": 0.80,
            }

    return None
