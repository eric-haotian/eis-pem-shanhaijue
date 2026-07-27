# learning AI at www.haotianblog.com
"""Distribution of Relaxation Times (DRT) analysis for EIS data.

Implements Tikhonov-regularised DRT deconvolution with automatic peak
detection, diffusion tail identification, and model complexity suggestion.

The DRT represents the impedance spectrum as:

    Z(ω) = R_∞ + ∫₀^∞  γ(τ) / (1 + jωτ) d(ln τ)

where γ(τ) is the distribution function.  Peaks in γ correspond to
distinct electrochemical processes (RC arcs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal as _signal

from .dataset import EISDataset

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DRTResult:
    """Result of a DRT analysis.

    Attributes
    ----------
    tau : FloatArray
        Log-spaced time constants (s) at which γ is evaluated.
    gamma : FloatArray
        DRT function γ(τ) — resistance density per unit ln(τ).
    z_fit : ComplexArray
        Impedance reconstructed from the DRT solution.
    residuals : FloatArray
        Per-frequency-point relative residual |Z_fit − Z_obs| / |Z_obs|.
    r_inf : float
        High-frequency ohmic resistance (R∞).
    n_peaks : int
        Number of detected peaks in γ(τ).
    peak_taus : FloatArray
        Time constants (s) at peak positions.
    peak_heights : FloatArray
        γ values at peak positions.
    peak_resistances : FloatArray
        Approximate resistance contribution of each peak (area ≈ height × width).
    regularization_param : float
        Optimal λ used in the Tikhonov solve.
    has_diffusion_tail : bool
        Whether a low-frequency diffusion tail was detected.
    has_inductive : bool
        Whether high-frequency inductive behavior was detected.
    """

    tau: FloatArray
    gamma: FloatArray
    z_fit: ComplexArray
    residuals: FloatArray
    r_inf: float
    n_peaks: int
    peak_taus: FloatArray
    peak_heights: FloatArray
    peak_resistances: FloatArray
    regularization_param: float
    has_diffusion_tail: bool
    has_inductive: bool

    def total_resistance(self) -> float:
        """Total polarisation resistance from DRT integration."""
        if self.tau.size < 2:
            return 0.0
        d_ln_tau = np.diff(np.log(self.tau))
        # Trapezoidal integration
        return float(np.sum(0.5 * (self.gamma[:-1] + self.gamma[1:]) * d_ln_tau))

    def summary(self) -> str:
        """Human-readable summary of the DRT analysis."""
        lines = [
            "DRT Analysis Summary",
            "=" * 40,
            f"  R_∞ (ohmic):          {self.r_inf:.4e}",
            f"  Total polarisation R:  {self.total_resistance():.4e}",
            f"  Peaks detected:        {self.n_peaks}",
            f"  Diffusion tail:        {'Yes' if self.has_diffusion_tail else 'No'}",
            f"  Inductive behavior:    {'Yes' if self.has_inductive else 'No'}",
            f"  Regularization λ:      {self.regularization_param:.2e}",
            f"  Fit quality (median):  {np.median(self.residuals):.2%}",
        ]
        if self.n_peaks > 0:
            lines.append("  Peak details:")
            for i in range(self.n_peaks):
                freq_peak = 1.0 / (2.0 * np.pi * self.peak_taus[i])
                lines.append(
                    f"    Peak {i + 1}: τ = {self.peak_taus[i]:.2e} s "
                    f"(f ≈ {freq_peak:.1f} Hz), "
                    f"γ = {self.peak_heights[i]:.4e}, "
                    f"ΔR ≈ {self.peak_resistances[i]:.4e}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DRTAnalyzer:
    """Tikhonov-regularised DRT deconvolution.

    Parameters
    ----------
    n_basis : int
        Number of basis functions (RBFs) for the discretisation.
    rbf_shape : float
        Shape parameter for Gaussian RBFs (controls peak width).
    lambda_method : str
        Method for choosing the regularisation parameter:
        ``"lcurve"`` (L-curve curvature), ``"gcv"`` (generalised
        cross-validation), or ``"fixed"`` (use ``lambda_fixed``).
    lambda_fixed : float
        Fixed regularisation parameter (only used if ``lambda_method="fixed"``).
    n_lambda_candidates : int
        Number of λ candidates to evaluate for lcurve/gcv.
    min_peak_prominence : float
        Minimum relative prominence (fraction of max γ) for peak detection.
    diffusion_tail_slope : float
        Minimum slope in log-log space to classify as a diffusion tail.
    """

    n_basis: int = 120
    rbf_shape: float = 0.5
    lambda_method: str = "lcurve"
    lambda_fixed: float = 1e-3
    n_lambda_candidates: int = 50
    min_peak_prominence: float = 0.02
    diffusion_tail_slope: float = 0.3

    def __post_init__(self) -> None:
        if self.n_basis < 5:
            raise ValueError("n_basis must be at least 5")
        if self.rbf_shape <= 0:
            raise ValueError("rbf_shape must be positive")
        if self.lambda_method not in ("lcurve", "gcv", "fixed"):
            raise ValueError("lambda_method must be 'lcurve', 'gcv', or 'fixed'")
        if self.lambda_fixed <= 0:
            raise ValueError("lambda_fixed must be positive")

    def analyze(self, dataset: EISDataset) -> DRTResult:
        """Run DRT analysis on an EIS dataset.

        Parameters
        ----------
        dataset : EISDataset
            Impedance dataset with ``freq_hz`` and ``z_obs``.

        Returns
        -------
        DRTResult
            Full DRT analysis result with peak detection.
        """
        freq = np.asarray(dataset.freq_hz, dtype=float)
        z_obs = np.asarray(dataset.z_obs, dtype=complex)
        n_freq = freq.size

        if n_freq < 5:
            raise ValueError("Need at least 5 frequency points for DRT")

        omega = 2.0 * np.pi * freq

        # --- Detect inductive behavior before DRT ---
        has_inductive = bool(np.any(z_obs.imag > 0))

        # Use only non-inductive points for DRT
        non_inductive = z_obs.imag <= 0
        if np.sum(non_inductive) < 5:
            # Not enough non-inductive points; use all
            non_inductive = np.ones(n_freq, dtype=bool)

        freq_use = freq[non_inductive]
        z_use = z_obs[non_inductive]
        omega_use = 2.0 * np.pi * freq_use

        # --- Set up τ grid ---
        tau_min = 1.0 / (2.0 * np.pi * freq_use.max()) / 10.0
        tau_max = 1.0 / (2.0 * np.pi * freq_use.min()) * 10.0
        ln_tau = np.linspace(np.log(tau_min), np.log(tau_max), self.n_basis)
        tau = np.exp(ln_tau)

        # --- Build design matrix ---
        # Z_model(ω) = R_∞ + Σ_k  γ_k · Δ(ln τ) / (1 + jωτ_k)
        # Discretise with uniform Δ(ln τ)
        d_ln_tau = ln_tau[1] - ln_tau[0] if self.n_basis > 1 else 1.0
        n_use = freq_use.size
        A_complex = np.zeros((n_use, self.n_basis), dtype=complex)
        for k in range(self.n_basis):
            A_complex[:, k] = d_ln_tau / (1.0 + 1j * omega_use * tau[k])

        # Add R_∞ column (constant)
        A_full = np.zeros((n_use, self.n_basis + 1), dtype=complex)
        A_full[:, 0] = 1.0  # R_∞
        A_full[:, 1:] = A_complex

        # Stack real and imaginary parts for real-valued least squares
        A_stacked = np.vstack([A_full.real, A_full.imag])
        b_stacked = np.concatenate([z_use.real, z_use.imag])

        # --- Second-order regularisation matrix (smoothness) ---
        n_unknowns = self.n_basis + 1
        L = np.zeros((max(self.n_basis - 2, 1), n_unknowns))
        for i in range(min(self.n_basis - 2, L.shape[0])):
            # Second difference on γ coefficients (skip R_∞)
            L[i, i + 1] = 1.0
            L[i, i + 2] = -2.0
            L[i, i + 3] = 1.0

        # --- Find optimal λ ---
        if self.lambda_method == "fixed":
            lambda_opt = self.lambda_fixed
        else:
            lambda_opt = self._find_optimal_lambda(A_stacked, b_stacked, L)

        # --- Solve with Tikhonov + non-negativity (iterative) ---
        gamma_full = self._solve_tikhonov_nnls(
            A_stacked, b_stacked, L, lambda_opt
        )

        r_inf = gamma_full[0]
        gamma = gamma_full[1:]

        # --- Reconstruct impedance on the FULL frequency grid ---
        A_recon_full = np.zeros((n_freq, self.n_basis + 1), dtype=complex)
        A_recon_full[:, 0] = 1.0
        for k in range(self.n_basis):
            A_recon_full[:, k + 1] = d_ln_tau / (1.0 + 1j * omega * tau[k])
        z_fit = A_recon_full @ gamma_full

        # Per-point residuals
        z_mag = np.maximum(np.abs(z_obs), 1e-30)
        residuals = np.abs(z_fit - z_obs) / z_mag

        # --- Peak detection ---
        peaks, peak_props = _signal.find_peaks(
            gamma,
            prominence=self.min_peak_prominence * np.max(gamma) if np.max(gamma) > 0 else 0,
        )

        n_peaks = len(peaks)
        peak_taus = tau[peaks] if n_peaks > 0 else np.array([])
        peak_heights = gamma[peaks] if n_peaks > 0 else np.array([])

        # Estimate peak resistances (trapezoidal around each peak)
        peak_resistances = np.zeros(n_peaks)
        for i, pk in enumerate(peaks):
            # Find half-height boundaries
            half_h = peak_heights[i] / 2.0
            left = pk
            while left > 0 and gamma[left] > half_h:
                left -= 1
            right = pk
            while right < self.n_basis - 1 and gamma[right] > half_h:
                right += 1
            # Integrate
            if right > left:
                peak_resistances[i] = float(
                    np.trapezoid(gamma[left : right + 1], np.log(tau[left : right + 1]))
                )

        # --- Diffusion tail detection ---
        # Check if γ rises at low frequencies (high τ) without forming a peak
        has_diffusion_tail = False
        if self.n_basis > 10:
            tail_region = gamma[-self.n_basis // 5 :]
            if tail_region.size >= 3 and tail_region[-1] > 0:
                # Check if the tail is monotonically rising
                diffs = np.diff(tail_region)
                if np.sum(diffs > 0) > len(diffs) * 0.6:
                    # Rising tail — likely diffusion
                    has_diffusion_tail = True
                # Also check slope in log-log
                if tail_region[-1] > 0 and tail_region[0] > 0:
                    slope = (np.log(tail_region[-1]) - np.log(tail_region[0])) / (
                        np.log(tau[-1]) - np.log(tau[-self.n_basis // 5])
                    )
                    if slope > self.diffusion_tail_slope:
                        has_diffusion_tail = True

        return DRTResult(
            tau=tau,
            gamma=gamma,
            z_fit=z_fit,
            residuals=residuals,
            r_inf=float(r_inf),
            n_peaks=n_peaks,
            peak_taus=peak_taus,
            peak_heights=peak_heights,
            peak_resistances=peak_resistances,
            regularization_param=float(lambda_opt),
            has_diffusion_tail=has_diffusion_tail,
            has_inductive=has_inductive,
        )

    # ----- Private helpers -----

    def _find_optimal_lambda(
        self,
        A: FloatArray,
        b: FloatArray,
        L: FloatArray,
    ) -> float:
        """Find optimal λ via L-curve or GCV."""
        lambdas = np.logspace(-8, 2, self.n_lambda_candidates)

        if self.lambda_method == "gcv":
            return self._gcv_lambda(A, b, L, lambdas)
        else:
            return self._lcurve_lambda(A, b, L, lambdas)

    def _lcurve_lambda(
        self,
        A: FloatArray,
        b: FloatArray,
        L: FloatArray,
        lambdas: FloatArray,
    ) -> float:
        """L-curve method: find the corner of maximum curvature."""
        log_residuals = np.zeros(len(lambdas))
        log_solutions = np.zeros(len(lambdas))

        for i, lam in enumerate(lambdas):
            x = self._solve_tikhonov_lstsq(A, b, L, lam)
            res = b - A @ x
            log_residuals[i] = np.log(np.linalg.norm(res) + 1e-30)
            log_solutions[i] = np.log(np.linalg.norm(L @ x) + 1e-30)

        # Curvature of the L-curve (parametric curve in log-log space)
        if len(lambdas) < 3:
            return lambdas[len(lambdas) // 2]

        # Numerical curvature: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        dx = np.gradient(log_residuals)
        dy = np.gradient(log_solutions)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-30) ** 1.5

        # Find maximum curvature (skip endpoints)
        inner = curvature[1:-1]
        if inner.size == 0:
            return lambdas[len(lambdas) // 2]
        best_idx = np.argmax(inner) + 1
        return float(lambdas[best_idx])

    def _gcv_lambda(
        self,
        A: FloatArray,
        b: FloatArray,
        L: FloatArray,
        lambdas: FloatArray,
    ) -> float:
        """Generalised Cross-Validation for λ selection."""
        n = A.shape[0]
        gcv_scores = np.zeros(len(lambdas))

        for i, lam in enumerate(lambdas):
            A_reg = np.vstack([A, lam * L])
            b_reg = np.concatenate([b, np.zeros(L.shape[0])])
            # Hat matrix: H = A (A^T A + λ²L^TL)^{-1} A^T
            # GCV = ‖(I-H)b‖² / (trace(I-H)/n)²
            AtA = A.T @ A + lam**2 * L.T @ L
            try:
                AtA_inv = np.linalg.inv(AtA + 1e-14 * np.eye(AtA.shape[0]))
            except np.linalg.LinAlgError:
                gcv_scores[i] = np.inf
                continue
            H = A @ AtA_inv @ A.T
            residuals = (np.eye(n) - H) @ b
            trace_I_H = n - np.trace(H)
            if trace_I_H < 1e-10:
                gcv_scores[i] = np.inf
            else:
                gcv_scores[i] = (np.linalg.norm(residuals) ** 2) / (
                    trace_I_H / n
                ) ** 2

        best_idx = np.argmin(gcv_scores)
        return float(lambdas[best_idx])

    def _solve_tikhonov_lstsq(
        self,
        A: FloatArray,
        b: FloatArray,
        L: FloatArray,
        lam: float,
    ) -> FloatArray:
        """Plain Tikhonov (no non-negativity)."""
        A_reg = np.vstack([A, lam * L])
        b_reg = np.concatenate([b, np.zeros(L.shape[0])])
        x, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
        return x

    def _solve_tikhonov_nnls(
        self,
        A: FloatArray,
        b: FloatArray,
        L: FloatArray,
        lam: float,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> FloatArray:
        """Tikhonov with non-negativity constraint on γ (not R_∞).

        Uses iterative alternating optimisation: fix R_∞ → NNLS for γ,
        fix γ → closed-form for R_∞, repeat until convergence or
        *max_iter* is reached.
        """
        from scipy.optimize import nnls

        A_reg = np.vstack([A, lam * L])
        b_reg = np.concatenate([b, np.zeros(L.shape[0])])

        # Split: col 0 is R_∞ (unconstrained), cols 1: are γ (non-negative)
        A_rinf = A_reg[:, 0]
        A_gamma = A_reg[:, 1:]
        rinf_dot = float(np.dot(A_rinf, A_rinf)) + 1e-30

        # Estimate R_∞ from the highest-frequency impedance real part
        n_freq = A.shape[0] // 2  # stacked real + imag
        r_inf = float(b[:n_freq].mean()) if n_freq > 0 else 0.0

        gamma = np.zeros(A_gamma.shape[1], dtype=float)

        prev_cost = np.inf
        for _iteration in range(max_iter):
            # Step 1: Fix R_∞, solve NNLS for γ
            b_adj = b_reg - A_rinf * r_inf
            try:
                gamma, _ = nnls(A_gamma, b_adj)
            except Exception:
                gamma_raw, _, _, _ = np.linalg.lstsq(A_gamma, b_adj, rcond=None)
                gamma = np.maximum(gamma_raw, 0.0)

            # Step 2: Fix γ, solve for R_∞ (closed-form projection)
            residual = b_reg - A_gamma @ gamma
            r_inf = float(np.dot(A_rinf, residual) / rinf_dot)

            # Convergence check
            cost = float(np.sum((b_reg - A_rinf * r_inf - A_gamma @ gamma) ** 2))
            if abs(prev_cost - cost) < tol * max(abs(prev_cost), 1.0):
                break
            prev_cost = cost

        result = np.zeros(A_reg.shape[1])
        result[0] = r_inf
        result[1:] = gamma
        return result
