# learning AI at www.haotianblog.com
"""Composable impedance elements for the Generation-2 ECM library.

Every element is a plain vectorized function ``omega -> complex array`` so
that higher-order circuits (transmission lines, fractional diffusion,
SEI + porous-electrode stacks) can be assembled without re-deriving the
underlying math in every model class.

Conventions
-----------
* ``omega`` is the angular frequency array (rad/s), strictly positive.
* Arc elements use the **ZARC / Cole-Cole parameterization** ``(R, tau,
  alpha)`` with ``Z = R / (1 + (j w tau)^alpha)``.  This is algebraically
  identical to the classic ``R || CPE`` element with ``Q = tau^alpha / R``
  but removes the notorious Q-alpha collinearity: ``tau`` directly locates
  the arc apex, so the Jacobian columns of ``tau`` and ``alpha`` are far
  closer to orthogonal than those of ``Q`` and ``alpha``.
* Diffusion time constants use ``tau_d = L^2 / D``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_EPS = 1e-30


def z_resistor(omega: FloatArray, r: float) -> ComplexArray:
    """Ohmic resistance: Z = R."""
    return np.full_like(omega, r, dtype=complex)


def z_inductor(omega: FloatArray, l: float) -> ComplexArray:
    """Ideal inductor: Z = jωL."""
    return 1j * omega * l


def z_capacitor(omega: FloatArray, c: float) -> ComplexArray:
    """Ideal capacitor: Z = 1 / (jωC)."""
    return 1.0 / (1j * omega * c + _EPS)


def z_cpe(omega: FloatArray, q: float, alpha: float) -> ComplexArray:
    """Constant phase element: Z = 1 / (Q (jω)^α)."""
    return 1.0 / (q * (1j * omega) ** alpha + _EPS)


def z_zarc(omega: FloatArray, r: float, tau: float, alpha: float) -> ComplexArray:
    """ZARC (Cole-Cole) element: Z = R / (1 + (jωτ)^α).

    Equivalent to R‖CPE with Q = τ^α / R.  Preferred in Generation 2
    because (R, τ, α) is a near-orthogonal parameterization.
    """
    return r / (1.0 + (1j * omega * tau) ** alpha)


def z_warburg_semi_infinite(omega: FloatArray, aw: float) -> ComplexArray:
    """Semi-infinite Warburg: Z = A_W / √(jω)."""
    return aw / np.sqrt(1j * omega)


def z_warburg_short(omega: FloatArray, rd: float, tau_d: float) -> ComplexArray:
    """Finite-length Warburg (FLW, transmissive boundary).

    Z = R_d · tanh(√(jωτ_d)) / √(jωτ_d)

    DC limit: Z → R_d.  Appropriate when the diffusion layer terminates
    in a reservoir of fixed concentration (e.g. electrolyte bulk).
    """
    s = np.sqrt(1j * omega * tau_d)
    return rd * np.tanh(s) / (s + _EPS)


def z_warburg_open(omega: FloatArray, rd: float, tau_d: float) -> ComplexArray:
    """Finite-space Warburg (FSW, reflective boundary).

    Z = R_d · coth(√(jωτ_d)) / √(jωτ_d)

    DC limit: capacitive divergence (blocked diffusion, e.g. intercalation
    into a bounded host particle).
    """
    s = np.sqrt(1j * omega * tau_d)
    return rd * _coth(s) / (s + _EPS)


def z_warburg_short_fractional(
    omega: FloatArray, rd: float, tau_d: float, gamma: float
) -> ComplexArray:
    """Generalized (fractional) finite-length Warburg.

    Z = R_d · tanh((jωτ_d)^γ) / (jωτ_d)^γ

    ``gamma = 0.5`` recovers the classic FLW; ``gamma < 0.5`` models
    anomalous (sub-diffusive) transport in disordered media.
    """
    s = (1j * omega * tau_d) ** gamma
    return rd * np.tanh(s) / (s + _EPS)


def z_gerischer(omega: FloatArray, rg: float, tau_g: float) -> ComplexArray:
    """Gerischer element: Z = R_G / √(1 + jωτ_G).

    Arises when diffusion is coupled to a preceding homogeneous chemical
    reaction (e.g. oxygen surface exchange, coupled side reactions).
    """
    return rg / np.sqrt(1.0 + 1j * omega * tau_g)


def z_tlm(omega: FloatArray, r_ion: float, z_interfacial: ComplexArray) -> ComplexArray:
    """Single-channel transmission line model (porous electrode).

    Z = √(R_ion · ζ) · coth(√(R_ion / ζ))

    where ``R_ion`` is the total ionic pore resistance and ``ζ`` the
    lumped interfacial impedance of the whole electrode (de Levie, with
    electronic conductivity ≫ ionic conductivity).

    High-frequency limit: Z → √(R_ion·ζ) (45° branch); low-frequency
    limit: Z → R_ion/3 + ζ.
    """
    ratio = r_ion / (z_interfacial + _EPS)
    s = np.sqrt(ratio.astype(complex))
    return np.sqrt(r_ion * z_interfacial.astype(complex)) * _coth(s)


def z_tlm_blocking(
    omega: FloatArray, r_ion: float, q: float, alpha: float
) -> ComplexArray:
    """Blocking-electrode TLM: interfacial impedance is a pure CPE."""
    return z_tlm(omega, r_ion, z_cpe(omega, q, alpha))


def z_tlm_faradaic(
    omega: FloatArray, r_ion: float, rct: float, q: float, alpha: float
) -> ComplexArray:
    """Faradaic TLM: interfacial impedance is R_ct ‖ CPE (charge transfer
    distributed along the pore)."""
    y_cpe = q * (1j * omega) ** alpha
    z_int = rct / (1.0 + rct * y_cpe)
    return z_tlm(omega, r_ion, z_int)


def zarc_equivalent_q(r: float, tau: float, alpha: float) -> float:
    """Convert a ZARC (R, τ, α) triple to the equivalent CPE Q value."""
    return float(tau**alpha / max(r, 1e-30))


def zarc_from_rq(r: float, q: float, alpha: float) -> float:
    """Return τ of the ZARC equivalent to an R‖CPE with (R, Q, α)."""
    return float((r * q) ** (1.0 / max(alpha, 1e-3)))


def _coth(s: ComplexArray) -> ComplexArray:
    """Numerically safe complex hyperbolic cotangent."""
    t = np.tanh(s)
    small = np.abs(t) < _EPS
    if np.any(small):
        t = np.where(small, _EPS, t)
    return 1.0 / t
