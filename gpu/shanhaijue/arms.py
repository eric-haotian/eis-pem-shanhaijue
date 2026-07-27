# learning AI at www.haotianblog.com
"""The estimator ensemble (ShanHaiJue) — GPU backend.

Same objective and same ensemble semantics as the CPU distribution:

    s_hat(lambda) = argmin_s  sum_k |Z_model(w_k; s) - Z_k|^2 / |Z_k|^2
                              + lambda^2 || W_p (s - s_0) ||^2

but every fit is solved by a batched Levenberg-Marquardt iteration compiled once and
vmapped over (starts x cells). The step is taken from the SVD of the exact forward-mode
AD Jacobian,

    delta = -V diag( sigma_i / (sigma_i^2 + mu) ) U^T r,      s <- Pi_[l,u]( s + delta ),

damping the singular values DIRECTLY rather than forming J^T J, whose condition number
would be the square of one that already reaches ~1e8. This is what keeps the flat
directions of the sloppy valley numerically controlled at scale.

Throughput is ~100x the single-fit CPU optimizer, which is what makes the eight-arm
ensemble (23 bounded fits per cell) practical: under one minute per cell.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from . import panel as P  # noqa: E402

LADDER_8 = (3.0, 1.0, 0.6, 0.3, 0.1, 0.03)
LADDER_6 = (3.0, 1.0, 0.3, 0.1)
LADDER_3 = (1.0,)

N_STARTS = 3
JITTER = 0.15
WIDE_JITTER = 0.35
N_WIDE = 3
LM_STEPS = 100
CHUNK = 10                     # fits per compiled batch (tuned for 8 GB VRAM)

_jFST = jnp.asarray(P.FSTACK)
_jLOG = jnp.asarray(P.LOG)
_jSLO = jnp.asarray(P.SLO)
_jSHI = jnp.asarray(P.SHI)
_jW = jnp.asarray(P.PRIOR_W)
_jCEN = jnp.asarray(P.SPMEAN)


@P.on_grid_change
def refresh_grid():
    """Re-stage the grid-derived device arrays after `panel.configure_grid`.

    The jitted kernels close over the frequency stack as a traced constant, so the
    compilation cache must be dropped as well; otherwise a grid change would be silently
    ignored and every fit would keep using the previous frequencies.
    """
    global _jFST
    _jFST = jnp.asarray(P.FSTACK)
    jax.clear_caches()


@dataclass
class ArmResult:
    name: str
    estimate_search: np.ndarray
    starts_search: list = field(default_factory=list)
    cost: float = float("nan")


def _to_physical_j(s):
    return jnp.where(_jLOG, 10.0 ** s, s)


def _residual_j(s, zr, zi, zm, wdata, prior_scale):
    """Weighted complex residual stacked with the prior rows.

    `wdata` is a per-frequency data weight (zero-weighted rows implement the held-out
    frequency mask); `prior_scale` is lambda (0 gives the data-only arm).
    """
    z = P.MODEL.simulate_jax(_jFST, _to_physical_j(s))
    return jnp.concatenate([wdata * (z.real - zr) / zm,
                            wdata * (z.imag - zi) / zm,
                            prior_scale * _jW * (s - _jCEN)])


def _cost_j(s, zr, zi, zm, wdata, prior_scale):
    r = _residual_j(s, zr, zi, zm, wdata, prior_scale)
    return 0.5 * jnp.sum(r * r)


@jax.jit
def _lm_batch(S0, ZR, ZI, ZM, WD, PS):
    """Batched bounded Levenberg-Marquardt with SVD-damped steps."""
    def one(s0, zr, zi, zm, wd, ps):
        def body(carry, _):
            s, lam, c = carry
            r = _residual_j(s, zr, zi, zm, wd, ps)
            J = jax.jacfwd(lambda x: _residual_j(x, zr, zi, zm, wd, ps))(s)
            U, sv, Vt = jnp.linalg.svd(J, full_matrices=False)
            step = -(Vt.T @ (sv / (sv * sv + lam * lam) * (U.T @ r)))
            s_new = jnp.clip(s + step, _jSLO, _jSHI)
            c_new = _cost_j(s_new, zr, zi, zm, wd, ps)
            ok = c_new < c
            return (jnp.where(ok, s_new, s),
                    jnp.where(ok, jnp.maximum(lam / 3.0, 1e-7), jnp.minimum(lam * 5.0, 1e7)),
                    jnp.where(ok, c_new, c)), c
        c0 = _cost_j(s0, zr, zi, zm, wd, ps)
        (s, lam, c), _ = jax.lax.scan(body, (s0, 1.0, c0), None, length=LM_STEPS)
        return s, c
    return jax.vmap(one)(S0, ZR, ZI, ZM, WD, PS)


@jax.jit
def _jacobian_batch(S, ZR, ZI, ZM):
    ones = jnp.ones(P.NFREQ)
    def one(s, zr, zi, zm):
        return jax.jacfwd(lambda x: _residual_j(x, zr, zi, zm, ones, 1.0))(s)
    return jax.vmap(one)(S, ZR, ZI, ZM)


def run_batch(S0, ZR, ZI, ZM, WD, PS, chunk: int = CHUNK, tag: str = "", progress=False):
    """Chunked driver so arbitrarily many fits fit in VRAM."""
    out_s, out_c = [], []
    for k in range(0, len(S0), chunk):
        sl = slice(k, k + chunk)
        s, c = _lm_batch(jnp.asarray(S0[sl]), jnp.asarray(ZR[sl]), jnp.asarray(ZI[sl]),
                         jnp.asarray(ZM[sl]), jnp.asarray(WD[sl]), jnp.asarray(PS[sl]))
        out_s.append(np.array(s))
        out_c.append(np.array(c))
        if progress:
            print(f"  [{tag}] {min(k + chunk, len(S0))}/{len(S0)} fits", flush=True)
    return np.concatenate(out_s), np.concatenate(out_c)


def jacobian_at(s_search: np.ndarray, z_obs: np.ndarray) -> np.ndarray:
    """Exact AD Jacobian of the stacked residual at one point."""
    zm = np.maximum(np.abs(z_obs), 1e-30)
    J = _jacobian_batch(jnp.asarray(s_search[None]), jnp.asarray(z_obs.real[None]),
                        jnp.asarray(z_obs.imag[None]), jnp.asarray(zm[None]))
    return np.array(J)[0]


def _starts(seed, n, scale):
    rng = np.random.default_rng(seed)
    return [np.clip(P.SPMEAN + rng.normal(0, scale, P.NDIM), P.SLO, P.SHI)
            for _ in range(n)]


def fit_ensemble(z_obs, seed: int, ladder=LADDER_8, chunk: int = CHUNK):
    """Fit the whole ensemble for one cell in two batched passes.

    Pass 1: every lambda arm x N_STARTS starts, plus the wide multistart starts.
    Pass 2: the data-only refinement (warm-started) and the held-out-frequency refit.
    """
    ladder = tuple(ladder)
    zm = np.maximum(np.abs(z_obs), 1e-30)
    base_starts = [P.SPMEAN.copy()] + _starts(seed + 11, N_STARTS - 1, JITTER)
    wide_starts = _starts(seed + 11, N_WIDE, WIDE_JITTER)

    S0, PS, tags = [], [], []
    for lam in ladder:
        for si, s0 in enumerate(base_starts):
            S0.append(s0); PS.append(lam); tags.append((f"lam{lam:g}".replace(".", ""), si))
    for si, s0 in enumerate(wide_starts):
        S0.append(s0); PS.append(1.0); tags.append(("M1x", si))
    n = len(S0)
    S1, C1 = run_batch(np.array(S0), np.tile(z_obs.real, (n, 1)), np.tile(z_obs.imag, (n, 1)),
                       np.tile(zm, (n, 1)), np.ones((n, P.NFREQ)), np.array(PS), chunk)
    idx = {t: i for i, t in enumerate(tags)}

    arms = {}
    for lam in ladder:
        name = f"lam{lam:g}".replace(".", "")
        sl = [S1[idx[(name, si)]] for si in range(N_STARTS)]
        arms[name] = ArmResult(name, np.median(np.vstack(sl), axis=0), sl,
                               float(min(C1[idx[(name, si)]] for si in range(N_STARTS))))

    ref_name = "lam1" if "lam1" in arms else f"lam{ladder[0]:g}".replace(".", "")
    s_ref = arms[ref_name].starts_search[0]

    cands = [(C1[idx[(ref_name, 0)]], s_ref)]
    for si in range(N_WIDE):
        cands.append((C1[idx[("M1x", si)]], S1[idx[("M1x", si)]]))
    c1, s_m1 = min(cands, key=lambda t: t[0])
    arms["M1"] = ArmResult("M1", s_m1, [s_m1], float(c1))

    S0b = np.vstack([s_ref, P.SPMEAN.copy()])
    WDb = np.vstack([np.ones(P.NFREQ), P.KEEP.astype(float)])
    PSb = np.array([0.0, 1.0])                       # data-only, then held-out refit
    S2, C2 = run_batch(S0b, np.tile(z_obs.real, (2, 1)), np.tile(z_obs.imag, (2, 1)),
                       np.tile(zm, (2, 1)), WDb, PSb, chunk)
    arms["M4"] = ArmResult("M4", S2[0], [S2[0]], float(C2[0]))

    aux = dict(jacobian=jacobian_at(s_ref, z_obs), reference_start=s_ref,
               holdout_estimate=S2[1])
    return arms, aux
