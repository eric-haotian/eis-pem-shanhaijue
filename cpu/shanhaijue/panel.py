# learning AI at www.haotianblog.com
"""Forward model, parameter space, and panel generation.

Everything downstream (estimator arms, router, certifier) depends only on this module,
so swapping the forward model or the acquisition grid is a single-file change.

The 48-parameter stacked frequency-domain impedance model is the DFN-type model of the
accompanying paper; the physics prior supplies the per-coordinate widths that define the
prior-strength axis along which the estimator ensemble is built.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

from eispem.seis_model import StackedSEISModel, all_seis_parameter_specs  # noqa: E402

from eispem.seis_pipeline import SEISPhysicsPriorBuilder  # noqa: E402

# ----------------------------------------------------------------------------------
# parameter space
# ----------------------------------------------------------------------------------
SPECS = tuple(all_seis_parameter_specs())
NAMES = [s.name for s in SPECS]
IDX = {n: i for i, n in enumerate(NAMES)}
NDIM = len(SPECS)

NOM = np.array([s.initial_value for s in SPECS])
LOWER = np.array([s.bounds[0] for s in SPECS])
UPPER = np.array([s.bounds[1] for s in SPECS])
LOG = np.array([s.log_transform for s in SPECS])

_PS = SEISPhysicsPriorBuilder().build(SPECS)
SIGMA = _PS.sigma_search_vector(SPECS)          # prior widths in search coordinates
PMEAN = _PS.mean_physical_vector(SPECS)         # prior centre, physical units
PRIOR_W = 0.005 / SIGMA                         # base prior weight (lambda = 1)

F_CONST = 96487.0
NOISE = 0.005                                   # proportional observation noise


def to_search(x):
    """Physical vector -> search coordinates (log10 for scale parameters)."""
    return np.where(LOG, np.log10(np.maximum(x, 1e-300)), x)


def to_physical(s):
    """Search coordinates -> physical vector."""
    return np.where(LOG, 10.0 ** s, s)


SLO = to_search(LOWER)
SHI = to_search(UPPER)
SPMEAN = to_search(PMEAN)
SNOM = to_search(NOM)

# ----------------------------------------------------------------------------------
# acquisition grid (baseline 5 SOC x 5 temperatures x 120 frequencies)
# ----------------------------------------------------------------------------------
TEMPERATURES = (278.15, 288.15, 298.15, 308.15, 318.15)
SOCS = (0.10, 0.30, 0.50, 0.70, 0.90)
CONDITIONS = [(t, s) for t in TEMPERATURES for s in SOCS]
FREQ = np.logspace(-3, 5, 120)

MODEL = StackedSEISModel(conditions=tuple(CONDITIONS), parameter_names=tuple(NAMES))
FSTACK = np.tile(FREQ, len(CONDITIONS))
NFREQ = FSTACK.size
NDATA = 2 * NFREQ

# held-out frequency mask: drop every fourth frequency inside each condition block
_hold = np.zeros(FREQ.size, bool)
_hold[3::4] = True
HOLDOUT = np.tile(_hold, len(CONDITIONS))
KEEP = ~HOLDOUT


_GRID_LISTENERS = []


def on_grid_change(fn):
    """Register a callback invoked after every `configure_grid`.

    Backends that cache grid-derived arrays (the JAX distribution pre-stages the
    frequency stack on device) register here so a grid change can never leave a stale
    copy behind.
    """
    _GRID_LISTENERS.append(fn)
    return fn


def configure_grid(temperatures=None, socs=None, frequencies=None, conditions=None):
    """Rebuild the acquisition grid (conditions x frequencies) in place.

    Call this BEFORE fitting anything if your spectra are not on the default
    5 SOC x 5 temperature x 120 frequency grid. Pass `conditions` directly as a list of
    (temperature_K, soc) pairs when the measured conditions are not a full cross product
    of temperatures and SOCs — a single spectrum, a temperature sweep at one SOC, and an
    arbitrary scattered set are all admissible.

    Identification runs on any grid. A frozen certifier, however, is only valid on the
    grid it was calibrated on: its calibrated probabilities and false-claim budget were
    fitted against that information content. Changing the grid means re-freezing.
    """
    global TEMPERATURES, SOCS, CONDITIONS, FREQ, MODEL, FSTACK, NFREQ, NDATA, HOLDOUT, KEEP
    if frequencies is not None:
        FREQ = np.asarray(frequencies, float)
    if conditions is not None:
        CONDITIONS = [(float(t), float(s)) for t, s in conditions]
        TEMPERATURES = tuple(sorted({c[0] for c in CONDITIONS}))
        SOCS = tuple(sorted({c[1] for c in CONDITIONS}))
    else:
        if temperatures is not None:
            TEMPERATURES = tuple(temperatures)
        if socs is not None:
            SOCS = tuple(socs)
        CONDITIONS = [(t, s) for t in TEMPERATURES for s in SOCS]
    MODEL = StackedSEISModel(conditions=tuple(CONDITIONS), parameter_names=tuple(NAMES))
    FSTACK = np.tile(FREQ, len(CONDITIONS))
    NFREQ = FSTACK.size
    NDATA = 2 * NFREQ
    h = np.zeros(FREQ.size, bool)
    h[3::4] = True
    HOLDOUT = np.tile(h, len(CONDITIONS))
    KEEP = ~HOLDOUT
    for fn in _GRID_LISTENERS:
        fn()
    return dict(conditions=len(CONDITIONS), frequencies=int(FREQ.size), points=int(NFREQ))

# ----------------------------------------------------------------------------------
# physical claim coordinates (products/quotients the spectra actually constrain)
# ----------------------------------------------------------------------------------
def _stoich(p, pre):
    return p[f"s0_{pre}"] + 0.5 * (p[f"s100_{pre}"] - p[f"s0_{pre}"])


def _i0(pre):
    def f(p):
        a = p[f"alpha_a_{pre}"]
        csm = p[f"cs_max_{pre}"]
        cs0 = csm * _stoich(p, pre)
        return (F_CONST * p["ce_0"] ** a * max(csm - cs0, 1e-300) ** a
                * max(cs0, 1e-300) ** (1.0 - a) * p[f"k_{pre}_0"])
    return f


def _tau_d(pre):
    def f(p):
        return p[f"rs_{pre}"] ** 2 / p[f"Ds_{pre}_0"]
    return f


def _r_sei(pre):
    def f(p):
        return p[f"rou_sei_{pre}_0"] * p[f"rs_{pre}"] / 51.0
    return f


def _c_sei(pre):
    def f(p):
        return p[f"epse_sei_{pre}"] * 51.0 / p[f"rs_{pre}"]
    return f


def _kappa_eff(p):
    return p["kappa_0"] * p["epse_neg"] ** p["brug_neg"]


COMBO = {
    "k_neg_0": ("i0_neg", _i0("neg")), "k_pos_0": ("i0_pos", _i0("pos")),
    "Ds_neg_0": ("tau_d_neg", _tau_d("neg")), "Ds_pos_0": ("tau_d_pos", _tau_d("pos")),
    "rou_sei_neg_0": ("R_sei_neg", _r_sei("neg")), "rou_sei_pos_0": ("R_sei_pos", _r_sei("pos")),
    "epse_sei_neg": ("C_sei_neg", _c_sei("neg")), "epse_sei_pos": ("C_sei_pos", _c_sei("pos")),
    "kappa_0": ("kappa_eff_neg", _kappa_eff),
}
COMBO_IDX = {IDX[b]: (nm, fn) for b, (nm, fn) in COMBO.items()}
COORD_NAME = {"raw": list(NAMES), "phys": [COMBO.get(n, (n,))[0] for n in NAMES]}


def coord_vector(basis: str, s: np.ndarray) -> np.ndarray:
    """Search coordinates -> claim coordinates in the requested basis."""
    th = to_physical(s)
    if basis == "raw":
        return th
    p = dict(zip(NAMES, th))
    out = th.copy()
    for i, (_nm, fn) in COMBO_IDX.items():
        out[i] = fn(p)
    return out


def family_of(name: str) -> str:
    """Physical family of a claim coordinate (used for per-family reporting)."""
    if name.startswith("tau_d") or name.startswith("Ds"):
        return "diffusion"
    if name.startswith("Ea_"):
        return "activation_energy"
    if name.startswith("i0") or name.startswith("k_") or name.startswith("alpha_a"):
        return "kinetics"
    if name.startswith("epse") or name.startswith("epsf") or name.startswith("brug"):
        return "porosity_transport"
    if name.startswith("s0_") or name.startswith("s100_") or name.startswith("cs_max"):
        return "stoichiometry"
    if "sei" in name.lower():
        return "sei"
    if name.startswith("Cdl") or name.startswith("alpha_dl"):
        return "double_layer"
    if name in ("R_contact", "L_ind"):
        return "series_parasitic"
    if name.startswith("rs_") or name.startswith("L_") or name.startswith("sigma"):
        return "geometry"
    return "electrolyte"


# ----------------------------------------------------------------------------------
# synthetic panel generation (for calibration, freezing, and validation)
# ----------------------------------------------------------------------------------
def _physically_valid(v: np.ndarray) -> bool:
    p = dict(zip(NAMES, v))
    return (p["epse_neg"] + p["epsf_neg"] < 0.99 and p["epse_pos"] + p["epsf_pos"] < 0.99
            and p["s0_neg"] < p["s100_neg"] and p["s100_pos"] < p["s0_pos"])


def draw_truth(index: int, seed_base: int, offset_decades=(1.10, 1.30)) -> np.ndarray:
    """Off-nominal generating truth: each parameter multiplied by 10^(+-u)."""
    rng = np.random.default_rng(seed_base + index)
    lo, hi = math.log10(offset_decades[0]), math.log10(offset_decades[1])
    for _ in range(20000):
        off = rng.choice([-1.0, 1.0], size=NDIM) * rng.uniform(lo, hi, size=NDIM)
        v = NOM * 10.0 ** off
        if np.any(v <= LOWER) or np.any(v >= UPPER):
            continue
        if _physically_valid(v):
            return v
    raise RuntimeError("truth draw failed after 20000 attempts")


def simulate_observation(truth: np.ndarray, seed: int) -> np.ndarray:
    """Stacked complex spectrum with proportional Gaussian noise."""
    rng = np.random.default_rng(seed)
    out = []
    for (T, soc) in CONDITIONS:
        m = StackedSEISModel(conditions=((T, soc),), parameter_names=tuple(NAMES))
        z = m.simulate(FREQ, truth)
        out.append(z + NOISE * np.abs(z) * (rng.normal(size=z.size)
                                            + 1j * rng.normal(size=z.size)))
    return np.concatenate(out)


def simulate_clean(theta: np.ndarray) -> np.ndarray:
    """Noise-free stacked spectrum at a physical parameter vector."""
    return MODEL.simulate(FSTACK, theta)
