# learning AI at www.haotianblog.com
"""Physics-native reparameterization for the SEIS model.

The SEIS forward calculation composes parameters through products and
power laws — exchange current density ``i0 = F k ce^a (cs_max-cs)^a
cs^(1-a)``, effective conductivity ``kappa_eff = kappa epse^brug``, SEI
film elements ``R_sei = rou_sei rs delta/(rs+delta)``, and the diffusion
time constant ``tau_d = rs^2 / Ds``.  In the raw coordinates the
stacked Jacobian columns of the factors are nearly parallel (measured
on the 3x3 grid in Arrhenius coordinates: rho(kappa_0, brug_neg) =
-0.997, rho(k_neg_0, alpha_a_neg) = +0.991, rho(k_neg_0, cs_max_neg) =
+0.970), so the Wald box around the posterior ridge explodes and the
factors fail the identifiability gate even though their *product* is
sharply determined.

Each transform below replaces one factor coordinate with the physical
product itself (same parameter name, new documented meaning), which is
a log-space shear.  In the shared interior of an exactly transformed
feasible domain, a bijective reparameterization cannot create local
information; it changes *which question* the gate asks, from "is k_0
determined?" to "is the exchange current density determined?", the
latter being what EIS actually measures.  The implementation uses static
rectangular search bounds evaluated at literature-default dependencies.
Those bounds are a numerical approximation to the coupled transformed
domain, so global posterior profiles involving a dependency parameter are
not claimed to be coordinate-invariant.

The publication diffusion coordinate is the physical time scale
``tau_d = rs^2 / Ds``.  The former exposed-development coordinate
``Ds*rs^2`` is intentionally unavailable: it had no defensible physical
claim semantics and cannot be selected by the publication candidate.

The specific-surface-area transform from the design brief
(``(epsf, rs) -> (a_s, rs)`` with ``a_s = 3 (1-epse-epsf)/rs`` at fixed
``epse`` and ``rs``) is deliberately NOT implemented: it is a monotone
rescale of the single coordinate ``epsf``, and both claim metrics
(information gain and relative CI in search space) are invariant under
single-coordinate monotone rescaling — it provably cannot change any
claim, so shipping it would only add dead complexity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .parameters import (
    ParameterSpec,
    decode_parameter_vector,
    encode_parameter_vector,
)
from .seis_model import all_seis_parameter_specs

from .priors import GaussianParameterPrior, GaussianSearchCovariance, PriorSet

FloatArray = NDArray[np.float64]

F_CONST = 96487.0  # C/mol, matches the Gen-1 parameter table


def _literature_defaults() -> dict[str, float]:
    return {s.name: float(s.initial_value) for s in all_seis_parameter_specs()}


_DEFAULTS = _literature_defaults()
_DEFAULT_SPECS = {s.name: s for s in all_seis_parameter_specs()}


def _stoich(prefix: str, values: Mapping[str, float], soc: float = 0.5) -> float:
    s0 = values.get(f"s0_{prefix}", _DEFAULTS[f"s0_{prefix}"])
    s100 = values.get(f"s100_{prefix}", _DEFAULTS[f"s100_{prefix}"])
    return s0 + soc * (s100 - s0)


def _i0_factor(prefix: str) -> Callable[[Mapping[str, float]], float]:
    """i0 = factor * k at reference SOC=0.5, ce=ce_0 (temperature rides
    on k itself, so the coordinate follows k's Arrhenius convention)."""

    def factor(values: Mapping[str, float]) -> float:
        alpha = values.get(f"alpha_a_{prefix}", _DEFAULTS[f"alpha_a_{prefix}"])
        cs_max = values.get(f"cs_max_{prefix}", _DEFAULTS[f"cs_max_{prefix}"])
        ce0 = values.get("ce_0", _DEFAULTS["ce_0"])
        st = _stoich(prefix, values)
        cs0 = cs_max * st
        return float(
            F_CONST * ce0**alpha * (cs_max - cs0) ** alpha * cs0 ** (1.0 - alpha)
        )

    return factor


def _kappa_eff_factor(region: str) -> Callable[[Mapping[str, float]], float]:
    def factor(values: Mapping[str, float]) -> float:
        epse = values.get(f"epse_{region}", _DEFAULTS[f"epse_{region}"])
        brug = values.get(f"brug_{region}", _DEFAULTS[f"brug_{region}"])
        return float(epse**brug)

    return factor


def _sei_r_factor(prefix: str) -> Callable[[Mapping[str, float]], float]:
    # delta_sei = rs / 50  =>  R_sei = rou_sei * rs / 51
    def factor(values: Mapping[str, float]) -> float:
        rs = values.get(f"rs_{prefix}", _DEFAULTS[f"rs_{prefix}"])
        return float(rs / 51.0)

    return factor


def _sei_c_factor(prefix: str) -> Callable[[Mapping[str, float]], float]:
    # C_sei = epse_sei * (rs + delta)/(rs*delta) = epse_sei * 51 / rs
    def factor(values: Mapping[str, float]) -> float:
        rs = values.get(f"rs_{prefix}", _DEFAULTS[f"rs_{prefix}"])
        return float(51.0 / rs)

    return factor


def _tau_diff_factor(prefix: str) -> Callable[[Mapping[str, float]], float]:
    # Coordinate v = tau_d = rs^2 / Ds  =>  v = (1/Ds) * rs^2: implemented
    # as v = Ds * factor with factor = rs^2 / Ds^2?  No — tau is not a
    # positive power of Ds.  Handled via invert=True: v = factor / Ds.
    def factor(values: Mapping[str, float]) -> float:
        rs = values.get(f"rs_{prefix}", _DEFAULTS[f"rs_{prefix}"])
        return float(rs**2)

    return factor


def _jax_transform_factor(
    transform: "ProductTransform", values: Mapping[str, Any], jnp: Any
) -> Any:
    """Evaluate a product factor without coercing JAX tracers to ``float``."""

    def value(name: str) -> Any:
        return values.get(name, _DEFAULTS[name])

    key = transform.key
    if key.startswith(("tau_diff_", "tau_d_")):
        prefix = transform.base.removeprefix("Ds_").removesuffix("_0")
        return value(f"rs_{prefix}") ** 2
    if key.startswith("sei_r_"):
        prefix = transform.base.removeprefix("rou_sei_").removesuffix("_0")
        return value(f"rs_{prefix}") / 51.0
    if key.startswith("sei_c_"):
        prefix = transform.base.removeprefix("epse_sei_")
        return 51.0 / value(f"rs_{prefix}")
    if key.startswith("kappa_eff_"):
        region = key.removeprefix("kappa_eff_")
        return value(f"epse_{region}") ** value(f"brug_{region}")
    if key.startswith("i0_"):
        prefix = key.removeprefix("i0_")
        alpha = value(f"alpha_a_{prefix}")
        cs_max = value(f"cs_max_{prefix}")
        stoich = value(f"s0_{prefix}") + 0.5 * (
            value(f"s100_{prefix}") - value(f"s0_{prefix}")
        )
        cs0 = cs_max * stoich
        return (
            F_CONST
            * value("ce_0") ** alpha
            * (cs_max - cs0) ** alpha
            * cs0 ** (1.0 - alpha)
        )
    raise ValueError(f"no JAX factor implementation for transform {key!r}")


@dataclass(frozen=True)
class ProductTransform:
    """One physics-product coordinate replacement.

    ``invert=False``:  v = base * factor(deps),   base = v / factor
    ``invert=True``:   v = factor(deps) / base,   base = factor / v
    (the diffusion time constant is inversely proportional to Ds).
    """

    key: str
    base: str
    factor: Callable[[Mapping[str, float]], float]
    dependencies: tuple[str, ...]
    description: str
    unit: str
    invert: bool = False

    def forward(self, base_value: float, values: Mapping[str, float]) -> float:
        f = self.factor(values)
        if not np.isfinite(base_value) or base_value <= 0:
            raise ValueError(f"{self.base} must be finite and positive")
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"{self.key} factor must be finite and positive")
        value = f / base_value if self.invert else base_value * f
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{self.key} coordinate must be finite and positive")
        return float(value)

    def backward(self, coord_value: float, values: Mapping[str, float]) -> float:
        f = self.factor(values)
        if not np.isfinite(coord_value) or coord_value <= 0:
            raise ValueError(f"{self.key} coordinate must be finite and positive")
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"{self.key} factor must be finite and positive")
        value = f / coord_value if self.invert else coord_value / f
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{self.base} inverse must be finite and positive")
        return float(value)


def _all_transforms() -> tuple[ProductTransform, ...]:
    transforms: list[ProductTransform] = []
    for prefix in ("neg", "pos"):
        transforms.append(
            ProductTransform(
                key=f"tau_d_{prefix}",
                base=f"Ds_{prefix}_0",
                factor=_tau_diff_factor(prefix),
                dependencies=(f"rs_{prefix}",),
                description=(
                    f"solid diffusion time constant tau_d = rs_{prefix}^2 / "
                    f"Ds_{prefix} (was: diffusivity)"
                ),
                unit="s",
                invert=True,
            )
        )
        transforms.append(
            ProductTransform(
                key=f"sei_r_{prefix}",
                base=f"rou_sei_{prefix}_0",
                factor=_sei_r_factor(prefix),
                dependencies=(f"rs_{prefix}",),
                description=(
                    f"SEI film resistance R_sei = rou_sei * rs_{prefix}/51 "
                    "(was: SEI resistivity)"
                ),
                unit="ohm*m^2",
            )
        )
        transforms.append(
            ProductTransform(
                key=f"sei_c_{prefix}",
                base=f"epse_sei_{prefix}",
                factor=_sei_c_factor(prefix),
                dependencies=(f"rs_{prefix}",),
                description=(
                    f"SEI film capacitance C_sei = epse_sei * 51/rs_{prefix} "
                    "(was: SEI permittivity)"
                ),
                unit="F/m^2",
            )
        )
        transforms.append(
            ProductTransform(
                key=f"i0_{prefix}",
                base=f"k_{prefix}_0",
                factor=_i0_factor(prefix),
                dependencies=(
                    f"alpha_a_{prefix}", f"cs_max_{prefix}",
                    f"s0_{prefix}", f"s100_{prefix}", "ce_0",
                ),
                description=(
                    f"exchange current density i0_{prefix} at SOC=0.5, "
                    "ce=ce_0 (was: rate coefficient)"
                ),
                unit="A/m^2",
            )
        )
    transforms.append(
        ProductTransform(
            key="kappa_eff_neg",
            base="kappa_0",
            factor=_kappa_eff_factor("neg"),
            dependencies=("epse_neg", "brug_neg"),
            description=(
                "effective electrolyte conductivity kappa_eff = "
                "kappa * epse_neg^brug_neg (was: bulk conductivity)"
            ),
            unit="S/m",
        )
    )
    return tuple(transforms)


#: Publication-candidate physical composite set.  No selection in this tuple
#: depends on exposed recovery performance.
DEFAULT_ENABLED: tuple[str, ...] = (
    "tau_d_neg",
    "tau_d_pos",
    "sei_r_neg",
    "sei_c_neg",
    "sei_r_pos",
    "sei_c_pos",
    "kappa_eff_neg",
)


@dataclass(frozen=True)
class PhysicsProductReparameterizer:
    """Bidirectional physics-product coordinate transformation.

    Parameter names are preserved; the *meaning* of each transformed
    coordinate is documented by :meth:`coordinate_descriptions`.
    Dependencies are read live from the parameter vector when present
    and from the literature defaults otherwise, which keeps subset
    coordinates consistent with full-set coordinates (the Gen-1 model
    freezes absent parameters at those same defaults).
    """

    enabled: tuple[str, ...] = DEFAULT_ENABLED

    def __post_init__(self) -> None:
        legacy = set(self.enabled) & {"d_eff_neg", "d_eff_pos"}
        if legacy:
            warnings.warn(
                "The Ds*rs^2 diffusion coordinate was removed from the "
                "publication candidate; use tau_d_neg/tau_d_pos. Old native "
                "results are not forward-compatible.",
                FutureWarning,
                stacklevel=2,
            )
            raise ValueError(
                f"removed backward-incompatible transforms requested: {sorted(legacy)}"
            )
        known = {t.key for t in _all_transforms()}
        unknown = set(self.enabled) - known
        if unknown:
            raise ValueError(f"unknown transforms: {sorted(unknown)}")

    @property
    def transforms(self) -> tuple[ProductTransform, ...]:
        return tuple(t for t in _all_transforms() if t.key in self.enabled)

    def _active(self, names: Sequence[str]) -> list[ProductTransform]:
        name_set = set(names)
        return [t for t in self.transforms if t.base in name_set]

    def _values(
        self, names: Sequence[str], theta: NDArray[np.floating]
    ) -> dict[str, float]:
        return dict(zip(names, np.asarray(theta, dtype=float), strict=True))

    # -- vector transforms --------------------------------------------------

    def from_original(
        self, names: Sequence[str], theta_original: NDArray[np.floating]
    ) -> FloatArray:
        theta = np.asarray(theta_original, dtype=float).copy()
        values = self._values(names, theta)
        name_list = list(names)
        for transform in self._active(names):
            i = name_list.index(transform.base)
            theta[i] = transform.forward(theta[i], values)
        return theta

    def to_original(
        self, names: Sequence[str], theta_reparam: NDArray[np.floating]
    ) -> FloatArray:
        theta = np.asarray(theta_reparam, dtype=float).copy()
        # Dependencies are never themselves transformed, so their values
        # can be read directly from the reparameterized vector.
        values = self._values(names, theta)
        name_list = list(names)
        for transform in self._active(names):
            i = name_list.index(transform.base)
            theta[i] = transform.backward(theta[i], values)
        return theta

    def validate_coupled_raw_bounds(
        self, names: Sequence[str], theta_reparam: NDArray[np.floating]
    ) -> None:
        """Reject native vectors outside the exact coupled raw domain."""

        raw = self.to_original(names, theta_reparam)
        for name, value in zip(names, raw, strict=True):
            spec = _DEFAULT_SPECS.get(name)
            if spec is None:
                continue
            lower, upper = spec.bounds
            tolerance = 1e-12 * max(
                abs(lower), abs(upper), np.finfo(float).tiny
            )
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"native vector back-transforms outside raw {name} bounds"
                )

    def native_coordinate_values(
        self, names: Sequence[str], theta_reparam: NDArray[np.floating]
    ) -> dict[str, float]:
        """Serialize transformed values under unambiguous composite keys."""

        theta = np.asarray(theta_reparam, dtype=float)
        name_to_index = {name: i for i, name in enumerate(names)}
        return {
            transform.key: float(theta[name_to_index[transform.base]])
            for transform in self._active(names)
        }

    def coordinate_manifest(self, names: Sequence[str]) -> list[dict[str, object]]:
        """Machine-readable publication coordinate definitions."""

        return [
            {
                "native_name": transform.key,
                "storage_parameter": transform.base,
                "dependencies": list(transform.dependencies),
                "unit": transform.unit,
                "definition": transform.description,
                "invert_base": bool(transform.invert),
            }
            for transform in self._active(names)
        ]

    def to_original_jax(self, names: Sequence[str], theta_reparam: Any) -> Any:
        """JAX-traceable inverse physics-product transform."""

        import jax.numpy as jnp

        theta = jnp.asarray(theta_reparam, dtype=jnp.float64)
        name_list = list(names)
        values = {
            name: theta[index] for index, name in enumerate(name_list)
        }
        for transform in self._active(names):
            index = name_list.index(transform.base)
            factor = _jax_transform_factor(transform, values, jnp)
            original = (
                factor / theta[index]
                if transform.invert
                else theta[index] / factor
            )
            theta = theta.at[index].set(original)
        return theta

    # -- spec / prior transforms ---------------------------------------------

    def _default_factor(self, transform: ProductTransform) -> float:
        return float(transform.factor(_DEFAULTS))

    def reparameterize(
        self,
        original_specs: Sequence[ParameterSpec],
        original_theta: NDArray[np.floating],
    ) -> tuple[list[ParameterSpec], FloatArray]:
        specs = list(original_specs)
        names = [spec.name for spec in specs]
        theta = self.from_original(names, original_theta)
        by_base = {t.base: t for t in self._active(names)}
        spec_by_name = {spec.name: spec for spec in specs}
        new_specs: list[ParameterSpec] = []
        for spec in specs:
            transform = by_base.get(spec.name)
            if transform is None:
                new_specs.append(spec)
                continue
            if not spec.log_transform:
                raise ValueError(
                    f"physics transform for {spec.name} requires a "
                    "log-transformed base parameter"
                )
            f = self._default_factor(transform)
            lower, upper = spec.bounds
            if transform.key.startswith("tau_d_"):
                radius = spec_by_name[transform.dependencies[0]]
                new_bounds = (
                    radius.bounds[0] ** 2 / upper,
                    radius.bounds[1] ** 2 / lower,
                )
                new_initial = (
                    radius.initial_value**2 / spec.initial_value
                )
            elif transform.invert:
                new_bounds = (f / upper, f / lower)
                new_initial = f / spec.initial_value
            else:
                new_bounds = (lower * f, upper * f)
                new_initial = spec.initial_value * f
            new_initial = float(np.clip(new_initial, *new_bounds))
            new_specs.append(
                ParameterSpec(
                    spec.name, new_initial, new_bounds, transform.unit, True
                )
            )
        return new_specs, theta

    def transform_prior_set(
        self, prior_set: PriorSet, specs: Sequence[ParameterSpec]
    ) -> PriorSet:
        """Push the Gaussian prior through the search-coordinate mapping.

        The delta-method covariance is exact for the log-linear tau relation
        ``log(tau_d)=2 log(rs)-log(Ds)`` and retains all induced correlations.
        """

        original_specs = tuple(specs)
        names = tuple(spec.name for spec in original_specs)
        mean_original = prior_set.mean_physical_vector(original_specs)
        new_specs, mean_native = self.reparameterize(original_specs, mean_original)
        new_specs_tuple = tuple(new_specs)
        jacobian = self.search_jacobian_original_to_native(
            original_specs, mean_original, new_specs_tuple
        )
        covariance_original = prior_set.covariance_search_matrix(original_specs)
        covariance_native = jacobian @ covariance_original @ jacobian.T
        covariance_native = 0.5 * (covariance_native + covariance_native.T)

        active_bases = {transform.base for transform in self._active(names)}
        old_by_name = {prior.name: prior for prior in prior_set.priors}
        priors: list[GaussianParameterPrior] = []
        for i, spec in enumerate(new_specs_tuple):
            old = old_by_name.get(spec.name)
            source = old.source if old is not None else "weak_bounds"
            if spec.name in active_bases:
                source += "+physics_pushforward"
            priors.append(
                GaussianParameterPrior(
                    name=spec.name,
                    mean_physical=float(mean_native[i]),
                    sigma_search=float(np.sqrt(covariance_native[i, i])),
                    source=source,
                )
            )

        covariances: list[GaussianSearchCovariance] = []
        for i, name_a in enumerate(names):
            for j in range(i + 1, len(names)):
                value = float(covariance_native[i, j])
                scale = float(
                    np.sqrt(covariance_native[i, i] * covariance_native[j, j])
                )
                if abs(value) <= max(1e-14 * scale, 1e-30):
                    continue
                covariances.append(
                    GaussianSearchCovariance(
                        name_a=name_a,
                        name_b=names[j],
                        covariance=value,
                    )
                )
        return PriorSet(
            priors=tuple(priors), search_covariances=tuple(covariances)
        )

    def search_jacobian_original_to_native(
        self,
        original_specs: Sequence[ParameterSpec],
        theta_original: NDArray[np.floating],
        native_specs: Sequence[ParameterSpec] | None = None,
        step: float = 1e-6,
    ) -> FloatArray:
        """Finite-difference coordinate Jacobian in optimizer coordinates."""

        original_specs = tuple(original_specs)
        if native_specs is None:
            native_specs = tuple(self.reparameterize(original_specs, theta_original)[0])
        else:
            native_specs = tuple(native_specs)
        search0 = encode_parameter_vector(original_specs, theta_original)
        jacobian = np.empty((len(original_specs), len(original_specs)), dtype=float)
        for column in range(len(original_specs)):
            plus = search0.copy()
            minus = search0.copy()
            plus[column] += step
            minus[column] -= step
            raw_plus = decode_parameter_vector(original_specs, plus)
            raw_minus = decode_parameter_vector(original_specs, minus)
            native_plus = encode_parameter_vector(
                native_specs, self.from_original([s.name for s in original_specs], raw_plus)
            )
            native_minus = encode_parameter_vector(
                native_specs, self.from_original([s.name for s in original_specs], raw_minus)
            )
            jacobian[:, column] = (native_plus - native_minus) / (2.0 * step)
        index = {spec.name: i for i, spec in enumerate(original_specs)}
        for transform in self._active(tuple(index)):
            if not transform.key.startswith("tau_d_"):
                continue
            row = index[transform.base]
            jacobian[row, :] = 0.0
            jacobian[row, index[transform.base]] = -1.0
            jacobian[row, index[transform.dependencies[0]]] = 2.0
        return jacobian

    def search_jacobian_native_to_original(
        self,
        original_specs: Sequence[ParameterSpec],
        theta_original: NDArray[np.floating],
        native_specs: Sequence[ParameterSpec] | None = None,
    ) -> FloatArray:
        """Inverse search-coordinate Jacobian for uncertainty propagation."""

        forward = self.search_jacobian_original_to_native(
            original_specs, theta_original, native_specs
        )
        return np.linalg.inv(forward)

    def backtransform_search_covariance(
        self,
        original_specs: Sequence[ParameterSpec],
        theta_original: NDArray[np.floating],
        covariance_native: NDArray[np.floating],
        native_specs: Sequence[ParameterSpec] | None = None,
    ) -> FloatArray:
        """Propagate native search covariance back to raw search coordinates."""

        jacobian = self.search_jacobian_native_to_original(
            original_specs, theta_original, native_specs
        )
        covariance = np.asarray(covariance_native, dtype=float)
        if covariance.shape != jacobian.shape:
            raise ValueError("native covariance must match parameter count")
        return jacobian @ covariance @ jacobian.T

    # -- model wrapper --------------------------------------------------------

    def wrap_model(self, original_model: Any) -> "PhysicsReparameterizedModel":
        return PhysicsReparameterizedModel(
            original_model=original_model, reparameterizer=self
        )

    def coordinate_descriptions(self, names: Sequence[str]) -> dict[str, str]:
        return {t.base: t.description for t in self._active(names)}


@dataclass(frozen=True)
class PhysicsReparameterizedModel:
    """ForwardModel wrapper: physics coordinates in, original model out."""

    original_model: Any
    reparameterizer: PhysicsProductReparameterizer

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.original_model.parameter_names)

    @property
    def conditions(self):
        return getattr(self.original_model, "conditions", None)

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        theta_original = self.reparameterizer.to_original(
            self.parameter_names, theta
        )
        return self.original_model.simulate(freq_hz, theta_original)

    def simulate_jax(self, freq_hz: Any, theta: Any) -> Any:
        simulator = getattr(self.original_model, "simulate_jax", None)
        if simulator is None:
            raise AttributeError("original model does not expose a JAX simulation path")
        theta_original = self.reparameterizer.to_original_jax(
            self.parameter_names, theta
        )
        return simulator(freq_hz, theta_original)

    def guess_from_drt(self, drt_result: Any) -> dict[str, float] | None:
        return None


def diffusion_composite_values(raw_values: Mapping[str, float]) -> dict[str, float]:
    """Compute publication diffusion composites from raw physical factors."""

    result: dict[str, float] = {}
    for prefix in ("neg", "pos"):
        ds_name = f"Ds_{prefix}_0"
        rs_name = f"rs_{prefix}"
        if ds_name not in raw_values or rs_name not in raw_values:
            continue
        ds = float(raw_values[ds_name])
        rs = float(raw_values[rs_name])
        if not np.isfinite(ds) or not np.isfinite(rs) or ds <= 0 or rs <= 0:
            raise ValueError("diffusion factors must be finite and positive")
        result[f"tau_d_{prefix}"] = float(rs**2 / ds)
    return result


def score_diffusion_composites(
    estimated_raw: Mapping[str, float],
    truth_raw: Mapping[str, float],
) -> list[dict[str, object]]:
    """Evaluation-only tau_d scoring, independent of primitive-factor claims."""

    estimated = diffusion_composite_values(estimated_raw)
    truth = diffusion_composite_values(truth_raw)
    rows: list[dict[str, object]] = []
    for name in sorted(set(estimated) & set(truth)):
        error = abs(estimated[name] - truth[name]) / max(abs(truth[name]), 1e-30)
        rows.append(
            {
                "parameter": name,
                "coordinate_class": "COMPOSITE",
                "estimated_value": estimated[name],
                "truth_value": truth[name],
                "relative_error": float(error),
                "primitive_factor_implication": False,
                "evaluation_only": True,
            }
        )
    return rows
