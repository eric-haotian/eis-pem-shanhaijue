# learning AI at www.haotianblog.com
"""Optional JAX residual and exact-Jacobian bridge for SciPy least squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .parameters import (
    ParameterSpec,
    decode_parameter_vector,
    encode_parameter_vector,
)


def _jax_backend(model: Any) -> tuple[bool, str | None]:
    """Find whether JAX is requested, and its platform, through wrappers."""

    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        backend = getattr(current, "compute_backend", None)
        if backend in {"jax-cpu", "jax-gpu"}:
            return True, backend.removeprefix("jax-")
        if backend == "jax":
            return True, None
        if backend == "numpy":
            return False, None
        next_model = getattr(current, "full_model", None)
        if next_model is None:
            next_model = getattr(current, "original_model", None)
        if next_model is None:
            break
        current = next_model
    return False, None


def supports_jax(model: Any) -> bool:
    requested, _ = _jax_backend(model)
    return requested and callable(getattr(model, "simulate_jax", None))


def _canonical_raw_problem(
    model: Any, specs: Sequence[ParameterSpec]
) -> tuple[Any, tuple[ParameterSpec, ...], Any, bool]:
    """Unwrap subset/reparameterization layers outside the compiled kernel.

    The expensive DFN graph is compiled in stable raw 48-parameter
    coordinates. Case-specific fixed values and native-coordinate shears are
    applied in NumPy, allowing the persistent XLA cache to be reused across
    held-out cases and BCD stages with the same condition grid.
    """

    current = model
    path: list[tuple[str, Any]] = []
    root_specs = tuple(specs)
    transformed = False
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "expand_theta", None)) and hasattr(
            current, "full_model"
        ):
            path.append(("subset", current))
            root_specs = tuple(current.full_specs)
            current = current.full_model
            transformed = True
            continue
        if hasattr(current, "reparameterizer") and hasattr(current, "original_model"):
            path.append(("reparameterization", current))
            current = current.original_model
            transformed = True
            continue
        break

    root_model = current
    root_names = tuple(getattr(root_model, "parameter_names", ()))
    if not root_names:
        raise ValueError("JAX root model must expose parameter_names")
    if tuple(spec.name for spec in root_specs) != root_names:
        from .seis_model import all_seis_parameter_specs

        raw_by_name = {spec.name: spec for spec in all_seis_parameter_specs()}
        try:
            root_specs = tuple(raw_by_name[name] for name in root_names)
        except KeyError as exc:
            raise ValueError(f"missing raw specification for {exc.args[0]!r}") from exc

    def to_root_search(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
        theta = decode_parameter_vector(tuple(specs), search_values)
        for kind, wrapper in path:
            if kind == "subset":
                theta = wrapper.expand_theta(theta)
            else:
                theta = wrapper.reparameterizer.to_original(
                    wrapper.parameter_names, theta
                )
        return encode_parameter_vector(root_specs, theta)

    return root_model, root_specs, to_root_search, transformed


@dataclass
class JaxLeastSquaresFunctions:
    residual: Any
    jacobian: Any
    platform: str
    device: str


def build_jax_least_squares_functions(
    *,
    model: Any,
    freq_hz: NDArray[np.floating],
    z_obs: NDArray[np.complexfloating],
    specs: Sequence[ParameterSpec],
    relative: bool,
    eps: float,
    weights: NDArray[np.floating] | None,
    alpha: float = 0.0,
    alpha_reference: NDArray[np.floating] | None = None,
    regularization_weights: NDArray[np.floating] | None = None,
    regularization_reference: NDArray[np.floating] | None = None,
    regularization_matrix: NDArray[np.floating] | None = None,
    regularization_matrix_reference: NDArray[np.floating] | None = None,
) -> JaxLeastSquaresFunctions:
    """Build NumPy-callable JIT residual/Jacobian functions for SciPy.

    The physical forward model remains on the selected JAX device while SciPy
    retains the established bounded trust-region policy and stopping rules.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "JAX optimizer path requested; install the optional 'gpu' dependencies"
        ) from exc

    jax.config.update("jax_enable_x64", True)
    requested, requested_platform = _jax_backend(model)
    if not requested:
        raise ValueError("JAX least-squares functions require a JAX model backend")
    devices = jax.devices(requested_platform) if requested_platform else jax.devices()
    if not devices:
        raise RuntimeError(f"no JAX device found for platform {requested_platform!r}")
    device = devices[0]

    spec_tuple = tuple(specs)
    root_model, root_specs, to_root_search, transformed = _canonical_raw_problem(
        model, spec_tuple
    )
    log_mask = jnp.asarray(
        [spec.log_transform for spec in root_specs], dtype=bool
    )
    freq = jax.device_put(jnp.asarray(freq_hz, dtype=jnp.float64), device)
    observed = np.asarray(z_obs, dtype=complex)
    scale = np.maximum(np.abs(observed), eps) if relative else None
    sqrt_weights = (
        None if weights is None else np.sqrt(np.asarray(weights, dtype=float))
    )
    alpha_ref = (
        None
        if alpha_reference is None
        else np.asarray(alpha_reference, dtype=float)
    )
    selective_weights = (
        None
        if regularization_weights is None
        else np.asarray(regularization_weights, dtype=float)
    )
    selective_reference = (
        None
        if regularization_reference is None
        else np.asarray(regularization_reference, dtype=float)
    )
    matrix = (
        None
        if regularization_matrix is None
        else np.asarray(regularization_matrix, dtype=float)
    )
    matrix_reference = (
        None
        if regularization_matrix_reference is None
        else np.asarray(regularization_matrix_reference, dtype=float)
    )

    def prediction_device(search_values: Any, frequencies: Any) -> Any:
        physical = jnp.where(log_mask, 10.0**search_values, search_values)
        return root_model.simulate_jax(frequencies, physical)

    prediction_jit = jax.jit(prediction_device)
    jacobian_jit = jax.jit(jax.jacfwd(prediction_device, argnums=0))
    residual_cache: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    jacobian_cache: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    def residual_numpy(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
        nonlocal residual_cache
        values = np.asarray(search_values, dtype=float)
        if residual_cache is None or not np.array_equal(values, residual_cache[0]):
            root_values = to_root_search(values)
            device_values = jax.device_put(jnp.asarray(root_values), device)
            prediction = np.asarray(prediction_jit(device_values, freq), dtype=complex)
            residual = observed - prediction
            if scale is not None:
                residual = residual / scale
            if sqrt_weights is not None:
                residual = residual * sqrt_weights
            rows = [np.concatenate((residual.real, residual.imag))]
            if alpha > 0 and alpha_ref is not None:
                rows.append(alpha * (values - alpha_ref))
            if selective_weights is not None and selective_reference is not None:
                rows.append(selective_weights * (values - selective_reference))
            if matrix is not None and matrix_reference is not None:
                rows.append(matrix @ (values - matrix_reference))
            result = np.concatenate(rows) if len(rows) > 1 else rows[0]
            residual_cache = (values.copy(), result)
        return residual_cache[1]

    def jacobian_numpy(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
        nonlocal jacobian_cache
        values = np.asarray(search_values, dtype=float)
        if jacobian_cache is None or not np.array_equal(values, jacobian_cache[0]):
            root_values = to_root_search(values)
            device_values = jax.device_put(jnp.asarray(root_values), device)
            prediction_jacobian = np.asarray(
                jacobian_jit(device_values, freq), dtype=complex
            )
            if transformed:
                step = 1e-6
                transform_columns = []
                for column in range(len(values)):
                    delta = np.zeros_like(values)
                    delta[column] = step
                    transform_columns.append(
                        (to_root_search(values + delta) - to_root_search(values - delta))
                        / (2.0 * step)
                    )
                coordinate_jacobian = np.column_stack(transform_columns)
                prediction_jacobian = prediction_jacobian @ coordinate_jacobian
            residual_jacobian = -prediction_jacobian
            if scale is not None:
                residual_jacobian = residual_jacobian / scale[:, None]
            if sqrt_weights is not None:
                residual_jacobian = residual_jacobian * sqrt_weights[:, None]
            rows = [
                np.concatenate(
                    (residual_jacobian.real, residual_jacobian.imag), axis=0
                )
            ]
            parameter_count = len(spec_tuple)
            if alpha > 0 and alpha_ref is not None:
                rows.append(alpha * np.eye(parameter_count))
            if selective_weights is not None and selective_reference is not None:
                rows.append(np.diag(selective_weights))
            if matrix is not None and matrix_reference is not None:
                rows.append(matrix)
            result = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
            jacobian_cache = (values.copy(), result)
        return jacobian_cache[1]

    return JaxLeastSquaresFunctions(
        residual=residual_numpy,
        jacobian=jacobian_numpy,
        platform=device.platform,
        device=str(device),
    )
