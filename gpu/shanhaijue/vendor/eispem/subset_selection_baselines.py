# learning AI at www.haotianblog.com
"""Standard matrix-based parameter-subset baselines.

These selectors consume the same weighted log-coordinate Jacobian as the
EIS-PEM gate. They select a free subset only; they do not assign strict,
protected, or fixed interpretation labels. Those labels require a separate
post-fit audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SubsetSelectionResult:
    """Deterministic output from one standard subset selector."""

    method: str
    selected_indices: tuple[int, ...]
    selection_order: tuple[int, ...]
    protected_indices: tuple[int, ...]
    trace: tuple[dict[str, float | int], ...]
    epsilon: float | None = None


def _validated_inputs(
    jacobian: NDArray[np.floating],
    target_count: int,
    protected_indices: Sequence[int],
) -> tuple[FloatArray, tuple[int, ...]]:
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("jacobian must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("jacobian must contain only finite values")
    if not 1 <= target_count <= matrix.shape[1]:
        raise ValueError("target_count must be between one and the column count")
    protected = tuple(int(index) for index in protected_indices)
    if len(set(protected)) != len(protected):
        raise ValueError("protected_indices must not contain duplicates")
    if any(index < 0 or index >= matrix.shape[1] for index in protected):
        raise ValueError("protected index is outside the Jacobian column range")
    if len(protected) > target_count:
        raise ValueError("target_count cannot be smaller than the protected set")
    return matrix, protected


def project_against_protected_span(
    candidate_columns: NDArray[np.floating],
    protected_columns: NDArray[np.floating],
) -> FloatArray:
    """Project candidate columns onto the protected span's orthogonal complement."""

    candidates = np.asarray(candidate_columns, dtype=float)
    protected = np.asarray(protected_columns, dtype=float)
    if candidates.ndim != 2 or protected.ndim != 2:
        raise ValueError("candidate and protected columns must be matrices")
    if candidates.shape[0] != protected.shape[0]:
        raise ValueError("candidate and protected columns must share row count")
    if protected.shape[1] == 0:
        return candidates.copy()
    q_matrix, r_matrix = np.linalg.qr(protected, mode="reduced")
    diagonal = np.abs(np.diag(r_matrix))
    if diagonal.size == 0:
        return candidates.copy()
    tolerance = max(protected.shape) * np.finfo(float).eps * diagonal.max()
    rank = int(np.sum(diagonal > tolerance))
    if rank == 0:
        return candidates.copy()
    basis = q_matrix[:, :rank]
    return candidates - basis @ (basis.T @ candidates)


def pivoted_qr_subset(
    jacobian: NDArray[np.floating],
    target_count: int,
    protected_indices: Sequence[int] = (),
) -> SubsetSelectionResult:
    """Select a matched-size subset using rank-revealing QR column pivoting."""

    matrix, protected = _validated_inputs(
        jacobian, target_count, protected_indices
    )
    protected_set = set(protected)
    remaining = tuple(
        index for index in range(matrix.shape[1]) if index not in protected_set
    )
    required = target_count - len(protected)
    if required:
        projected = project_against_protected_span(
            matrix[:, remaining], matrix[:, protected]
        )
        _, r_matrix, pivots = qr(
            projected, mode="economic", pivoting=True, check_finite=False
        )
        pivot_order = tuple(remaining[int(position)] for position in pivots)
        diagonal = np.abs(np.diag(r_matrix))
    else:
        pivot_order = ()
        diagonal = np.array([], dtype=float)
    chosen_order = protected + pivot_order[:required]
    if len(chosen_order) != target_count or len(set(chosen_order)) != target_count:
        raise RuntimeError("QR selector did not produce the requested unique subset")
    trace = tuple(
        {
            "step": step,
            "parameter_index": index,
            "abs_r_diagonal": float(diagonal[step - len(protected)])
            if step >= len(protected) and step - len(protected) < diagonal.size
            else float("nan"),
        }
        for step, index in enumerate(chosen_order)
    )
    return SubsetSelectionResult(
        method="rank_revealing_qr_column_pivoting",
        selected_indices=tuple(sorted(chosen_order)),
        selection_order=chosen_order,
        protected_indices=protected,
        trace=trace,
    )


def threshold_qr_subset(
    jacobian: NDArray[np.floating],
    relative_diagonal_threshold: float,
    protected_indices: Sequence[int] = (),
) -> SubsetSelectionResult:
    """Return a pre-specified QR diagonal-threshold sensitivity subset."""

    matrix = np.asarray(jacobian, dtype=float)
    if not np.isfinite(relative_diagonal_threshold) or not (
        0 < relative_diagonal_threshold <= 1
    ):
        raise ValueError("relative_diagonal_threshold must be in (0, 1]")
    protected = tuple(int(index) for index in protected_indices)
    # Validate with the largest possible target; the actual count is thresholded.
    matrix, protected = _validated_inputs(matrix, matrix.shape[1], protected)
    protected_set = set(protected)
    remaining = tuple(
        index for index in range(matrix.shape[1]) if index not in protected_set
    )
    projected = project_against_protected_span(
        matrix[:, remaining], matrix[:, protected]
    )
    _, r_matrix, pivots = qr(
        projected, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r_matrix))
    scale = float(diagonal[0]) if diagonal.size else 0.0
    count = (
        int(np.sum(diagonal / scale >= relative_diagonal_threshold))
        if scale > 0
        else 0
    )
    pivot_order = tuple(remaining[int(position)] for position in pivots)
    chosen_order = protected + pivot_order[:count]
    trace = tuple(
        {
            "step": step,
            "parameter_index": index,
            "relative_r_diagonal": float(diagonal[step - len(protected)] / scale)
            if step >= len(protected) and scale > 0
            else float("nan"),
        }
        for step, index in enumerate(chosen_order)
    )
    return SubsetSelectionResult(
        method=f"qr_relative_diagonal_threshold_{relative_diagonal_threshold:g}",
        selected_indices=tuple(sorted(chosen_order)),
        selection_order=chosen_order,
        protected_indices=protected,
        trace=trace,
    )


def fim_logdet(
    jacobian: NDArray[np.floating],
    selected_indices: Sequence[int],
    epsilon: float,
) -> tuple[float, float]:
    """Return raw and epsilon-normalized regularized FIM log determinants.

    The normalized value subtracts ``|S| log(epsilon)``. It has identical
    within-step candidate ranking to the requested raw objective and is
    monotone as information columns are added.
    """

    matrix = np.asarray(jacobian, dtype=float)
    selected = tuple(int(index) for index in selected_indices)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not selected:
        return 0.0, 0.0
    subset = matrix[:, selected]
    information = subset.T @ subset + epsilon * np.eye(len(selected))
    sign, raw = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(raw):
        raise RuntimeError("regularized FIM is not positive definite")
    normalized = float(raw - len(selected) * np.log(epsilon))
    return float(raw), normalized


def greedy_fim_subset(
    jacobian: NDArray[np.floating],
    target_count: int,
    protected_indices: Sequence[int] = (),
    epsilon_scale: float = 1e-12,
) -> SubsetSelectionResult:
    """Select a matched-size subset with deterministic greedy D-optimal FIM."""

    matrix, protected = _validated_inputs(
        jacobian, target_count, protected_indices
    )
    if not np.isfinite(epsilon_scale) or epsilon_scale <= 0:
        raise ValueError("epsilon_scale must be finite and positive")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    epsilon = epsilon_scale * max(sigma_max**2, np.finfo(float).tiny)
    selected = list(protected)
    candidates = [
        index for index in range(matrix.shape[1]) if index not in set(selected)
    ]
    trace: list[dict[str, float | int]] = []
    previous_normalized = 0.0
    if selected:
        raw, previous_normalized = fim_logdet(matrix, selected, epsilon)
        trace.append(
            {
                "step": 0,
                "parameter_index": -1,
                "raw_logdet": raw,
                "normalized_logdet": previous_normalized,
                "normalized_gain": previous_normalized,
            }
        )
    while len(selected) < target_count:
        best_index: int | None = None
        best_raw = -np.inf
        best_normalized = -np.inf
        for candidate in candidates:
            raw, normalized = fim_logdet(matrix, selected + [candidate], epsilon)
            if (
                best_index is None
                or raw > best_raw + 1e-12
                or (abs(raw - best_raw) <= 1e-12 and candidate < best_index)
            ):
                best_index = candidate
                best_raw = raw
                best_normalized = normalized
        if best_index is None:
            raise RuntimeError("FIM selector exhausted candidates before target count")
        gain = best_normalized - previous_normalized
        if gain < -1e-9:
            raise RuntimeError("normalized FIM information decreased after selection")
        selected.append(best_index)
        candidates.remove(best_index)
        trace.append(
            {
                "step": len(selected),
                "parameter_index": best_index,
                "raw_logdet": best_raw,
                "normalized_logdet": best_normalized,
                "normalized_gain": gain,
            }
        )
        previous_normalized = best_normalized
    return SubsetSelectionResult(
        method="greedy_d_optimal_fim",
        selected_indices=tuple(sorted(selected)),
        selection_order=tuple(selected),
        protected_indices=protected,
        trace=tuple(trace),
        epsilon=float(epsilon),
    )


def pairwise_jaccard(subsets: Sequence[Sequence[int]]) -> FloatArray:
    """Return all pairwise Jaccard similarities for a sequence of subsets."""

    values: list[float] = []
    sets = [set(subset) for subset in subsets]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left].union(sets[right])
            values.append(
                1.0 if not union else len(sets[left].intersection(sets[right])) / len(union)
            )
    return np.asarray(values, dtype=float)
