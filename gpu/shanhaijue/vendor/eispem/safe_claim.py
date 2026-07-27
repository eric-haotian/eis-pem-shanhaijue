# learning AI at www.haotianblog.com
"""Prior-robust safe-claim certification for SEIS identification.

The Gen-2 tiered gate answers "is this parameter determined by data+prior
at the fitted point?".  On the synthetic benchmark that reads as 22/48,
but the benchmark sets ``theta_true = prior mean``, so prior-informed
claims hit for free.  Under realistic **prior misspecification** (truth
!= literature prior) the same pipeline claims 15-27 parameters yet only
5-7 have a confidence interval that actually covers the true value — a
~70 % false-claim rate (measured, 5x5 grid, 1 % noise).

A claim is only *safe* — near-zero false-claim — when it survives two
independent stress tests that the posterior gate does not apply:

1. **Marginal (not conditional) uncertainty.**  The claimed parameters
   carry **no prior on themselves** (the claim must come from data), and
   the remaining nuisances are **marginalized with their prior
   uncertainty** rather than fixed at literature values.  With claimed
   block ``C`` and nuisance block ``N`` the information is the Schur
   complement

       F_marg = F_CC - F_CN (F_NN + P_N)^{-1} F_NC ,

   and the CI is ``ci_scale * sqrt(diag(F_marg^{-1}))``.  This is the
   honest interval; the conditional posterior CI the gate uses is
   systematically too narrow (it fixes nuisances and leans on the
   prior).

2. **Bias robustness.**  A parameter the data constrains only
   *conditionally* on an uncertain nuisance is biased when that nuisance
   (or its prior) is misspecified.  The linearized worst-case bias of a
   data-only fit is

       bias_C = F_CC^{-1} F_CN * delta_N ,

   and treating each nuisance offset ``delta_N`` as its own posterior
   standard deviation gives a bias CI per claimed parameter.  A claim is
   rejected unless its bias CI is a small fraction of its claim CI — so
   the reported estimate cannot be silently pulled to a wrong prior.

The certifier reports which parameters pass both tests (``safe`` claims)
alongside the full audit (marginal CI, bias CI).  On the 5x5 grid the
safe ceiling is ~9 at CI<=20 % and ~12-14 at CI<=30 % — far below the
optimistic 22, but genuinely near-zero-false-claim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)


def _safe_inverse(matrix: FloatArray) -> FloatArray | None:
    """Invert a symmetric PD matrix; return ``None`` if singular."""

    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None


def marginal_covariance(
    data_fim: FloatArray,
    prior_precision: FloatArray,
    claimed: Sequence[int],
    nuisance_prior: bool = True,
) -> FloatArray | None:
    """Schur-complement marginal covariance of the claimed block.

    Claimed parameters carry no prior on themselves.  With
    ``nuisance_prior=True`` the nuisances are marginalized with their
    prior precision (they are literature-constrained but uncertain);
    with ``nuisance_prior=False`` they are profiled out completely (no
    prior trust), the fully prior-robust "profile marginal" — a Moore-
    Penrose pseudo-inverse absorbs the rank-deficient nuisance block.
    Returns ``None`` when the claimed block is data-singular.
    """

    claimed = list(claimed)
    n = data_fim.shape[0]
    nuisance = [i for i in range(n) if i not in claimed]
    fim_cc = data_fim[np.ix_(claimed, claimed)]
    if nuisance:
        fim_cn = data_fim[np.ix_(claimed, nuisance)]
        fim_nn = data_fim[np.ix_(nuisance, nuisance)]
        if nuisance_prior:
            fim_nn = fim_nn + np.diag(prior_precision[nuisance])
            fim_nn_inv = _safe_inverse(fim_nn)
            if fim_nn_inv is None:  # pragma: no cover - prior makes this PD
                return None
        else:
            # Fully free nuisances: profile them out with a pseudo-inverse.
            fim_nn_inv = np.linalg.pinv(fim_nn, hermitian=True)
        fim_marg = fim_cc - fim_cn @ fim_nn_inv @ fim_cn.T
    else:
        fim_marg = fim_cc
    return _safe_inverse(fim_marg)


def nuisance_bias_sd(
    data_fim: FloatArray,
    claimed: Sequence[int],
    nuisance_offset_sd: FloatArray,
) -> FloatArray | None:
    """Search-space bias std of each claimed param from nuisance offsets.

    Linearized bias of a data-only fit that fixes nuisances offset by
    ``nuisance_offset_sd``: ``bias_C = F_CC^{-1} F_CN delta_N``.  Treating
    the offsets as independent zero-mean with the given std gives a
    per-parameter bias standard deviation.  Returns ``None`` if the
    claimed block is data-singular.
    """

    claimed = list(claimed)
    n = data_fim.shape[0]
    nuisance = [i for i in range(n) if i not in claimed]
    fim_cc_inv = _safe_inverse(data_fim[np.ix_(claimed, claimed)])
    if fim_cc_inv is None:
        return None
    if not nuisance:
        return np.zeros(len(claimed))
    sensitivity = fim_cc_inv @ data_fim[np.ix_(claimed, nuisance)]  # (|C|, |N|)
    return np.sqrt((sensitivity**2) @ (nuisance_offset_sd[nuisance] ** 2))


@dataclass(frozen=True)
class SafeClaimReport:
    """Prior-robust safe-claim certification over the full parameter set."""

    parameter_names: tuple[str, ...]
    safe_params: tuple[str, ...]
    claimed_block: tuple[str, ...]
    marginal_ci95: dict[str, float]
    bias_ci95: dict[str, float]
    max_marginal_ci95: float
    max_bias_fraction: float

    @property
    def n_safe(self) -> int:
        return len(self.safe_params)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for name in self.parameter_names:
            rows.append(
                {
                    "parameter": name,
                    "safe": name in self.safe_params,
                    "in_claimed_block": name in self.claimed_block,
                    "marginal_ci95": self.marginal_ci95.get(name, float("inf")),
                    "bias_ci95": self.bias_ci95.get(name, float("inf")),
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class SafeClaimCertifier:
    """Certify prior-robust safe claims from a data FIM and prior geometry.

    Parameters
    ----------
    max_marginal_ci95
        Upper bound on the marginal (nuisance-marginalized) relative CI95
        of a safe claim.
    max_bias_fraction
        A safe claim's nuisance-induced bias CI must not exceed
        ``max_bias_fraction * max_marginal_ci95``.  1.0 means the
        worst-case bias may be as large as the claim interval itself
        (the estimate stays inside its own CI under misspecification);
        smaller is stricter.
    search_seeds, search_steps, max_block
        Simulated-annealing budget for the safe-subset search.
    """

    max_marginal_ci95: float = 0.20
    max_bias_fraction: float = 1.0
    nuisance_prior: bool = True
    search_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    search_steps: int = 20000
    max_block: int = 34

    def __post_init__(self) -> None:
        if not 0 < self.max_marginal_ci95:
            raise ValueError("max_marginal_ci95 must be positive")
        if not 0 < self.max_bias_fraction:
            raise ValueError("max_bias_fraction must be positive")

    def _offset_sd(
        self, data_fim: FloatArray, prior_sigma: FloatArray
    ) -> FloatArray:
        """Self-consistent nuisance offset = full-posterior standard dev.

        A nuisance the data pins can only bias a claim by its own
        (small) posterior spread, not by its full prior width.
        """

        prior_precision = 1.0 / prior_sigma**2
        cov = _safe_inverse(data_fim + np.diag(prior_precision))
        if cov is None:  # pragma: no cover - prior makes this PD
            return prior_sigma.copy()
        return np.sqrt(np.maximum(np.diag(cov), 0.0))

    def certify_block(
        self,
        parameter_names: Sequence[str],
        data_fim: FloatArray,
        prior_sigma: FloatArray,
        ci_scale: FloatArray,
        claimed_block: Sequence[int],
        offset_sd: FloatArray | None = None,
    ) -> list[int]:
        """Indices in ``claimed_block`` that pass both safe-claim tests."""

        prior_precision = 1.0 / prior_sigma**2
        cov = marginal_covariance(
            data_fim, prior_precision, claimed_block,
            nuisance_prior=self.nuisance_prior,
        )
        if cov is None:
            return []
        claimed = list(claimed_block)
        marg_ci = ci_scale[claimed] * np.sqrt(np.maximum(np.diag(cov), 0.0))
        if offset_sd is None:
            offset_sd = self._offset_sd(data_fim, prior_sigma)
        bias_sd = nuisance_bias_sd(data_fim, claimed, offset_sd)
        if bias_sd is None:
            return []
        bias_ci = ci_scale[claimed] * bias_sd
        bias_bar = self.max_bias_fraction * self.max_marginal_ci95
        return [
            claimed[k]
            for k in range(len(claimed))
            if marg_ci[k] <= self.max_marginal_ci95 and bias_ci[k] <= bias_bar
        ]

    def search(
        self,
        parameter_names: Sequence[str],
        data_fim: FloatArray,
        prior_sigma: FloatArray,
        ci_scale: FloatArray,
        candidate_indices: Sequence[int] | None = None,
    ) -> SafeClaimReport:
        """Maximize the number of prior-robust safe claims by annealing.

        The safe status of a parameter depends on the whole partition
        (which parameters are claimed vs treated as nuisances), so this
        is a subset search, solved with the same deterministic annealing
        used elsewhere in the pipeline.
        """

        names = tuple(parameter_names)
        n = len(names)
        offset_sd = self._offset_sd(data_fim, prior_sigma)
        pool = (
            list(candidate_indices)
            if candidate_indices is not None
            else list(range(n))
        )

        def n_safe(block: Sequence[int]) -> int:
            return len(
                self.certify_block(
                    names, data_fim, prior_sigma, ci_scale, block, offset_sd
                )
            )

        best_count, best_block = 0, []
        for seed in self.search_seeds:
            rng = np.random.default_rng(seed)
            current: set[int] = set()
            cur = 0
            for step in range(self.search_steps):
                temperature = max(0.02, float(np.exp(-step / max(self.search_steps / 4, 1))))
                trial = set(current)
                r = rng.random()
                if r < 0.45 and len(trial) < self.max_block:
                    trial.add(int(pool[rng.integers(len(pool))]))
                elif r < 0.9 and len(trial) > 1:
                    trial.discard(int(rng.choice(list(trial))))
                elif len(trial) > 1:
                    trial.discard(int(rng.choice(list(trial))))
                    trial.add(int(pool[rng.integers(len(pool))]))
                if not trial or trial == current:
                    continue
                count = n_safe(sorted(trial))
                if count - cur > 0 or rng.random() < np.exp((count - cur) / temperature):
                    current, cur = trial, count
                    if count > best_count:
                        best_count, best_block = count, sorted(trial)

        safe_idx = self.certify_block(
            names, data_fim, prior_sigma, ci_scale, best_block, offset_sd
        )
        prior_precision = 1.0 / prior_sigma**2
        cov = marginal_covariance(
            data_fim, prior_precision, best_block,
            nuisance_prior=self.nuisance_prior,
        )
        bias_sd = nuisance_bias_sd(data_fim, best_block, offset_sd)
        marg = {}
        bias = {}
        if cov is not None and bias_sd is not None:
            marg_ci = ci_scale[best_block] * np.sqrt(np.maximum(np.diag(cov), 0.0))
            bias_ci = ci_scale[best_block] * bias_sd
            for k, i in enumerate(best_block):
                marg[names[i]] = float(marg_ci[k])
                bias[names[i]] = float(bias_ci[k])
        return SafeClaimReport(
            parameter_names=names,
            safe_params=tuple(names[i] for i in safe_idx),
            claimed_block=tuple(names[i] for i in best_block),
            marginal_ci95=marg,
            bias_ci95=bias,
            max_marginal_ci95=self.max_marginal_ci95,
            max_bias_fraction=self.max_bias_fraction,
        )
