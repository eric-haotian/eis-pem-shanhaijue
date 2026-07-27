# learning AI at www.haotianblog.com
"""One cell's fit, as a self-contained job.

The interface fits several cells per run, and on the CPU backend those cells are
independent, so they are handed to a process pool. Everything the child needs travels in
the task tuple — the grid, the solver settings, and the spectrum — because the child
starts with a fresh import of `panel` and must configure the grid itself before the
backend caches anything derived from it.

The GPU backend keeps its own batching inside a single process and is run sequentially;
splitting it across processes would have several of them contending for one device.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit_cell(task):
    """Fit and featurise one cell. Returns (cell_id, rows) or (cell_id, error string)."""
    cell_id, conditions, freq, z, knobs = task
    try:
        from shanhaijue import panel as P
        P.configure_grid(conditions=conditions, frequencies=freq)
        if knobs.get("prior_weight", 1.0) != 1.0:
            P.PRIOR_W = P.PRIOR_W * float(knobs["prior_weight"])

        from shanhaijue import arms as A
        from shanhaijue import features as F
        if hasattr(A, "MAX_NFEV"):
            A.MAX_NFEV = int(knobs["max_nfev"])
        else:
            A.LM_STEPS = int(knobs["max_nfev"])
            A.CHUNK = int(knobs.get("chunk", 10))
        A.N_STARTS = int(knobs["starts"])
        A.JITTER = float(knobs["jitter"])
        A.WIDE_JITTER = float(knobs["wide_jitter"])
        A.N_WIDE = int(knobs["n_wide"])

        ladder = getattr(A, {3: "LADDER_3", 6: "LADDER_6", 8: "LADDER_8"}[int(knobs["arms"])])
        fitted, aux = A.fit_ensemble(z, seed=abs(hash(str(cell_id))) % (2 ** 31),
                                     ladder=ladder)
        rows = F.extract(fitted, aux, z, basis=knobs.get("basis", "phys"),
                         cell_id=(str(cell_id), 0))
        return cell_id, rows
    except Exception as exc:                                          # noqa: BLE001
        return cell_id, f"__error__{type(exc).__name__}: {exc}"
