#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Fast end-to-end self-test on a deliberately small grid.

Exercises every stage — forward model, estimator ensemble, ground-truth-free features,
routing, calibration, budget control, freezing, and single-pass validation — on a
reduced acquisition grid with a small iteration budget, so the whole pipeline runs in
a couple of minutes instead of hours.

    python3 bin/selftest.py

This verifies that the installation is correct. It does NOT reproduce the paper's
numbers: those require the full grid and panel (see README section 3).
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shanhaijue import panel as P

# small grid + small iteration budget: correctness check, not a performance run
P.configure_grid(temperatures=(288.15, 308.15), socs=(0.3, 0.7),
                 frequencies=np.logspace(-2, 4, 24))

from shanhaijue import arms as A, features as F, pipeline, router as R  # noqa: E402

A.MAX_NFEV = 30
A.N_STARTS = 2
A.N_WIDE = 1


def main():
    t0 = time.time()
    ok = True

    print("[1/6] forward model")
    truth = P.draw_truth(0, seed_base=424242001)
    z = P.simulate_observation(truth, seed=424243001)
    assert z.shape == (P.NFREQ,) and np.isfinite(z).all()
    print(f"      {len(P.CONDITIONS)} conditions x {P.FREQ.size} f = {P.NFREQ} points  OK")

    print("[2/6] estimator ensemble")
    fitted, aux = A.fit_ensemble(z, seed=424243001, ladder=A.LADDER_3)
    assert len(fitted) == len(A.LADDER_3) + 2
    print(f"      arms: {sorted(fitted)}  OK")

    print("[3/6] ground-truth-free features")
    rows = F.extract(fitted, aux, z, truth=truth, cell_id=(0, 0))
    assert len(rows) == len(fitted) * P.NDIM
    missing = [f for f in F.FEATURES if rows[0].get(f) is None and f != "boundary_prox"]
    assert not missing, f"missing features: {missing}"
    print(f"      {len(rows)} rows x {len(F.FEATURES)} features  OK")

    print("[4/6] panel + routing (4 cells)")
    panel_rows = pipeline.build_panel(2, 2, 424242001, 424243001,
                                      ladder=A.LADDER_3, progress=False)
    for r in panel_rows:
        r["cell"] = tuple(r["cell"])
    arm_names = sorted({r["arm"] for r in panel_rows})
    report, routed = pipeline.evaluate_routing(panel_rows, arm_names, n_folds=2)
    assert 0 <= report["routed_recovery"] <= P.NDIM
    print(f"      routed recovery {report['routed_recovery']:.1f}/{P.NDIM} "
          f"over {report['n_clusters']} clusters  OK")

    print("[5/6] freeze (router + calibrator + budget thresholds)")
    with tempfile.TemporaryDirectory() as tmp:
        man = pipeline.train_and_freeze(panel_rows, arm_names, [0.5, 1.0], tmp,
                                        val_draw_seed=555000001, val_noise_seed=555100001,
                                        n_folds=2, note="selftest")
        assert man["artifact_sha256"] and man["thresholds"]
        print(f"      thresholds {man['thresholds']}  OK")

        print("[6/6] single-pass validation on a fresh panel")
        val_rows = pipeline.build_panel(2, 1, man["validation_seed_bases"]["draw"],
                                        man["validation_seed_bases"]["noise"],
                                        ladder=A.LADDER_3, progress=False)
        for r in val_rows:
            r["cell"] = tuple(r["cell"])
        frozen = pipeline.load_frozen(os.path.join(tmp, "frozen_system.joblib"))
        res = pipeline.validation_report(frozen, val_rows)
        assert res["rungs"], "no rungs reported"
        print(f"      recovery {res['routed_recovery']:.1f}/{P.NDIM}; rungs:")
        for b, r in res["rungs"].items():
            print(f"        budget {b}: {r['correct_per_cell']:.2f} correct / "
                  f"{r['false_per_cell']:.2f} false, precision {r['realized_precision']:.3f}")

    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
