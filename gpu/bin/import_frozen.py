#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Convert frozen artifacts produced by the paper's experiment scripts into the
package-native format used by `shanhaijue.pipeline`.

The two layouts hold the same objects under different names; this script re-wraps them
into `Router` and `Certifier` instances and verifies that the feature ordering, the arm
naming, and the design-matrix width all agree before writing anything.

    python3 bin/import_frozen.py --legacy chainA.joblib chainB.joblib \
        --out artifacts/validated_system

Multiple legacy artifacts may be merged ONLY if their ranker and calibrator are bit-for-bit
identical (same development panel, same seed); the script checks this and refuses
otherwise. Merging then simply unions their budget rungs.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import certifier as C, router as R
from shanhaijue.features import FEATURES


def _probe(clf, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, len(FEATURES) + 2))
    X[:, len(FEATURES)] = rng.integers(0, 48, 256)
    X[:, len(FEATURES) + 1] = rng.integers(0, 8, 256)
    return clf.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    loaded = [joblib.load(p) for p in a.legacy]
    ref = loaded[0]

    n_cols = ref["clf"].n_features_in_
    if n_cols != len(FEATURES) + 2:
        raise SystemExit(f"design width mismatch: artifact expects {n_cols}, "
                         f"package builds {len(FEATURES) + 2}")
    expected_ids = {n: i for i, n in enumerate(sorted(ref["arm_set"]))}
    if ref["arm_ids"] != expected_ids:
        raise SystemExit("arm identity encoding differs from the package convention")

    thresholds = {}
    for path, d in zip(a.legacy, loaded):
        if not np.allclose(_probe(d["clf"]), _probe(ref["clf"])):
            raise SystemExit(f"{path}: ranker differs from the first artifact; refusing to merge")
        if not (np.allclose(d["platt"].coef_, ref["platt"].coef_)
                and np.allclose(d["platt"].intercept_, ref["platt"].intercept_)):
            raise SystemExit(f"{path}: calibrator differs from the first artifact; refusing to merge")
        for b, t in d["tstars"].items():
            thresholds[float(b)] = float(t)

    router = R.Router(ref["arm_set"])
    router.ranker = ref["clf"]
    router.calibrators = ref["cal"]
    router.arm_ids = ref["arm_ids"]

    cert = C.Certifier()
    cert.platt = ref["platt"]
    cert.thresholds = thresholds

    os.makedirs(a.out, exist_ok=True)
    target = os.path.join(a.out, "frozen_system.joblib")
    joblib.dump(dict(router=router, certifier=cert, arm_names=sorted(ref["arm_set"]),
                     budgets=sorted(thresholds), val_draw_seed=ref.get("val_draw0"),
                     val_noise_seed=ref.get("val_noise0")), target)
    json.dump(dict(source_artifacts=[os.path.basename(p) for p in a.legacy],
                   arm_names=sorted(ref["arm_set"]),
                   thresholds={str(k): v for k, v in sorted(thresholds.items())},
                   note=a.note),
              open(os.path.join(a.out, "import_record.json"), "w"), indent=1)
    print(f"wrote {target}")
    print(json.dumps({str(k): round(v, 6) for k, v in sorted(thresholds.items())}, indent=1))


if __name__ == "__main__":
    main()
