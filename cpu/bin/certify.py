#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Deployment: certify the parameters of one or more measured/simulated spectra.

    python3 bin/certify.py --system artifacts/my_system --budget 0.60 --demo

Each certified claim carries a complete decision record: the winning arm, the calibrated
probability, the applied threshold, and the estimate. Coordinates below the threshold are
ABSTAINED, which is the safe default.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import arms as A, features as F, panel as P, pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--demo", action="store_true",
                    help="certify one freshly simulated cell (self-test)")
    ap.add_argument("--spectrum", help="npz with arrays 'freq' and 'z' (stacked)")
    ap.add_argument("--out", default="certified_claims.json")
    a = ap.parse_args()

    path = a.system
    if os.path.isdir(path):
        path = os.path.join(path, "frozen_system.joblib")
    frozen = pipeline.load_frozen(path)

    if a.demo:
        truth = P.draw_truth(0, seed_base=777000001)
        z = P.simulate_observation(truth, seed=777100001)
        fitted, aux = A.fit_ensemble(z, seed=777100001, ladder=A.LADDER_8)
        rows = F.extract(fitted, aux, z, truth=truth, cell_id=("demo", 0))
    else:
        d = np.load(a.spectrum)
        z = d["z"]
        fitted, aux = A.fit_ensemble(z, seed=0, ladder=A.LADDER_8)
        rows = F.extract(fitted, aux, z, cell_id=("measured", 0))

    certified, _ = pipeline.apply_frozen(frozen, rows, a.budget)
    claims = [c for c in certified if c["certified"]]
    print(f"certified {len(claims)} of {len(certified)} coordinates at budget {a.budget}")
    for c in sorted(claims, key=lambda r: -r["p_certified"])[:15]:
        mark = "" if "y" not in c else ("  [correct]" if c["y"] else "  [FALSE]")
        print(f"  {c['coord']:22s} p={c['p_certified']:.3f}  arm={c['arm']:8s}{mark}")
    json.dump([{k: v for k, v in c.items() if k != "cell"} for c in claims],
              open(a.out, "w"), indent=1, default=str)
    print(f"full records -> {a.out}")

if __name__ == "__main__":
    main()
