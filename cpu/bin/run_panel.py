#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Fit the estimator ensemble on a synthetic panel and write feature rows.

    python3 bin/run_panel.py --truths 30 --noise 2 --out dev_rows.jsonl

Use --draw-seed/--noise-seed to control the panel identity. A validation panel MUST use
the seed bases recorded in the freeze manifest and must be generated only after freezing.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import arms as A, pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truths", type=int, default=30)
    ap.add_argument("--noise", type=int, default=2)
    ap.add_argument("--draw-seed", type=int, default=20260101)
    ap.add_argument("--noise-seed", type=int, default=20260201)
    ap.add_argument("--arms", choices=["3", "6", "8"], default="8")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="rows.jsonl")
    a = ap.parse_args()

    ladder = {"3": A.LADDER_3, "6": A.LADDER_6, "8": A.LADDER_8}[a.arms]
    n_arms = len(ladder) + 2
    print(f"ensemble: {n_arms} arms (lambda ladder {ladder} + multistart + data-only)")
    print(f"panel: {a.truths} truths x {a.noise} noise = {a.truths*a.noise} cells, "
          f"{a.procs} processes")
    t0 = time.time()
    pool = Pool(a.procs) if a.procs > 1 else None
    try:
        rows = pipeline.build_panel(a.truths, a.noise, a.draw_seed, a.noise_seed,
                                    ladder=ladder, pool=pool)
    finally:
        if pool:
            pool.close(); pool.join()
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
