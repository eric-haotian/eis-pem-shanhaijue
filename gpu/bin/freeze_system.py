#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Freeze a deployable system (router + calibrator + budget thresholds).

    python3 bin/freeze_system.py --rows dev_rows.jsonl --out artifacts/my_system \
        --val-draw-seed 990101001 --val-noise-seed 990102001

The validation seed bases are RESERVED here and written into the manifest. Commit the
manifest BEFORE generating the validation panel; that ordering is what makes the later
single-pass number meaningful.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="0.32,0.50,0.60")
    ap.add_argument("--val-draw-seed", type=int, required=True)
    ap.add_argument("--val-noise-seed", type=int, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r["basis"] == "phys"]
    for r in rows:
        r["cell"] = tuple(r["cell"])
    arm_names = sorted({r["arm"] for r in rows})
    budgets = [float(x) for x in a.budgets.split(",")]
    man = pipeline.train_and_freeze(rows, arm_names, budgets, a.out,
                                    a.val_draw_seed, a.val_noise_seed, note=a.note)
    print(json.dumps(man, indent=1))

if __name__ == "__main__":
    main()
