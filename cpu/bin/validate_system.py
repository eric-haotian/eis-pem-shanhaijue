#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Single-pass validation of a frozen system on a never-before-generated panel.

    python3 bin/validate_system.py --system artifacts/my_system --rows val_rows.jsonl

Nothing is re-trained or re-thresholded. Run this ONCE per validation panel.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="directory or .joblib path")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="validation_result.json")
    a = ap.parse_args()

    path = a.system
    if os.path.isdir(path):
        path = os.path.join(path, "frozen_system.joblib")
    frozen = pipeline.load_frozen(path)
    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r["basis"] == "phys"]
    for r in rows:
        r["cell"] = tuple(r["cell"])
    res = pipeline.validation_report(frozen, rows)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
