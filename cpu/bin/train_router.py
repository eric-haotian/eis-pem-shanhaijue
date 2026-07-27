#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Train the router on a development panel and report out-of-fold routed recovery.

    python3 bin/train_router.py --rows dev_rows.jsonl
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shanhaijue import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="routing_report.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r["basis"] == "phys"]
    for r in rows:
        r["cell"] = tuple(r["cell"])
    arm_names = sorted({r["arm"] for r in rows})
    print(f"arms: {arm_names}")
    report, _ = pipeline.evaluate_routing(rows, arm_names, n_folds=a.folds)
    print(json.dumps(report, indent=1))
    json.dump(report, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
