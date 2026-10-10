#!/usr/bin/env python3
"""
Score answer timing with the Answer Readiness Score, without running a model.

Takes a results file and reports ARS, accuracy and effective accuracy. Useful for
scoring predictions produced elsewhere, for re-scoring a saved run at a different
penalty sharpness, or for sanity-checking a benchmark's own annotations.

Input is JSON: either a list of records, or an object with a ``results`` or
``records`` key holding one. Each record needs::

    answer_time       when the model answered (seconds)
    evidence_start    t_s, when supporting evidence first appears
    evidence_end      t_e, when it stops being valid

and optionally::

    correct     bool or 0/1   -> enables accuracy and effective accuracy
    task        str           -> enables the per-task breakdown
    turn_group  str           -> multi-turn dialogues averaged before aggregating

Usage:
    python tools/evaluate_ars.py --results results.json
    python tools/evaluate_ars.py --results results.json --gamma-early 4 --gamma-late 2
    python tools/evaluate_ars.py --results results.json --json > scores.json
"""

import argparse
import json
import sys

from lavis.metrics.ars import compute_ars, format_ars_report

REQUIRED = ("answer_time", "evidence_start", "evidence_end")


def load_records(path):
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        for key in ("results", "records", "predictions"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        sys.exit(f"{path}: expected a list of records, or an object with a "
                 f"'results'/'records' key holding one")
    return payload


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", required=True)
    ap.add_argument("--gamma-early", type=float, default=6.0,
                    help="early-penalty sharpness (paper: 6)")
    ap.add_argument("--gamma-late", type=float, default=1.0,
                    help="late-penalty decay slope (paper: 1)")
    ap.add_argument("--tau", type=float, default=None,
                    help="median evidence duration; computed from the data if omitted")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    records = load_records(args.results)
    if not records:
        sys.exit(f"{args.results}: no records")

    missing = [k for k in REQUIRED if k not in records[0]]
    if missing:
        sys.exit(f"{args.results}: records are missing required field(s) "
                 f"{missing}; present keys are {sorted(records[0])}")

    summary = compute_ars(
        records, gamma_e=args.gamma_early, gamma_l=args.gamma_late, tau=args.tau
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_ars_report(summary))


if __name__ == "__main__":
    main()
