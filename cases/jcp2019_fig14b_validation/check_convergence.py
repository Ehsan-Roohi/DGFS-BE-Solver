#!/usr/bin/env python3
"""Evaluate the paper-normalized residual on full nominal time steps only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--segment-start", type=float, required=True)
    ap.add_argument("--segment-end", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=2e-5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    candidates = []
    previous = None
    with args.csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                time = float(row["t"])
                raw = float(row["f"])
                normalized = float(row["f_normalized"])
            except (KeyError, TypeError, ValueError):
                continue
            step = time - previous if previous is not None else math.nan
            previous = time
            if (args.segment_start < time <= args.segment_end and
                    0.5 * args.dt <= step <= 1.5 * args.dt and
                    math.isfinite(raw) and math.isfinite(normalized)):
                candidates.append((time, raw, normalized, step))
    if not candidates:
        raise SystemExit("NO_REGULAR_RESIDUAL_SAMPLE_IN_SEGMENT")
    time, raw, normalized, step = max(candidates)
    status = {
        "converged": normalized < args.threshold,
        "threshold": args.threshold,
        "sample_time": time,
        "raw_residual": raw,
        "paper_normalized_residual": normalized,
        "accepted_step_size": step,
        "segment_start": args.segment_start,
        "segment_end": args.segment_end,
        "nominal_dt": args.dt,
    }
    args.output.write_text(json.dumps(status, indent=2) + "\n")
    print(f"LAST_REGULAR_RESIDUAL_TIME={time:.12g}")
    print(f"FINAL_PAPER_NORMALIZED_RESIDUAL={normalized:.12e}")
    print(f"PAPER_CONVERGED={'yes' if status['converged'] else 'no'}")
    raise SystemExit(0 if status["converged"] else 2)


if __name__ == "__main__":
    main()

