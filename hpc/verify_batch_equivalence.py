#!/usr/bin/env python3
"""Verify that larger velocity batches preserve the DGFS solution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np


def datasets(path: Path, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for name in sorted(h5):
            if name.startswith(prefix):
                value = np.asarray(h5[name][()])
                if not np.all(np.isfinite(value)):
                    raise RuntimeError(f"non-finite values in {path}:{name}")
                out[name] = value.astype(np.float64, copy=False)
    if not out:
        raise RuntimeError(f"no {prefix} datasets in {path}")
    return out


def compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> tuple[float, float]:
    if reference.keys() != candidate.keys():
        raise RuntimeError("dataset names differ")
    num = den = linf = scale = 0.0
    for name in reference:
        a, b = reference[name], candidate[name]
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch for {name}: {a.shape} != {b.shape}")
        delta = b - a
        num += float(np.vdot(delta.ravel(), delta.ravel()).real)
        den += float(np.vdot(a.ravel(), a.ravel()).real)
        linf = max(linf, float(np.max(np.abs(delta))))
        scale = max(scale, float(np.max(np.abs(a))))
    return math.sqrt(num / max(den, np.finfo(float).tiny)), linf / max(scale, np.finfo(float).tiny)


def last_residual(path: Path) -> tuple[float, float, float]:
    last = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                values = float(row["t"]), float(row["f"]), float(row["f_normalized"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(x) for x in values):
                last = values
    if last is None:
        raise RuntimeError(f"no finite residual in {path}")
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--batches", type=int, nargs="+", required=True)
    ap.add_argument("--rtol", type=float, default=2e-12)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    baseline = args.root / "batch64"
    ref_dist = datasets(baseline / "dist-0.1.frfss", "soln_")
    ref_bulk = datasets(baseline / "bulk-0.1.frfss", "moments_")
    t0, raw0, norm0 = last_residual(baseline / "kinetic_residual.csv")

    rows = []
    for batch in args.batches:
        run = args.root / f"batch{batch}"
        status_file = run / "RUN_STATUS.txt"
        status = status_file.read_text().strip() if status_file.exists() else "missing"
        row = {"batch": batch, "run_status": status, "passed": False}
        if status == "complete":
            dist_l2, dist_linf = compare(ref_dist, datasets(run / "dist-0.1.frfss", "soln_"))
            bulk_l2, bulk_linf = compare(ref_bulk, datasets(run / "bulk-0.1.frfss", "moments_"))
            t, raw, norm = last_residual(run / "kinetic_residual.csv")
            wall = float((run / "WALLTIME_SECONDS.txt").read_text())
            row.update(wall_seconds=wall, dist_rel_l2=dist_l2, dist_rel_linf=dist_linf,
                       bulk_rel_l2=bulk_l2, bulk_rel_linf=bulk_linf,
                       final_time=t, residual_raw=raw, residual_normalized=norm,
                       residual_raw_abs_delta=abs(raw - raw0),
                       residual_normalized_abs_delta=abs(norm - norm0))
            row["passed"] = (abs(t - t0) < 5e-13 and dist_l2 <= args.rtol and
                             dist_linf <= 10 * args.rtol and bulk_l2 <= args.rtol and
                             bulk_linf <= 10 * args.rtol)
        rows.append(row)

    passing = [r for r in rows if r["passed"]]
    if not passing or not any(r["batch"] == 64 for r in passing):
        raise SystemExit("BATCH_EQUIVALENCE_BASELINE_FAILED")
    baseline_wall = next(r["wall_seconds"] for r in passing if r["batch"] == 64)
    for row in rows:
        if row.get("wall_seconds"):
            row["speedup_vs_64"] = baseline_wall / row["wall_seconds"]
    best = min(passing, key=lambda r: r["wall_seconds"])
    report = {"gate": "PASS", "relative_tolerance": args.rtol,
              "recommended_batch": best["batch"],
              "recommended_speedup_vs_64": best["speedup_vs_64"], "runs": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.output.parent / "RECOMMENDATION.env").write_text(
        f"DGFS_FAST_BATCH={best['batch']}\nDGFS_MEASURED_SPEEDUP={best['speedup_vs_64']:.12g}\n"
    )
    print("JCP14_BATCH_EQUIVALENCE_PASS")
    print(f"DGFS_FAST_BATCH={best['batch']}")
    print(f"DGFS_MEASURED_SPEEDUP={best['speedup_vs_64']:.6f}")


if __name__ == "__main__":
    main()
