#!/usr/bin/env python3
"""Gate a two-GPU continuation against the same one-GPU continuation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def arrays(path: Path, prefix: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        out = {key: np.asarray(h5[key][...], dtype=np.float64)
               for key in h5 if key.startswith(prefix)}
    if not out or any(not np.all(np.isfinite(value)) for value in out.values()):
        raise RuntimeError(f"invalid datasets in {path}")
    return out


def error(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> tuple[float, float]:
    if reference.keys() != candidate.keys():
        raise RuntimeError(f"dataset mismatch: {reference.keys()} != {candidate.keys()}")
    numerator = denominator = linf = scale = 0.0
    for key in reference:
        a, b = reference[key], candidate[key]
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch for {key}: {a.shape} != {b.shape}")
        delta = b - a
        numerator += float(np.vdot(delta.ravel(), delta.ravel()).real)
        denominator += float(np.vdot(a.ravel(), a.ravel()).real)
        linf = max(linf, float(np.max(np.abs(delta))))
        scale = max(scale, float(np.max(np.abs(a))))
    tiny = np.finfo(float).tiny
    return math.sqrt(numerator / max(denominator, tiny)), linf / max(scale, tiny)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial-dist", type=Path, required=True)
    ap.add_argument("--serial-bulk", type=Path, required=True)
    ap.add_argument("--mpi-dist", type=Path, required=True)
    ap.add_argument("--mpi-bulk", type=Path, required=True)
    ap.add_argument("--serial-wall", type=Path, required=True)
    ap.add_argument("--mpi-wall", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rtol-l2", type=float, default=5e-11)
    ap.add_argument("--rtol-linf", type=float, default=5e-10)
    ap.add_argument("--minimum-speedup", type=float, default=1.10)
    args = ap.parse_args()

    dist_l2, dist_linf = error(arrays(args.serial_dist, "soln_"), arrays(args.mpi_dist, "soln_"))
    bulk_l2, bulk_linf = error(arrays(args.serial_bulk, "moments_"), arrays(args.mpi_bulk, "moments_"))
    serial_wall = float(args.serial_wall.read_text())
    mpi_wall = float(args.mpi_wall.read_text())
    speedup = serial_wall / mpi_wall
    equivalent = (dist_l2 <= args.rtol_l2 and dist_linf <= args.rtol_linf and
                  bulk_l2 <= args.rtol_l2 and bulk_linf <= args.rtol_linf)
    profitable = speedup >= args.minimum_speedup
    report = {
        "equivalent": equivalent,
        "profitable": profitable,
        "production_gate": equivalent and profitable,
        "serial_wall_seconds": serial_wall,
        "mpi_wall_seconds": mpi_wall,
        "speedup_2gpu_vs_1gpu": speedup,
        "dist_rel_l2": dist_l2,
        "dist_rel_linf": dist_linf,
        "bulk_rel_l2": bulk_l2,
        "bulk_rel_linf": bulk_linf,
        "rtol_l2": args.rtol_l2,
        "rtol_linf": args.rtol_linf,
        "minimum_speedup": args.minimum_speedup,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"JCP14_MPI_EQUIVALENT={'yes' if equivalent else 'no'}")
    print(f"JCP14_MPI_SPEEDUP={speedup:.6f}")
    print(f"JCP14_MPI_PROFITABLE={'yes' if profitable else 'no'}")
    if not equivalent:
        raise SystemExit(2)
    if not profitable:
        raise SystemExit(3)
    print("JCP14_MPI_PRODUCTION_GATE_PASS")


if __name__ == "__main__":
    main()
