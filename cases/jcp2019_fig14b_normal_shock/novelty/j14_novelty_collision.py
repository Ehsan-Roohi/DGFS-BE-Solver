#!/usr/bin/env python3
"""Audit raw and fplus-projected DGFS collision invariants at every DG point.

This diagnostic evaluates Q(f,f) twice at each spatial solution point: the
unaltered fast-spectral result and the same result followed by the weighted
five-moment projection w=max(f,0).  The projection acts on Q, not on f; it is
therefore a conservative collision correction, not a positivity limiter.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np

NAMES = ("mass", "momentum_x", "momentum_y", "momentum_z", "energy")


def invariants(q, cv, cw):
    q = np.asarray(q, dtype=float)
    B = np.vstack((np.ones(q.size), cv, 0.5*np.sum(cv*cv, axis=0)))
    signed = cw*(B @ q)
    scale = cw*(np.abs(B) @ np.abs(q))
    defect = np.abs(signed)/np.maximum(scale, np.finfo(float).tiny)
    return {
        "signed": dict(zip(NAMES, map(float, signed))),
        "scale": dict(zip(NAMES, map(float, scale))),
        "relative_cancellation_defect": dict(zip(NAMES, map(float, defect))),
        "max_relative_cancellation_defect": float(np.max(defect)),
    }


def load_snapshot(snapshot, mesh):
    with h5py.File(snapshot, "r") as h5:
        f = h5["soln_line_p0"][()]
    with h5py.File(mesh, "r") as h5:
        vertices = h5["spt_line_p0"][()]
    if f.shape[0] != 3 or vertices.shape[1] != f.shape[2]:
        raise ValueError("expected three GLL points and matching line mesh")
    left = vertices[:, :, 0].min(axis=0)
    right = vertices[:, :, 0].max(axis=0)
    x = np.vstack((left, 0.5*(left + right), right))
    return f, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")

    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()

    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.projection import GPUConservativeProjector
    from frfs.solvers.dgfs.scattering import DGFSScatteringModel
    from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
    from frfs.util import subclass_where
    import pycuda.driver as cuda

    cfg = Inifile.load(str(args.config))
    backend = get_backend("cuda", cfg)
    vm = DGFSVelocityMesh(backend, cfg)
    scattering_cls = subclass_where(
        DGFSScatteringModel,
        scattering_model=cfg.get("scattering-model", "type"),
    )
    t0 = time.perf_counter()
    scattering = scattering_cls(backend, cfg, vm)
    projector = GPUConservativeProjector(backend, vm, "fplus", "device")
    cuda.Context.synchronize()
    precompute = time.perf_counter() - t0

    soln, x = load_snapshot(args.snapshot, args.mesh)
    if soln.shape[1] != vm.vsize():
        raise ValueError(
            f"snapshot has {soln.shape[1]} velocities; config expects {vm.vsize()}"
        )
    cv, cw = np.asarray(vm.cv(), dtype=float), float(vm.cw())
    shape = (1, vm.vsize(), 1)
    zeros = np.zeros(shape, dtype=backend.fpdtype)
    d_f = backend.matrix(shape, zeros, tags={"align"})
    d_q = backend.matrix(shape, zeros, tags={"align"})
    rows = []

    for elem in range(soln.shape[2]):
        for upt in range(soln.shape[0]):
            f = np.asarray(soln[upt, :, elem], dtype=backend.fpdtype)
            d_f.set(f.reshape(shape))

            d_q.set(zeros)
            scattering.fs(d_f, d_q, 0, 0)
            cuda.Context.synchronize()
            raw_times = []
            for _ in range(args.repeats):
                d_q.set(zeros)
                start = time.perf_counter()
                scattering.fs(d_f, d_q, 0, 0)
                cuda.Context.synchronize()
                raw_times.append(time.perf_counter() - start)
            q_raw = d_q.get()[0, :, 0].copy()

            d_q.set(zeros)
            scattering.fs(d_f, d_q, 0, 0)
            projector.apply(d_f, d_q, 0, 0)
            cuda.Context.synchronize()
            plus_times = []
            for _ in range(args.repeats):
                d_q.set(zeros)
                start = time.perf_counter()
                scattering.fs(d_f, d_q, 0, 0)
                projector.apply(d_f, d_q, 0, 0)
                cuda.Context.synchronize()
                plus_times.append(time.perf_counter() - start)
            q_plus = d_q.get()[0, :, 0].copy()

            raw_inv = invariants(q_raw, cv, cw)
            plus_inv = invariants(q_plus, cv, cw)
            neg = np.maximum(-f, 0.0)
            pos = np.maximum(f, 0.0)
            corr = np.linalg.norm(q_plus - q_raw)/max(
                np.linalg.norm(q_raw), np.finfo(float).tiny
            )
            row = {
                "solution_point": upt,
                "element": elem,
                "x_nondim": float(x[upt, elem]),
                "min_f": float(np.min(f)),
                "negative_mass_fraction": float(
                    cw*np.sum(neg)/max(cw*np.sum(pos), np.finfo(float).tiny)
                ),
                "raw": raw_inv,
                "fplus": plus_inv,
                "relative_correction_l2": float(corr),
                "raw_time_ms": float(1e3*np.median(raw_times)),
                "fplus_time_ms": float(1e3*np.median(plus_times)),
                "overhead_ratio": float(
                    np.median(plus_times)/max(np.median(raw_times), 1e-300)
                ),
            }
            rows.append(row)
            print(
                "J14NOV_COLLISION_POINT "
                f"M={vm.M()} e={elem} u={upt} "
                f"raw={raw_inv['max_relative_cancellation_defect']:.3e} "
                f"fplus={plus_inv['max_relative_cancellation_defect']:.3e} "
                f"overhead={row['overhead_ratio']:.3f}",
                flush=True,
            )

    def max_by(mode, name):
        return float(max(
            r[mode]["relative_cancellation_defect"][name] for r in rows
        ))

    raw_max = max(r["raw"]["max_relative_cancellation_defect"] for r in rows)
    plus_max = max(r["fplus"]["max_relative_cancellation_defect"] for r in rows)
    raw_t = float(np.median([r["raw_time_ms"] for r in rows]))
    plus_t = float(np.median([r["fplus_time_ms"] for r in rows]))
    summary = {
        "points": len(rows),
        "raw_max_defect": float(raw_max),
        "fplus_max_defect": float(plus_max),
        "raw_max_defect_by_invariant": {n: max_by("raw", n) for n in NAMES},
        "fplus_max_defect_by_invariant": {n: max_by("fplus", n) for n in NAMES},
        "defect_reduction": float(raw_max/max(plus_max, 1e-300)),
        "median_raw_collision_ms": raw_t,
        "median_fplus_collision_ms": plus_t,
        "median_projection_overhead_ratio": float(plus_t/max(raw_t, 1e-300)),
        "max_relative_correction_l2": float(
            max(r["relative_correction_l2"] for r in rows)
        ),
        "max_negative_mass_fraction": float(
            max(r["negative_mass_fraction"] for r in rows)
        ),
    }
    report = {
        "schema_version": 1,
        "interpretation": (
            "fplus is a weighted five-moment projection of Q using w=max(f,0); "
            "it does not assert global positivity of f"
        ),
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "collision_precompute_seconds": precompute,
        },
        "velocity_grid": {
            "Nv": int(vm.Nv()), "Nrho": int(vm.Nrho()), "M": int(vm.M()),
            "vsize": int(vm.vsize()), "cw": cw,
        },
        "summary": summary,
        "records": rows,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "M", "solution_point", "element", "x_nondim", "min_f",
        "negative_mass_fraction", "raw_max_defect", "fplus_max_defect",
        "relative_correction_l2", "raw_time_ms", "fplus_time_ms",
        "overhead_ratio",
    ]
    with args.output_csv.open("w", newline="") as stream:
        wr = csv.DictWriter(stream, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({
                "M": vm.M(), "solution_point": r["solution_point"],
                "element": r["element"], "x_nondim": r["x_nondim"],
                "min_f": r["min_f"],
                "negative_mass_fraction": r["negative_mass_fraction"],
                "raw_max_defect": r["raw"]["max_relative_cancellation_defect"],
                "fplus_max_defect": r["fplus"]["max_relative_cancellation_defect"],
                "relative_correction_l2": r["relative_correction_l2"],
                "raw_time_ms": r["raw_time_ms"],
                "fplus_time_ms": r["fplus_time_ms"],
                "overhead_ratio": r["overhead_ratio"],
            })
    print(f"J14NOV_COLLISION_COMPLETE M={vm.M()} points={len(rows)}")
    print("J14NOV_COLLISION_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

