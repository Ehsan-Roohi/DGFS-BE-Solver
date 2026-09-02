#!/usr/bin/env python3
"""GPU-vs-NumPy parity audit for DGFS conservative collision projection.

Uses the same raw fast-spectral Q(f,f) at each DG solution point, then compares
GPUConservativeProjector against the double-precision ConservativeProjector
reference for euclidean and fplus weightings.  No time integration is done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TINY = np.finfo(float).tiny


def load_snapshot(path: Path):
    with h5py.File(path, "r") as h5:
        return h5["soln_line_p0"][()]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), TINY))


def rel_linf(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), TINY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=5e-12)
    args = ap.parse_args()

    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()

    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.projection import ConservativeProjector, GPUConservativeProjector
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
    scattering = scattering_cls(backend, cfg, vm)
    cpu = ConservativeProjector(np.asarray(vm.cv(), dtype=float), float(vm.cw()))
    gpu = {
        mode: GPUConservativeProjector(backend, vm, mode, "device")
        for mode in ("euclidean", "fplus")
    }

    soln = load_snapshot(args.snapshot)
    if soln.shape[0] != 3 or soln.shape[1] != vm.vsize():
        raise ValueError(f"unexpected snapshot shape {soln.shape}; vsize={vm.vsize()}")

    shape = (1, vm.vsize(), 1)
    zeros = np.zeros(shape, dtype=backend.fpdtype)
    d_f = backend.matrix(shape, zeros, tags={"align"})
    d_q = backend.matrix(shape, zeros, tags={"align"})

    records = []
    for elem in range(soln.shape[2]):
        for upt in range(soln.shape[0]):
            f = np.asarray(soln[upt, :, elem], dtype=float)
            d_f.set(f.astype(backend.fpdtype).reshape(shape))
            d_q.set(zeros)
            scattering.fs(d_f, d_q, 0, 0)
            cuda.Context.synchronize()
            qraw = np.asarray(d_q.get()[0, :, 0], dtype=float).copy()

            for mode in ("euclidean", "fplus"):
                qcpu = cpu.project(qraw, f=f, weighting=mode).Qc
                d_q.set(qraw.astype(backend.fpdtype).reshape(shape))
                gpu[mode].apply(d_f, d_q, 0, 0)
                cuda.Context.synchronize()
                qgpu = np.asarray(d_q.get()[0, :, 0], dtype=float).copy()
                l2 = rel_l2(qgpu, qcpu)
                li = rel_linf(qgpu, qcpu)
                inv_gpu = cpu.cancellation_defects(qgpu)
                rec = {
                    "element": elem,
                    "solution_point": upt,
                    "mode": mode,
                    "gpu_vs_cpu_rel_l2": l2,
                    "gpu_vs_cpu_rel_linf": li,
                    "gpu_max_invariant_defect": float(np.max(inv_gpu)),
                }
                records.append(rec)
                print(
                    f"PARITY_POINT e={elem} u={upt} mode={mode} "
                    f"relL2={l2:.3e} relLinf={li:.3e} "
                    f"defect={np.max(inv_gpu):.3e}", flush=True
                )

    summary = {}
    passed = True
    for mode in ("euclidean", "fplus"):
        r = [x for x in records if x["mode"] == mode]
        s = {
            "max_gpu_vs_cpu_rel_l2": max(x["gpu_vs_cpu_rel_l2"] for x in r),
            "max_gpu_vs_cpu_rel_linf": max(x["gpu_vs_cpu_rel_linf"] for x in r),
            "max_gpu_invariant_defect": max(x["gpu_max_invariant_defect"] for x in r),
        }
        s["pass"] = bool(
            s["max_gpu_vs_cpu_rel_l2"] <= args.tol
            and s["max_gpu_vs_cpu_rel_linf"] <= args.tol
            and s["max_gpu_invariant_defect"] <= args.tol
        )
        passed = passed and s["pass"]
        summary[mode] = s

    report = {
        "schema_version": 1,
        "purpose": "GPU vs CPU conservative projection parity",
        "tolerance": args.tol,
        "snapshot": str(args.snapshot),
        "M_omega": int(vm.M()),
        "Nv": int(vm.Nv()),
        "summary": summary,
        "records": records,
        "overall_pass": bool(passed),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PARITY_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print("GPU_CPU_PARITY=" + ("PASS" if passed else "FAIL"))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
