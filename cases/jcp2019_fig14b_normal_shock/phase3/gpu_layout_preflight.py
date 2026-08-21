#!/usr/bin/env python3
"""Verify the projection kernel on the solver's real (upt, velocity, element) layout."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from conservative_projection import ConservativeProjector, GPUConservativeProjector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=("fplus", "fplus-transverse"), default="fplus")
    args = ap.parse_args()

    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
    import pycuda.driver as cuda

    cfg = Inifile.load(str(args.config.resolve()))
    cfg.set("constants", "Nrho", "16")
    cfg.set("spherical-design-rule", "M", "16")
    backend = get_backend("cuda", cfg)
    vm = DGFSVelocityMesh(backend, cfg)
    shape = (3, vm.vsize(), 8)
    rng = np.random.default_rng(20260819)
    cv = np.asarray(vm.cv(), dtype=float)
    base = np.exp(-np.sum(cv * cv, axis=0))
    f = np.empty(shape, dtype=backend.fpdtype)
    q = rng.normal(scale=1.0e-4, size=shape).astype(backend.fpdtype)
    for upt in range(shape[0]):
        for elem in range(shape[2]):
            f[upt, :, elem] = base * (1.0 + 0.01 * upt + 0.001 * elem)

    d_f = backend.matrix(shape, f, tags={"align"})
    d_q = backend.matrix(shape, q, tags={"align"})
    elem, upt = 5, 1
    cpu = ConservativeProjector(cv, float(vm.cw()))
    if args.mode == "fplus":
        ref = cpu.project(q[upt, :, elem], f[upt, :, elem], "fplus").Qc
    else:
        q0, f0 = q[upt, :, elem], f[upt, :, elem]
        moments = cpu.moments(q0)
        weight = np.maximum(f0, 0.0)
        gram = float(vm.cw()) * ((cpu.B * weight[None, :]) @ cpu.B.T)
        target = moments * np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        lam = np.linalg.solve(gram, target)
        ref = q0 - weight * (cpu.B.T @ lam)
    projector = GPUConservativeProjector(backend, vm, args.mode, "device")
    projector.apply(d_f, d_q, elem, upt)
    cuda.Context.synchronize()
    got = d_q.get()

    rel = float(np.max(np.abs(got[upt, :, elem] - ref)) /
                max(np.max(np.abs(ref)), np.finfo(float).tiny))
    untouched = q.copy()
    untouched[upt, :, elem] = got[upt, :, elem]
    collateral = float(np.max(np.abs(got - untouched)))
    print(f"P3B_LAYOUT_MODE={args.mode}")
    print(f"P3B_LAYOUT_REL_LINF={rel:.16e}")
    print(f"P3B_LAYOUT_COLLATERAL_LINF={collateral:.16e}")
    if rel >= 1.0e-10 or collateral != 0.0:
        raise SystemExit("P3B_GPU_LAYOUT_PREFLIGHT_FAIL")
    print("P3B_GPU_LAYOUT_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
