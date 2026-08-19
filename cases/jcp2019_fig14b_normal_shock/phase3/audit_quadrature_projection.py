#!/usr/bin/env python3
"""Phase-3 GPU audit: quadrature sensitivity + five-moment conservative projection
of the DGFS ``vhs-gll`` fast-spectral collision term at all DG solution points of a
saved full-distribution snapshot.  No time integration is performed.

For every (Nrho, M) in ``--quadratures`` (default: baseline 32:6, proposed 16:16,
reference 16:24) and every DG point it records
  * raw Q(f,f): five signed moments, cancellation- and state-normalised defects,
    artificial u_z source rate Q_z/rho, negativity of the Euler update f + dt*Q;
  * projected Q_c for the weightings euclidean / f / fplus (GPU kernels, verified
    against the numpy reference) and maxwellian (numpy only): post-projection
    defects, ||dQ||/||Q||, negativity of f + dt*Q_c versus f + dt*Q, the positivity
    indicator dt*max|B^T lambda|;
  * synchronised GPU timings of the collision call and of the projection call.
It also evaluates Q on a resting and on a drifting (shock-centre) Maxwellian for
each quadrature, then applies the pass/fail gates (see p3_analysis.DEFAULT_GATES).

Run on one GPU node:
    python audit_quadrature_projection.py --config case.ini --snapshot snapshot.frfss \
        --mesh mesh.frfsm --quadratures 32:6,16:16,16:24
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from conservative_projection import (GPU_WEIGHTINGS, INVARIANT_NAMES, ConservativeProjector,
                                     GPUConservativeProjector, bulk_moments, maxwellian)
from p3_analysis import (DEFAULT_GATES, analyse_point, evaluate_gates, print_summary_table,
                         summarise_quadrature, write_points_csv)


def load_snapshot(snapshot: Path, mesh: Path):
    import h5py

    with h5py.File(snapshot, "r") as h5:
        soln = h5["soln_line_p0"][()]
    with h5py.File(mesh, "r") as h5:
        vertices = h5["spt_line_p0"][()]
    if soln.shape[0] != 3:
        raise ValueError("audit expects three GLL points per line element")
    if vertices.shape[1] != soln.shape[2]:
        raise ValueError("snapshot and mesh element counts differ")
    left = vertices[:, :, 0].min(axis=0)
    right = vertices[:, :, 0].max(axis=0)
    x = np.vstack((left, 0.5 * (left + right), right))
    return soln, x


def select_points(soln, x, max_points):
    pts = [(u, e) for e in range(soln.shape[2]) for u in range(soln.shape[0])]
    pts.sort(key=lambda p: (abs(float(x[p])), p[1], p[0]))
    return pts[:max_points] if max_points > 0 else pts


def parse_quads(spec: str):
    out = []
    for item in spec.split(","):
        nrho, m = item.split(":")
        out.append((int(nrho), int(m)))
    return out


def timing_stats(samples):
    s = [1.0e3 * float(v) for v in samples]
    return {"samples": s, "minimum": min(s), "median": float(np.median(s)),
            "mean": float(np.mean(s)), "maximum": max(s)}


def gpu_identity():
    import pycuda.driver as cuda
    dev = cuda.Context.get_device()
    return {"name": dev.name(), "compute_capability": list(dev.compute_capability()),
            "total_memory_bytes": int(dev.total_memory()),
            "driver_version": int(cuda.get_driver_version())}


def run(args) -> dict:
    import mpi4py.rc
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.scattering import DGFSScatteringModel
    from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
    from frfs.util import subclass_where
    import pycuda.driver as cuda

    cfg = Inifile.load(str(args.config))
    backend = get_backend("cuda", cfg)
    scattering_name = cfg.get("scattering-model", "type")
    scattering_cls = subclass_where(DGFSScatteringModel, scattering_model=scattering_name)

    soln, x = load_snapshot(args.snapshot, args.mesh)
    pts = select_points(soln, x, args.max_points)
    gates = dict(DEFAULT_GATES)
    gates.update({"post_defect": args.gate_post_defect, "raw_proposed": args.gate_raw_proposed,
                  "rel_correction": args.gate_rel_correction, "negmass_ratio": args.gate_negmass_ratio,
                  "newly_negative": args.gate_newly_negative, "min_ratio": args.gate_min_ratio,
                  "z_reduction": args.gate_z_reduction,
                  "projection_cost_fraction": args.gate_projection_cost_fraction})
    roles = ["baseline", "proposed", "reference"]
    weightings = [w for w in GPU_WEIGHTINGS if w not in args.skip_gpu_weighting]
    quads = []

    for i, (nrho, m) in enumerate(parse_quads(args.quadratures)):
        name = f"Nrho{nrho}_M{m}"
        print(f"\n================ {name} ================", flush=True)
        cfg.set("constants", "Nrho", str(nrho))
        cfg.set("spherical-design-rule", "M", str(m))
        vm = DGFSVelocityMesh(backend, cfg)
        if soln.shape[1] != vm.vsize():
            raise ValueError("snapshot velocity dimension does not match configuration")
        cv, cw = np.asarray(vm.cv(), dtype=float), float(vm.cw())
        t0 = time.perf_counter()
        scattering = scattering_cls(backend, cfg, vm)
        cuda.Context.synchronize()
        precompute_s = time.perf_counter() - t0
        P = ConservativeProjector(cv, cw)
        t0 = time.perf_counter()
        projectors = {w: GPUConservativeProjector(backend, vm, w, args.gpu_solve)
                      for w in weightings}
        cuda.Context.synchronize()
        proj_setup_s = time.perf_counter() - t0
        print(f"[{name}] collision precompute {precompute_s:.3f}s, projector setup {proj_setup_s:.3f}s, "
              f"cond(G)={P.cond_unscaled:.1f} -> scaled {P.cond_scaled:.2f}", flush=True)

        shape = (1, vm.vsize(), 1)
        zeros = np.zeros(shape, dtype=backend.fpdtype)
        d_input = backend.matrix(shape, zeros, tags={"align"})
        d_output = backend.matrix(shape, zeros, tags={"align"})

        def collide(f: np.ndarray, repeats: int):
            d_input.set(np.asarray(f, dtype=backend.fpdtype).reshape(shape))
            d_output.set(zeros)
            scattering.fs(d_input, d_output, 0, 0)      # warm-up (plans, kernels)
            cuda.Context.synchronize()
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                scattering.fs(d_input, d_output, 0, 0)
                cuda.Context.synchronize()
                samples.append(time.perf_counter() - t0)
            return d_output.get()[0, :, 0].astype(float).copy(), samples

        def project_on_gpu(q_raw: np.ndarray, repeats: int) -> dict:
            packed = np.asarray(q_raw, dtype=backend.fpdtype).reshape(shape)
            out = {}
            for w, proj in projectors.items():
                d_output.set(packed)
                proj.apply(d_input, d_output, 0, 0)     # warm-up
                cuda.Context.synchronize()
                samples = []
                for _ in range(repeats):
                    d_output.set(packed)
                    cuda.Context.synchronize()
                    t0 = time.perf_counter()
                    proj.apply(d_input, d_output, 0, 0)
                    cuda.Context.synchronize()
                    samples.append(time.perf_counter() - t0)
                qc = d_output.get()[0, :, 0].astype(float).copy()
                lam = proj.fetch_lambda()
                out[w] = {"Qc": qc, "timing_ms": timing_stats(samples), "lam": lam}
            return out

        records = []
        for upt, elem in pts:
            f = soln[upt, :, elem]
            q_raw, samples = collide(f, args.repeats)
            gpu_results = project_on_gpu(q_raw, args.projection_repeats)
            rec = analyse_point(f"snapshot_u{upt}_e{elem}", f, q_raw, P, args.dt, cv, cw,
                                float(x[upt, elem]), upt, elem, gpu_results, timing_stats(samples))
            records.append(rec)
            op = rec["operators"]
            eg = op["euclidean"].get("gpu", {})
            fg = op["fplus"].get("gpu", {})
            print(f"P3_POINT [{name}] {rec['label']:15s} x={rec['x_nondim']:+.4f} "
                  f"rawCanc={max(op['raw']['cancellation_defect'].values()):.3e} "
                  f"rawState={max(op['raw']['state_defect'].values()):.3e} "
                  f"duz/dt={op['raw']['uz_source_rate']:+.3e} | "
                  f"euc: postGPU={max(eg.get('cancellation_defect', {'x': float('nan')}).values()):.1e} "
                  f"gpu-np={eg.get('max_abs_diff_vs_numpy', float('nan')):.1e} "
                  f"relL2={op['euclidean']['rel_correction_l2']:.2e} "
                  f"negratio={op['euclidean']['euler_update']['negative_mass_ratio_vs_reference']:.3f} | "
                  f"fplus: relL2={op['fplus']['rel_correction_l2']:.2e} "
                  f"negratio={op['fplus']['euler_update']['negative_mass_ratio_vs_reference']:.3f} | "
                  f"coll={rec['collision_timing_ms']['median']:.3f}ms "
                  f"proj={eg.get('timing_ms', {}).get('median', float('nan')):.3f}/"
                  f"{fg.get('timing_ms', {}).get('median', float('nan')):.3f}ms", flush=True)

        controls = {}
        if not args.skip_maxwellian:
            rc, uc, Tc = bulk_moments(soln[2, :, 3], cv, cw)       # shock-centre state (x = 0)
            cases = {"maxwellian_rest": maxwellian(cv, 1.0, np.zeros(3), 1.0),
                     "maxwellian_shockcentre_drift": maxwellian(cv, rc, uc * np.array([1.0, 0.0, 0.0]), Tc)}
            for lab, fm in cases.items():
                qm, samples = collide(fm, args.repeats)
                mm = P.moments(qm)
                controls[lab] = {
                    "moments_signed": {k: float(v) for k, v in zip(INVARIANT_NAMES, mm)},
                    "cancellation_defect": {k: float(v) for k, v in zip(INVARIANT_NAMES,
                                                                         P.cancellation_defects(qm))},
                    "L2_over_input_L2": float(math.sqrt(np.dot(qm, qm) / np.dot(fm, fm))),
                    "uz_source_rate": float(mm[3] / (cw * np.sum(fm))),
                    "collision_timing_ms": timing_stats(samples),
                }
                print(f"P3_CONTROL [{name}] {lab}: canc_max={max(controls[lab]['cancellation_defect'].values()):.3e} "
                      f"L2/L2={controls[lab]['L2_over_input_L2']:.3e} duz/dt={controls[lab]['uz_source_rate']:+.3e}",
                      flush=True)

        summary = summarise_quadrature(records, args.dt)
        summary["timing"]["collision_precomputation_seconds"] = precompute_s
        summary["timing"]["projector_setup_seconds"] = proj_setup_s
        for w in weightings:
            g = summary["projected"][w].get("gpu")
            if g and "median_collision_ms_per_point" in summary["timing"]:
                g["estimated_serial_24_point_stage_ms"] = 24.0 * (
                    summary["timing"]["median_collision_ms_per_point"] + g["median_projection_ms"])
        q = {"name": name, "Nrho": nrho, "M": m, "role": roles[i] if i < 3 else f"extra{i}",
             "velocity_grid": {"Nv": vm.Nv(), "number_of_velocities": vm.vsize(), "Nrho": vm.Nrho(),
                               "M": vm.M(), "cw": cw, "L": float(vm.L()), "sw": float(vm.sw())},
             "summary": summary, "controls": controls, "records": records}
        quads.append(q)
        (args.output_dir / f"p3_partial_{name}.json").write_text(json.dumps(q, indent=1) + "\n")

        # release device memory before the next configuration
        del projectors, scattering, vm, d_input, d_output, P
        gc.collect()

    gate_results = evaluate_gates(quads, gates)
    report = {
        "schema_version": 1,
        "purpose": "DGFS phase-3 quadrature sensitivity and conservative-projection GPU audit",
        "inputs": {"config": str(args.config), "snapshot": str(args.snapshot), "mesh": str(args.mesh),
                   "scattering_model": scattering_name, "quadratures": args.quadratures,
                   "repeats": args.repeats, "projection_repeats": args.projection_repeats,
                   "gpu_solve": args.gpu_solve, "dt": args.dt},
        "runtime": {"host": platform.node(), "python": platform.python_version(), "gpu": gpu_identity()},
        "gates": gates, "gate_results": gate_results, "quadratures": quads,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--quadratures", default="32:6,16:16,16:24",
                    help="comma list Nrho:M; order = baseline, proposed, reference")
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--output-json", type=Path, default=Path("p3_quadrature_projection_audit.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("p3_quadrature_projection_audit.csv"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--projection-repeats", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=0, help="0 = all DG solution points")
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--gpu-solve", choices=("device", "host"), default="device")
    ap.add_argument("--skip-gpu-weighting", nargs="*", default=[])
    ap.add_argument("--skip-maxwellian", action="store_true")
    ap.add_argument("--gate-post-defect", type=float, default=DEFAULT_GATES["post_defect"])
    ap.add_argument("--gate-raw-proposed", type=float, default=DEFAULT_GATES["raw_proposed"])
    ap.add_argument("--gate-rel-correction", type=float, default=DEFAULT_GATES["rel_correction"])
    ap.add_argument("--gate-negmass-ratio", type=float, default=DEFAULT_GATES["negmass_ratio"])
    ap.add_argument("--gate-newly-negative", type=int, default=DEFAULT_GATES["newly_negative"])
    ap.add_argument("--gate-min-ratio", type=float, default=DEFAULT_GATES["min_ratio"])
    ap.add_argument("--gate-z-reduction", type=float, default=DEFAULT_GATES["z_reduction"])
    ap.add_argument("--gate-projection-cost-fraction", type=float,
                    default=DEFAULT_GATES["projection_cost_fraction"])
    args = ap.parse_args()
    if args.repeats < 1 or args.projection_repeats < 1:
        ap.error("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config, args.snapshot, args.mesh = (p.resolve() for p in (args.config, args.snapshot, args.mesh))

    report = run(args)
    out_json = args.output_dir / args.output_json
    out_csv = args.output_dir / args.output_csv
    out_json.write_text(json.dumps(report, indent=1) + "\n")
    write_points_csv(out_csv, report["quadratures"])
    print_summary_table(report["quadratures"], report["gate_results"])
    npass = sum(1 for g in report["gate_results"] if g["pass"])
    print(f"P3_AUDIT_JSON={out_json.resolve()}")
    print(f"P3_AUDIT_CSV={out_csv.resolve()}")
    print(f"P3_AUDIT_GATES_PASSED={npass}/{len(report['gate_results'])}")
    critical = [g for g in report["gate_results"] if g["gate"].startswith(("G0", "G1", "G6"))]
    print(f"P3_AUDIT_CRITICAL_GATES={'PASS' if all(g['pass'] for g in critical) else 'FAIL'}")


if __name__ == "__main__":
    main()
