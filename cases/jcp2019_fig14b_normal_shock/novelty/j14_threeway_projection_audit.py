#!/usr/bin/env python3
"""Three-way operator audit: raw vs unweighted vs f+-weighted projection.

This is POST-PROCESSING ONLY.  It evaluates the existing fast-spectral
collision term Q(f,f) at each DG solution point of an existing distribution
snapshot, then applies two CPU-side five-moment projections to the same raw Q:

  unweighted:  w_i = 1
  weighted:    w_i = max(f_i, 0)

No DGFS time integration is performed.  The purpose is to decide whether the
f+-weighted projection does anything materially different from the standard
unweighted constrained correction.

Important interpretation:
- The unweighted projection is the minimum-Euclidean-norm correction, so its
  plain L2 correction should not be expected to exceed the weighted method.
- The discriminating diagnostics are support/tail localization, correction on
  negative/near-empty velocity nodes, high-order collision-moment disturbance,
  and Gram-matrix conditioning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import h5py
import numpy as np

INV_NAMES = ("mass", "momentum_x", "momentum_y", "momentum_z", "energy")
TINY = np.finfo(float).tiny


def invariant_basis(cv: np.ndarray) -> np.ndarray:
    return np.vstack((
        np.ones(cv.shape[1]),
        cv,
        0.5*np.sum(cv*cv, axis=0),
    ))


def invariant_report(q: np.ndarray, B: np.ndarray, cw: float) -> dict:
    signed = cw*(B @ q)
    scale = cw*(np.abs(B) @ np.abs(q))
    defect = np.abs(signed)/np.maximum(scale, TINY)
    return {
        "signed": dict(zip(INV_NAMES, map(float, signed))),
        "scale": dict(zip(INV_NAMES, map(float, scale))),
        "relative_cancellation_defect": dict(zip(INV_NAMES, map(float, defect))),
        "max_relative_cancellation_defect": float(np.max(defect)),
    }


def scaled_condition_number(G: np.ndarray) -> float:
    d = np.sqrt(np.maximum(np.diag(G), TINY))
    Gs = G/np.outer(d, d)
    return float(np.linalg.cond(Gs))


def project(q: np.ndarray, B: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict]:
    # Constraint: B(q - w B^T lambda) = 0.
    G = (B*w[None, :]) @ B.T
    rhs = B @ q
    solver = "solve"
    try:
        lam = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(G, rhs, rcond=None)[0]
        solver = "lstsq"
    delta = -w*(B.T @ lam)
    qc = q + delta
    return qc, {
        "solver": solver,
        "gram_condition_2": float(np.linalg.cond(G)),
        "gram_condition_scaled_2": scaled_condition_number(G),
        "lambda_l2": float(np.linalg.norm(lam)),
    }


def load_snapshot(snapshot: Path, mesh: Path):
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


def local_support_geometry(f: np.ndarray, cv: np.ndarray):
    fp = np.maximum(f, 0.0)
    mass = float(np.sum(fp))
    if mass <= TINY:
        u = np.zeros(3)
        sigma = 1.0
    else:
        u = (cv*fp[None, :]).sum(axis=1)/mass
        c = cv - u[:, None]
        sigma = math.sqrt(max(float(np.sum(fp*np.sum(c*c, axis=0))/(3.0*mass)), TINY))
    c = cv - u[:, None]
    speed = np.sqrt(np.sum(c*c, axis=0))
    tail = speed > 3.0*sigma
    fmax = float(np.max(fp)) if fp.size else 0.0
    low = fp <= max(1.0e-8*fmax, TINY)
    neg = f < 0.0
    return fp, u, c, sigma, tail, low, neg


def rel_norm(delta: np.ndarray, q: np.ndarray, ord_kind: str) -> float:
    if ord_kind == "l2":
        return float(np.linalg.norm(delta)/max(np.linalg.norm(q), TINY))
    if ord_kind == "linf":
        return float(np.max(np.abs(delta))/max(float(np.max(np.abs(q))), TINY))
    raise ValueError(ord_kind)


def region_fraction(delta: np.ndarray, mask: np.ndarray) -> float:
    den = np.linalg.norm(delta)
    return float(np.linalg.norm(delta[mask])/max(den, TINY))


def high_order_kernel_report(qraw, qc, c, cw):
    c2 = np.sum(c*c, axis=0)
    kernels = {
        "heatflux_like_x": 0.5*c2*c[0],
        "heatflux_like_y": 0.5*c2*c[1],
        "stress_xy": c[0]*c[1],
        "fourth_scalar": c2*c2,
    }
    out = {}
    for name, K in kernels.items():
        raw_signed = cw*float(np.sum(K*qraw))
        corr_signed = cw*float(np.sum(K*qc))
        delta_signed = corr_signed - raw_signed
        cancellation_scale = cw*float(np.sum(np.abs(K*qraw)))
        out[name] = {
            "raw_signed": raw_signed,
            "corrected_signed": corr_signed,
            "delta_signed": delta_signed,
            "relative_disturbance_to_raw_scale": float(
                abs(delta_signed)/max(cancellation_scale, TINY)
            ),
        }
    return out


def mode_metrics(qraw, qc, qinfo, f, B, cv, cw, fp, c, tail, low, neg):
    delta = qc - qraw
    floor = max(float(np.max(fp))*1.0e-14, 1.0e-300)
    invw = 1.0/np.maximum(fp, floor)
    q_weighted_norm = math.sqrt(float(np.sum(qraw*qraw*invw)))
    d_weighted_norm = math.sqrt(float(np.sum(delta*delta*invw)))
    return {
        "invariants": invariant_report(qc, B, cw),
        "projection": qinfo,
        "relative_correction_l2": rel_norm(delta, qraw, "l2"),
        "relative_correction_linf": rel_norm(delta, qraw, "linf"),
        "support_penalized_relative_correction": float(
            d_weighted_norm/max(q_weighted_norm, TINY)
        ),
        "tail_correction_l2_fraction": region_fraction(delta, tail),
        "low_support_correction_l2_fraction": region_fraction(delta, low),
        "negative_node_correction_l2_fraction": region_fraction(delta, neg),
        "max_abs_correction_on_low_support": float(
            np.max(np.abs(delta[low])) if np.any(low) else 0.0
        ),
        "max_abs_correction_on_negative_nodes": float(
            np.max(np.abs(delta[neg])) if np.any(neg) else 0.0
        ),
        "high_order_collision_moments": high_order_kernel_report(qraw, qc, c, cw),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

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
    vm = DGFSVelocityMesh(backend, cfg)
    scattering_cls = subclass_where(
        DGFSScatteringModel,
        scattering_model=cfg.get("scattering-model", "type"),
    )
    t0 = time.perf_counter()
    scattering = scattering_cls(backend, cfg, vm)
    cuda.Context.synchronize()
    precompute = time.perf_counter() - t0

    soln, x = load_snapshot(args.snapshot, args.mesh)
    if soln.shape[1] != vm.vsize():
        raise ValueError(
            f"snapshot has {soln.shape[1]} velocities; config expects {vm.vsize()}"
        )

    cv = np.asarray(vm.cv(), dtype=float)
    cw = float(vm.cw())
    B = invariant_basis(cv)
    shape = (1, vm.vsize(), 1)
    zeros = np.zeros(shape, dtype=backend.fpdtype)
    d_f = backend.matrix(shape, zeros, tags={"align"})
    d_q = backend.matrix(shape, zeros, tags={"align"})

    rows = []
    for elem in range(soln.shape[2]):
        for upt in range(soln.shape[0]):
            f = np.asarray(soln[upt, :, elem], dtype=float)
            d_f.set(f.astype(backend.fpdtype).reshape(shape))

            # Warm + timed raw collision evaluations.  Projection is deliberately
            # CPU-side to compare mathematical corrections on exactly the same Q.
            d_q.set(zeros)
            scattering.fs(d_f, d_q, 0, 0)
            cuda.Context.synchronize()
            raw_times = []
            for _ in range(max(args.repeats, 1)):
                d_q.set(zeros)
                st = time.perf_counter()
                scattering.fs(d_f, d_q, 0, 0)
                cuda.Context.synchronize()
                raw_times.append(time.perf_counter() - st)
            qraw = np.asarray(d_q.get()[0, :, 0], dtype=float).copy()

            raw_inv = invariant_report(qraw, B, cw)
            fp, u, c, sigma, tail, low, neg = local_support_geometry(f, cv)

            qu, info_u = project(qraw, B, np.ones_like(f))
            qw, info_w = project(qraw, B, fp)
            mu = mode_metrics(qraw, qu, info_u, f, B, cv, cw, fp, c, tail, low, neg)
            mw = mode_metrics(qraw, qw, info_w, f, B, cv, cw, fp, c, tail, low, neg)

            neg_mass = float(np.sum(np.maximum(-f, 0.0))/max(np.sum(fp), TINY))
            row = {
                "label": args.label,
                "M_omega": int(vm.M()),
                "solution_point": int(upt),
                "element": int(elem),
                "x_nondim": float(x[upt, elem]),
                "min_f": float(np.min(f)),
                "negative_mass_fraction": neg_mass,
                "positive_velocity_node_fraction": float(np.mean(f > 0.0)),
                "tail_velocity_node_fraction": float(np.mean(tail)),
                "low_support_node_fraction": float(np.mean(low)),
                "thermal_sigma_component": float(sigma),
                "local_positive_mean_velocity": list(map(float, u)),
                "raw": {
                    "invariants": raw_inv,
                    "collision_time_ms": float(1e3*np.median(raw_times)),
                },
                "unweighted": mu,
                "weighted_fplus": mw,
            }
            rows.append(row)
            print(
                "J14_THREEWAY_POINT "
                f"case={args.label} M={vm.M()} e={elem} u={upt} "
                f"raw={raw_inv['max_relative_cancellation_defect']:.3e} "
                f"unw={mu['invariants']['max_relative_cancellation_defect']:.3e} "
                f"wgt={mw['invariants']['max_relative_cancellation_defect']:.3e} "
                f"tail_unw={mu['tail_correction_l2_fraction']:.3e} "
                f"tail_wgt={mw['tail_correction_l2_fraction']:.3e}",
                flush=True,
            )

    def arr(path):
        vals = []
        for r in rows:
            v = r
            for key in path:
                v = v[key]
            vals.append(float(v))
        return np.asarray(vals)

    def frac_weighted_smaller(path):
        a = arr(("unweighted",) + tuple(path))
        b = arr(("weighted_fplus",) + tuple(path))
        return float(np.mean(b < a))

    summary = {
        "label": args.label,
        "M_omega": int(vm.M()),
        "points": len(rows),
        "raw_max_defect": float(max(r["raw"]["invariants"]["max_relative_cancellation_defect"] for r in rows)),
        "unweighted_max_defect": float(max(r["unweighted"]["invariants"]["max_relative_cancellation_defect"] for r in rows)),
        "weighted_max_defect": float(max(r["weighted_fplus"]["invariants"]["max_relative_cancellation_defect"] for r in rows)),
        "median_relative_correction_l2": {
            "unweighted": float(np.median(arr(("unweighted", "relative_correction_l2")))),
            "weighted": float(np.median(arr(("weighted_fplus", "relative_correction_l2")))),
        },
        "median_tail_correction_l2_fraction": {
            "unweighted": float(np.median(arr(("unweighted", "tail_correction_l2_fraction")))),
            "weighted": float(np.median(arr(("weighted_fplus", "tail_correction_l2_fraction")))),
        },
        "median_low_support_correction_l2_fraction": {
            "unweighted": float(np.median(arr(("unweighted", "low_support_correction_l2_fraction")))),
            "weighted": float(np.median(arr(("weighted_fplus", "low_support_correction_l2_fraction")))),
        },
        "median_negative_node_correction_l2_fraction": {
            "unweighted": float(np.median(arr(("unweighted", "negative_node_correction_l2_fraction")))),
            "weighted": float(np.median(arr(("weighted_fplus", "negative_node_correction_l2_fraction")))),
        },
        "median_scaled_gram_condition": {
            "unweighted": float(np.median(arr(("unweighted", "projection", "gram_condition_scaled_2")))),
            "weighted": float(np.median(arr(("weighted_fplus", "projection", "gram_condition_scaled_2")))),
        },
        "weighted_smaller_fraction": {
            "tail_correction_fraction": frac_weighted_smaller(("tail_correction_l2_fraction",)),
            "low_support_correction_fraction": frac_weighted_smaller(("low_support_correction_l2_fraction",)),
            "negative_node_correction_fraction": frac_weighted_smaller(("negative_node_correction_l2_fraction",)),
            "heatflux_like_x_disturbance": frac_weighted_smaller(("high_order_collision_moments", "heatflux_like_x", "relative_disturbance_to_raw_scale")),
            "fourth_scalar_disturbance": frac_weighted_smaller(("high_order_collision_moments", "fourth_scalar", "relative_disturbance_to_raw_scale")),
        },
        "median_high_order_relative_disturbance": {},
        "max_negative_mass_fraction_of_state": float(max(r["negative_mass_fraction"] for r in rows)),
        "median_raw_collision_ms": float(np.median([r["raw"]["collision_time_ms"] for r in rows])),
    }
    for mom in ("heatflux_like_x", "heatflux_like_y", "stress_xy", "fourth_scalar"):
        summary["median_high_order_relative_disturbance"][mom] = {
            "unweighted": float(np.median(arr(("unweighted", "high_order_collision_moments", mom, "relative_disturbance_to_raw_scale")))),
            "weighted": float(np.median(arr(("weighted_fplus", "high_order_collision_moments", mom, "relative_disturbance_to_raw_scale")))),
        }

    report = {
        "schema_version": 1,
        "purpose": "raw vs standard unweighted vs fplus-weighted collision projection audit",
        "interpretation_notes": [
            "No time integration is performed.",
            "Unweighted is the minimum Euclidean-L2 correction by construction; plain L2 is not the weighted method's target metric.",
            "The principal discriminants are velocity-tail/support localization, high-order collision moments, and conditioning.",
            "The fplus projection acts on Q only and is not a positivity limiter for f.",
        ],
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "collision_precompute_seconds": precompute,
        },
        "velocity_grid": {
            "Nv": int(vm.Nv()),
            "Nrho": int(vm.Nrho()),
            "M_omega": int(vm.M()),
            "vsize": int(vm.vsize()),
            "cw": cw,
        },
        "summary": summary,
        "records": rows,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "label", "M_omega", "element", "solution_point", "x_nondim",
        "min_f", "negative_mass_fraction", "raw_defect", "unweighted_defect",
        "weighted_defect", "unweighted_corr_l2", "weighted_corr_l2",
        "unweighted_tail_fraction", "weighted_tail_fraction",
        "unweighted_low_support_fraction", "weighted_low_support_fraction",
        "unweighted_negative_node_fraction", "weighted_negative_node_fraction",
        "unweighted_cond_scaled", "weighted_cond_scaled",
        "unweighted_qx_disturbance", "weighted_qx_disturbance",
        "unweighted_R4_disturbance", "weighted_R4_disturbance",
    ]
    with args.output_csv.open("w", newline="") as stream:
        wr = csv.DictWriter(stream, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            u = r["unweighted"]
            w = r["weighted_fplus"]
            wr.writerow({
                "label": r["label"], "M_omega": r["M_omega"],
                "element": r["element"], "solution_point": r["solution_point"],
                "x_nondim": r["x_nondim"], "min_f": r["min_f"],
                "negative_mass_fraction": r["negative_mass_fraction"],
                "raw_defect": r["raw"]["invariants"]["max_relative_cancellation_defect"],
                "unweighted_defect": u["invariants"]["max_relative_cancellation_defect"],
                "weighted_defect": w["invariants"]["max_relative_cancellation_defect"],
                "unweighted_corr_l2": u["relative_correction_l2"],
                "weighted_corr_l2": w["relative_correction_l2"],
                "unweighted_tail_fraction": u["tail_correction_l2_fraction"],
                "weighted_tail_fraction": w["tail_correction_l2_fraction"],
                "unweighted_low_support_fraction": u["low_support_correction_l2_fraction"],
                "weighted_low_support_fraction": w["low_support_correction_l2_fraction"],
                "unweighted_negative_node_fraction": u["negative_node_correction_l2_fraction"],
                "weighted_negative_node_fraction": w["negative_node_correction_l2_fraction"],
                "unweighted_cond_scaled": u["projection"]["gram_condition_scaled_2"],
                "weighted_cond_scaled": w["projection"]["gram_condition_scaled_2"],
                "unweighted_qx_disturbance": u["high_order_collision_moments"]["heatflux_like_x"]["relative_disturbance_to_raw_scale"],
                "weighted_qx_disturbance": w["high_order_collision_moments"]["heatflux_like_x"]["relative_disturbance_to_raw_scale"],
                "unweighted_R4_disturbance": u["high_order_collision_moments"]["fourth_scalar"]["relative_disturbance_to_raw_scale"],
                "weighted_R4_disturbance": w["high_order_collision_moments"]["fourth_scalar"]["relative_disturbance_to_raw_scale"],
            })

    print("J14_THREEWAY_COMPLETE")
    print("J14_THREEWAY_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
