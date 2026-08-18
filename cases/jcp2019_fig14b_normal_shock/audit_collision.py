#!/usr/bin/env python3
"""Measure the five invariants and cost of the DGFS fast-spectral collision.

This is a short GPU diagnostic, not a time integration.  It loads one saved
full-distribution snapshot, evaluates Q(f, f) independently at selected DG
solution points, and reports cancellation defects for mass, three momentum
components, and kinetic energy.  The existing serial collision implementation
is timed to establish the baseline for the later batched-GPU implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time

import h5py
import numpy as np


INVARIANT_NAMES = ("mass", "momentum_x", "momentum_y", "momentum_z", "energy")


def collision_invariants(q: np.ndarray, cv: np.ndarray, cw: float) -> dict[str, object]:
    """Return absolute and cancellation-normalized moments of Q."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or cv.shape != (3, q.size):
        raise ValueError("q and velocity grid have incompatible shapes")
    c2 = np.sum(cv * cv, axis=0)
    basis = np.vstack((np.ones(q.size), cv, 0.5 * c2))
    signed = cw * (basis @ q)
    scales = cw * (np.abs(basis) @ np.abs(q))
    relative = np.abs(signed) / np.maximum(scales, np.finfo(float).tiny)
    return {
        "signed": {name: float(value) for name, value in zip(INVARIANT_NAMES, signed)},
        "absolute": {name: float(abs(value)) for name, value in zip(INVARIANT_NAMES, signed)},
        "relative_cancellation_defect": {
            name: float(value) for name, value in zip(INVARIANT_NAMES, relative)
        },
        "scales": {name: float(value) for name, value in zip(INVARIANT_NAMES, scales)},
    }


def distribution_moments(f: np.ndarray, cv: np.ndarray, cw: float) -> dict[str, object]:
    density = cw * np.sum(f)
    momentum = cw * (cv @ f)
    velocity = momentum / density
    peculiar = cv - velocity[:, None]
    temperature = (2.0 / 3.0) * cw * np.dot(np.sum(peculiar * peculiar, axis=0), f) / density
    negative = np.maximum(-f, 0.0)
    positive = np.maximum(f, 0.0)
    negative_mass = cw * np.sum(negative)
    positive_mass = cw * np.sum(positive)
    return {
        "density": float(density),
        "velocity": [float(value) for value in velocity],
        "temperature": float(temperature),
        "min_f": float(np.min(f)),
        "max_f": float(np.max(f)),
        "negative_count": int(np.count_nonzero(f < 0.0)),
        "negative_mass_fraction": float(
            negative_mass / max(positive_mass, np.finfo(float).tiny)
        ),
    }


def maxwellian(cv: np.ndarray, density: float = 1.0, temperature: float = 1.0) -> np.ndarray:
    c2 = np.sum(cv * cv, axis=0)
    return density / (math.pi * temperature) ** 1.5 * np.exp(-c2 / temperature)


def entropy_production(q: np.ndarray, f: np.ndarray, cw: float) -> float | None:
    if np.any(f <= 0.0):
        return None
    return float(cw * np.dot(q, np.log(f)))


def load_snapshot(snapshot: Path, mesh: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(snapshot, "r") as h5:
        soln = h5["soln_line_p0"][()]
    with h5py.File(mesh, "r") as h5:
        vertices = h5["spt_line_p0"][()]
    if soln.shape[0] != 3:
        raise ValueError("collision audit expects three GLL points per line element")
    if vertices.shape[1] != soln.shape[2]:
        raise ValueError("snapshot and mesh element counts differ")
    left = vertices[:, :, 0].min(axis=0)
    right = vertices[:, :, 0].max(axis=0)
    x = np.vstack((left, 0.5 * (left + right), right))
    return soln, x


def select_points(soln: np.ndarray, x: np.ndarray, max_points: int) -> list[tuple[int, int]]:
    points = [(upt, elem) for elem in range(soln.shape[2]) for upt in range(soln.shape[0])]
    points.sort(key=lambda pair: (abs(float(x[pair])), pair[1], pair[0]))
    if max_points > 0:
        points = points[:max_points]
    return points


def gpu_identity() -> dict[str, object]:
    import pycuda.driver as cuda

    device = cuda.Context.get_device()
    return {
        "name": device.name(),
        "compute_capability": list(device.compute_capability()),
        "total_memory_bytes": int(device.total_memory()),
        "driver_version": int(cuda.get_driver_version()),
    }


def run_gpu_audit(
    config_path: Path,
    snapshot_path: Path,
    mesh_path: Path,
    repeats: int,
    max_points: int,
    include_maxwellian: bool,
    tolerance: float,
) -> dict[str, object]:
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

    cfg = Inifile.load(str(config_path))
    backend = get_backend("cuda", cfg)
    vm = DGFSVelocityMesh(backend, cfg)
    cv = vm.cv()
    cw = vm.cw()

    precompute_start = time.perf_counter()
    scattering_name = cfg.get("scattering-model", "type")
    scattering_cls = subclass_where(
        DGFSScatteringModel, scattering_model=scattering_name
    )
    scattering = scattering_cls(backend, cfg, vm)
    cuda.Context.synchronize()
    precompute_seconds = time.perf_counter() - precompute_start

    soln, x = load_snapshot(snapshot_path, mesh_path)
    if soln.shape[1] != vm.vsize():
        raise ValueError("snapshot velocity dimension does not match configuration")
    points = select_points(soln, x, max_points)

    shape = (1, vm.vsize(), 1)
    zeros = np.zeros(shape, dtype=backend.fpdtype)
    d_input = backend.matrix(shape, zeros, tags={"align"})
    d_output = backend.matrix(shape, zeros, tags={"align"})

    records = []

    def evaluate(label: str, f: np.ndarray, upt: int | None, elem: int | None,
                 x_value: float | None) -> None:
        packed = np.asarray(f, dtype=backend.fpdtype).reshape(shape)
        d_input.set(packed)
        d_output.set(zeros)

        # Warm up kernels and FFT plans outside the reported timing samples.
        scattering.fs(d_input, d_output, 0, 0)
        cuda.Context.synchronize()

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            scattering.fs(d_input, d_output, 0, 0)
            cuda.Context.synchronize()
            timings.append(time.perf_counter() - start)
        q = d_output.get()[0, :, 0]

        invariants = collision_invariants(q, cv, cw)
        input_moments = distribution_moments(f, cv, cw)
        relative = invariants["relative_cancellation_defect"]
        record = {
            "label": label,
            "solution_point": upt,
            "element": elem,
            "x_nondim": x_value,
            "input": input_moments,
            "collision": {
                "L1": float(cw * np.sum(np.abs(q))),
                "L2": float(math.sqrt(cw * np.dot(q, q))),
                "Linf": float(np.max(np.abs(q))),
                "entropy_production": entropy_production(q, f, cw),
                "invariants": invariants,
            },
            "timing_seconds": {
                "samples": [float(value) for value in timings],
                "minimum": float(np.min(timings)),
                "median": float(np.median(timings)),
                "mean": float(np.mean(timings)),
                "maximum": float(np.max(timings)),
            },
            "passes_relative_tolerance": bool(max(relative.values()) <= tolerance),
        }
        if label == "maxwellian":
            record["collision"]["L2_over_input_L2"] = float(
                math.sqrt(cw * np.dot(q, q))
                / max(math.sqrt(cw * np.dot(f, f)), np.finfo(float).tiny)
            )
        records.append(record)
        print(
            f"COLLISION_AUDIT_POINT label={label} x={x_value} "
            f"max_rel={max(relative.values()):.6e} "
            f"median_ms={1.0e3*np.median(timings):.3f}",
            flush=True,
        )

    for upt, elem in points:
        evaluate(
            f"snapshot_u{upt}_e{elem}", soln[upt, :, elem], upt, elem,
            float(x[upt, elem]),
        )
    if include_maxwellian:
        evaluate("maxwellian", maxwellian(cv), None, None, None)

    snapshot_records = [record for record in records if record["element"] is not None]
    all_relative = {
        name: [
            record["collision"]["invariants"]["relative_cancellation_defect"][name]
            for record in snapshot_records
        ]
        for name in INVARIANT_NAMES
    }
    timing_values = [
        sample
        for record in snapshot_records
        for sample in record["timing_seconds"]["samples"]
    ]
    summary = {
        "snapshot_points": len(snapshot_records),
        "relative_tolerance": tolerance,
        "max_relative_cancellation_defect": {
            name: float(max(values)) for name, values in all_relative.items()
        },
        "median_collision_time_ms": float(1.0e3 * np.median(timing_values)),
        "mean_collision_time_ms": float(1.0e3 * np.mean(timing_values)),
        "estimated_serial_24_point_time_ms": float(
            24.0 * 1.0e3 * np.median(timing_values)
        ),
        "all_snapshot_points_pass": bool(
            all(record["passes_relative_tolerance"] for record in snapshot_records)
        ),
    }
    report = {
        "schema_version": 1,
        "purpose": "DGFS full-Boltzmann collision invariant and serial timing baseline",
        "inputs": {
            "config": str(config_path),
            "snapshot": str(snapshot_path),
            "mesh": str(mesh_path),
            "scattering_model": scattering_name,
            "repeats": repeats,
        },
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "gpu": gpu_identity(),
            "collision_precomputation_seconds": precompute_seconds,
        },
        "velocity_grid": {
            "Nv": vm.Nv(),
            "number_of_velocities": vm.vsize(),
            "Nrho": vm.Nrho(),
            "M": vm.M(),
            "cw": cw,
        },
        "summary": summary,
        "records": records,
    }
    return report


def write_csv(path: Path, report: dict[str, object]) -> None:
    fields = [
        "label", "solution_point", "element", "x_nondim", "min_f",
        "negative_count", "negative_mass_fraction", "collision_L2",
        "median_collision_ms", *[f"relative_{name}" for name in INVARIANT_NAMES],
        "passes_relative_tolerance",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in report["records"]:
            relative = record["collision"]["invariants"]["relative_cancellation_defect"]
            row = {
                "label": record["label"],
                "solution_point": record["solution_point"],
                "element": record["element"],
                "x_nondim": record["x_nondim"],
                "min_f": record["input"]["min_f"],
                "negative_count": record["input"]["negative_count"],
                "negative_mass_fraction": record["input"]["negative_mass_fraction"],
                "collision_L2": record["collision"]["L2"],
                "median_collision_ms": 1.0e3 * record["timing_seconds"]["median"],
                "passes_relative_tolerance": record["passes_relative_tolerance"],
            }
            row.update({f"relative_{name}": relative[name] for name in INVARIANT_NAMES})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("collision_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("collision_audit.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=0,
                        help="0 audits all DG solution points")
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--skip-maxwellian", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_points < 0:
        parser.error("--max-points must be nonnegative")
    if not (math.isfinite(args.relative_tolerance) and args.relative_tolerance > 0.0):
        parser.error("--relative-tolerance must be finite and positive")

    report = run_gpu_audit(
        args.config.resolve(), args.snapshot.resolve(), args.mesh.resolve(),
        args.repeats, args.max_points, not args.skip_maxwellian,
        args.relative_tolerance,
    )
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output_csv, report)
    summary = report["summary"]
    print(f"COLLISION_AUDIT_JSON={args.output_json.resolve()}")
    print(f"COLLISION_AUDIT_CSV={args.output_csv.resolve()}")
    print(f"COLLISION_AUDIT_POINTS={summary['snapshot_points']}")
    print(f"COLLISION_AUDIT_ALL_PASS={'yes' if summary['all_snapshot_points_pass'] else 'no'}")
    print(f"COLLISION_MEDIAN_MS={summary['median_collision_time_ms']:.6f}")


if __name__ == "__main__":
    main()
