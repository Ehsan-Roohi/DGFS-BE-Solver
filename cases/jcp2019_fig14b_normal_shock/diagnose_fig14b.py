#!/usr/bin/env python3
"""Audit positivity, phase drift, and kinetic inventories for Figure 14(b).

The steady normal shock has a translational neutral mode.  A conventional
residual therefore mixes true profile changes with a slow displacement of the
shock.  This script preserves the paper residual and adds an independent,
phase-aligned audit of the saved full-distribution snapshots.

The phase-space integrals reported here are inventories for an open domain;
they are not claimed to be conserved in time because particles enter and leave
through the two reservoir boundaries.  Collision-invariant defects require an
online audit of Q(f, f), which is a separate development step.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
from pathlib import Path
import re
import h5py
import numpy as np


R_UNIVERSAL = 8.3144598
MEAN_FREE_PATH_M = 1.648e-3
GLL2_WEIGHTS = np.array([1.0, 4.0, 1.0]) / 6.0
FIELD_NAMES = {"rho": "rho", "u": "U:x", "T": "T", "q": "Q:x", "p": "p"}


def output_time(path: Path) -> float:
    match = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.frfss$", path.name)
    if not match:
        raise ValueError(f"cannot obtain output time from {path.name}")
    return float(match.group(1))


def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with path.open() as stream:
        cfg.read_file(stream)
    return cfg


def velocity_mesh(cfg: configparser.ConfigParser) -> tuple[np.ndarray, float]:
    """Reproduce the midpoint velocity grid used by DGFSVelocityMesh."""
    nv = cfg.getint("constants", "Nv")
    t0 = cfg.getfloat("non-dim", "T0")
    molar_mass = cfg.getfloat("non-dim", "molarMass0")
    u0 = math.sqrt(2.0 * R_UNIVERSAL * t0 / molar_mass)
    cmax = cfg.getfloat("velocity-mesh", "cmax") / u0
    tmax = cfg.getfloat("velocity-mesh", "Tmax") / t0
    dev = cfg.getfloat("velocity-mesh", "dev")
    length = cmax + dev * math.sqrt(tmax)
    c0 = np.linspace(-length + length / nv, length - length / nv, nv)
    cx, cy, cz = np.meshgrid(c0, c0, c0, indexing="ij")
    cv = np.vstack((cx.ravel(), cy.ravel(), cz.ravel()))
    cw = (2.0 * length / nv) ** 3
    return cv, cw


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        mesh = h5["spt_line_p0"][()]
    left = mesh[:, :, 0].min(axis=0)
    right = mesh[:, :, 0].max(axis=0)
    order = np.argsort(left)
    return left[order], right[order], order


def lagrange_gll2(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r)
    return np.vstack((0.5 * r * (r - 1.0), 1.0 - r * r,
                      0.5 * r * (r + 1.0)))


def spatial_weights(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return exact degree-two GLL integration weights, shape (3, nelem)."""
    return GLL2_WEIGHTS[:, None] * (right - left)[None, :]


def read_fields(stats_value: object) -> list[str]:
    raw = stats_value.decode() if isinstance(stats_value, bytes) else str(stats_value)
    stats = configparser.ConfigParser()
    stats.optionxform = str
    stats.read_string(raw)
    return [item.strip() for item in stats.get("data", "fields").split(",")]


def read_distribution(path: Path, order: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        soln = h5["soln_line_p0"][()]
    if soln.shape[0] != 3:
        raise ValueError("Figure 14(b) audit requires three GLL solution points")
    return soln[:, :, order]


def read_bulk(path: Path, order: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    with h5py.File(path, "r") as h5:
        moments = h5["moments_line_p0"][()]
        fields = read_fields(h5["stats"][()])
    index = {key: fields.index(name) for key, name in FIELD_NAMES.items()}
    return moments[:, :, order], index


def phase_space_inventory(
    soln: np.ndarray,
    xweights: np.ndarray,
    cv: np.ndarray,
    cw: float,
) -> dict[str, object]:
    """Integrate dimensionless phase-space moments and the discrete H functional."""
    if soln.shape[1] != cv.shape[1]:
        raise ValueError("velocity grid and distribution have incompatible sizes")
    weighted_x = xweights[:, None, :]
    mass_by_v = np.sum(soln * weighted_x, axis=(0, 2))
    mass = cw * np.sum(mass_by_v)
    momentum = cw * (cv @ mass_by_v)
    energy = 0.5 * cw * np.dot(np.sum(cv * cv, axis=0), mass_by_v)

    positive = np.maximum(soln, 0.0)
    negative = np.maximum(-soln, 0.0)
    positive_mass = cw * np.sum(positive * weighted_x)
    negative_mass = cw * np.sum(negative * weighted_x)
    entropy_integrand = np.zeros_like(soln)
    mask = soln > 0.0
    entropy_integrand[mask] = soln[mask] * np.log(soln[mask])
    h_functional = cw * np.sum(entropy_integrand * weighted_x)
    negative_count = int(np.count_nonzero(soln < 0.0))

    return {
        "mass": float(mass),
        "momentum": [float(value) for value in momentum],
        "kinetic_energy": float(energy),
        "H": float(h_functional),
        "min_f": float(np.min(soln)),
        "max_f": float(np.max(soln)),
        "negative_count": negative_count,
        "negative_fraction_by_count": float(negative_count / soln.size),
        "negative_mass": float(negative_mass),
        "negative_mass_fraction": float(
            negative_mass / max(positive_mass, np.finfo(float).tiny)
        ),
    }


def dense_bulk_profile(
    moments: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    points_per_element: int = 241,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(-1.0, 1.0, points_per_element)
    basis = lagrange_gll2(r)
    dense_x = []
    dense_values = []
    for elem, (xl, xr) in enumerate(zip(left, right)):
        dense_x.append(0.5 * ((1.0 - r) * xl + (1.0 + r) * xr))
        dense_values.append(basis.T @ moments[:, :, elem])
    cell_average = np.einsum("i,ive->ve", GLL2_WEIGHTS, moments)
    return np.concatenate(dense_x), np.concatenate(dense_values), cell_average


def midpoint_crossing(x: np.ndarray, rho: np.ndarray, midpoint: float) -> float:
    hit = np.flatnonzero((rho[:-1] - midpoint) * (rho[1:] - midpoint) <= 0.0)
    if not len(hit):
        raise RuntimeError("density-midpoint crossing not found")
    crossings = []
    for idx in hit:
        if rho[idx + 1] == rho[idx]:
            crossings.append(float(x[idx]))
        else:
            crossings.append(float(
                x[idx] + (midpoint - rho[idx]) * (x[idx + 1] - x[idx])
                / (rho[idx + 1] - rho[idx])
            ))
    return min(crossings, key=abs)


def plateau_bounds(cfg: configparser.ConfigParser) -> dict[str, tuple[float, float]]:
    result = {}
    for key, option in (("rho", "rho"), ("u", "ux"), ("T", "T")):
        result[key] = (
            cfg.getfloat("soln-bcs-left", option),
            cfg.getfloat("soln-bcs-right", option),
        )
    gas_constant = R_UNIVERSAL / cfg.getfloat("non-dim", "molarMass0")
    result["p"] = (
        result["rho"][0] * gas_constant * result["T"][0],
        result["rho"][1] * gas_constant * result["T"][1],
    )
    return result


def bounded_overshoot(values: np.ndarray, bounds: tuple[float, float]) -> dict[str, float]:
    lower, upper = sorted(bounds)
    jump = max(upper - lower, np.finfo(float).tiny)
    below = max(lower - float(np.min(values)), 0.0)
    above = max(float(np.max(values)) - upper, 0.0)
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "below_plateau": below,
        "above_plateau": above,
        "max_overshoot_fraction_of_jump": max(below, above) / jump,
    }


def heat_flux_audit(values: np.ndarray) -> dict[str, float | int]:
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    tol = 1.0e-10 * scale
    signs = np.sign(values[np.abs(values) > tol])
    sign_changes = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) else 0
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_lobe": max(float(np.max(values)), 0.0),
        "positive_lobe_fraction_of_peak": max(float(np.max(values)), 0.0) / scale,
        "sign_changes_above_tolerance": sign_changes,
    }


def bulk_audit(
    moments: np.ndarray,
    index: dict[str, int],
    left: np.ndarray,
    right: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, object]:
    x, dense, cell_average = dense_bulk_profile(moments, left, right)
    shock_center = midpoint_crossing(
        x, dense[:, index["rho"]], 0.5 * sum(bounds["rho"])
    )
    raw = {}
    cell = {}
    for key in ("rho", "u", "T", "p"):
        raw[key] = bounded_overshoot(dense[:, index[key]], bounds[key])
        cell[key] = bounded_overshoot(cell_average[index[key]], bounds[key])
    raw["q"] = heat_flux_audit(dense[:, index["q"]])
    cell["q"] = heat_flux_audit(cell_average[index["q"]])
    return {
        "shock_center_nondim": shock_center,
        "raw_DG": raw,
        "cell_average": cell,
    }


def quadrature_grid(
    left: np.ndarray, right: np.ndarray, nquad: int
) -> tuple[np.ndarray, np.ndarray]:
    r, w = np.polynomial.legendre.leggauss(nquad)
    x = []
    xw = []
    for xl, xr in zip(left, right):
        x.append(0.5 * ((1.0 - r) * xl + (1.0 + r) * xr))
        xw.append(0.5 * (xr - xl) * w)
    return np.concatenate(x), np.concatenate(xw)


def evaluate_piecewise(
    soln: np.ndarray,
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    velocity_slice: slice,
) -> np.ndarray:
    elem_index = np.searchsorted(right, x, side="left")
    elem_index = np.clip(elem_index, 0, len(right) - 1)
    values = np.empty((len(x), soln[:, velocity_slice, :].shape[1]))
    for elem in np.unique(elem_index):
        mask = elem_index == elem
        r = (2.0 * x[mask] - left[elem] - right[elem]) / (right[elem] - left[elem])
        values[mask] = lagrange_gll2(r).T @ soln[:, velocity_slice, elem]
    return values


def distribution_residual(
    previous: np.ndarray,
    current: np.ndarray,
    previous_center: float,
    current_center: float,
    left: np.ndarray,
    right: np.ndarray,
    cw: float,
    h0: float,
    chunk_size: int = 2048,
    nquad: int = 12,
) -> dict[str, float]:
    """Return full-f L2 changes before and after translating the old shock."""
    x, xw = quadrature_grid(left, right, nquad)
    delta = current_center - previous_center
    shifted_x = x - delta
    overlap = (shifted_x >= left[0]) & (shifted_x <= right[-1])
    numerator = denominator = 0.0
    aligned_numerator = aligned_denominator = 0.0
    for start in range(0, previous.shape[1], chunk_size):
        stop = min(start + chunk_size, previous.shape[1])
        slc = slice(start, stop)
        old = evaluate_piecewise(previous, x, left, right, slc)
        new = evaluate_piecewise(current, x, left, right, slc)
        numerator += float(np.sum(xw[:, None] * (new - old) ** 2))
        denominator += float(np.sum(xw[:, None] * old ** 2))
        old_aligned = evaluate_piecewise(previous, shifted_x[overlap], left, right, slc)
        new_overlap = new[overlap]
        aligned_numerator += float(
            np.sum(xw[overlap, None] * (new_overlap - old_aligned) ** 2)
        )
        aligned_denominator += float(
            np.sum(xw[overlap, None] * old_aligned ** 2)
        )
    # cw cancels in each normalized ratio, but retain it explicitly to document
    # the phase-space norm and to protect future nonuniform velocity quadrature.
    numerator *= cw
    denominator *= cw
    aligned_numerator *= cw
    aligned_denominator *= cw
    return {
        "shock_displacement_nondim": float(delta),
        "shock_displacement_lambda": float(delta * h0 / MEAN_FREE_PATH_M),
        "unaligned_full_f_L2": math.sqrt(numerator / denominator),
        "phase_aligned_full_f_L2": math.sqrt(aligned_numerator / aligned_denominator),
        "phase_alignment_reduction_factor": math.sqrt(
            numerator / denominator
        ) / max(math.sqrt(aligned_numerator / aligned_denominator), np.finfo(float).tiny),
        "aligned_overlap_fraction": float(np.mean(overlap)),
    }


def relative_change(value: float, reference: float) -> float:
    return (value - reference) / max(abs(reference), np.finfo(float).tiny)


def write_csv(path: Path, snapshots: list[dict[str, object]]) -> None:
    fields = [
        "time", "shock_center_mm", "shock_center_lambda", "min_f",
        "negative_count", "negative_mass_fraction", "mass", "kinetic_energy", "H",
        "mass_change_from_t0", "energy_change_from_t0", "H_change_from_t0",
        "raw_T_overshoot_fraction", "cell_T_overshoot_fraction",
        "raw_q_positive_lobe_fraction", "cell_q_positive_lobe_fraction",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for snapshot in snapshots:
            inv = snapshot["phase_space_inventory"]
            bulk = snapshot["bulk"]
            writer.writerow({
                "time": snapshot["time"],
                "shock_center_mm": snapshot["shock_center_mm"],
                "shock_center_lambda": snapshot["shock_center_lambda"],
                "min_f": inv["min_f"],
                "negative_count": inv["negative_count"],
                "negative_mass_fraction": inv["negative_mass_fraction"],
                "mass": inv["mass"],
                "kinetic_energy": inv["kinetic_energy"],
                "H": inv["H"],
                "mass_change_from_t0": inv["relative_change_from_t0"]["mass"],
                "energy_change_from_t0": inv["relative_change_from_t0"]["kinetic_energy"],
                "H_change_from_t0": inv["relative_change_from_t0"]["H"],
                "raw_T_overshoot_fraction": bulk["raw_DG"]["T"]["max_overshoot_fraction_of_jump"],
                "cell_T_overshoot_fraction": bulk["cell_average"]["T"]["max_overshoot_fraction_of_jump"],
                "raw_q_positive_lobe_fraction": bulk["raw_DG"]["q"]["positive_lobe_fraction_of_peak"],
                "cell_q_positive_lobe_fraction": bulk["cell_average"]["q"]["positive_lobe_fraction_of_peak"],
            })


def plot_report(
    path: Path,
    snapshots: list[dict[str, object]],
    transitions: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    times = np.array([item["time"] for item in snapshots])
    centers = np.array([item["shock_center_lambda"] for item in snapshots])
    min_f = np.array([item["phase_space_inventory"]["min_f"] for item in snapshots])
    negative_fraction = np.array([
        item["phase_space_inventory"]["negative_mass_fraction"] for item in snapshots
    ])
    mass_drift = 100.0 * np.array([
        item["phase_space_inventory"]["relative_change_from_t0"]["mass"]
        for item in snapshots
    ])
    energy_drift = 100.0 * np.array([
        item["phase_space_inventory"]["relative_change_from_t0"]["kinetic_energy"]
        for item in snapshots
    ])
    entropy_drift = 100.0 * np.array([
        item["phase_space_inventory"]["relative_change_from_t0"]["H"]
        for item in snapshots
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    axes[0, 0].plot(times, centers, "o-", lw=2)
    axes[0, 0].set(xlabel="Nondimensional time", ylabel=r"$x_s/\lambda$",
                   title="Shock phase")

    if transitions:
        labels = [f"{item['from_time']:g}→{item['to_time']:g}" for item in transitions]
        x = np.arange(len(labels))
        axes[0, 1].plot(x, [item["unaligned_full_f_L2"] for item in transitions],
                        "o-", lw=2, label="Unaligned")
        axes[0, 1].plot(x, [item["phase_aligned_full_f_L2"] for item in transitions],
                        "s-", lw=2, label="Phase aligned")
        axes[0, 1].set_xticks(x, labels)
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()
    axes[0, 1].set(xlabel="Snapshot interval", ylabel=r"Normalized $L_2(f)$",
                   title="Full-distribution change")

    axes[1, 0].plot(times, min_f, "o-", lw=2, label=r"$\min f$")
    axes[1, 0].plot(times, negative_fraction, "s-", lw=2,
                    label="Negative mass fraction")
    axes[1, 0].set_yscale("symlog", linthresh=1.0e-16)
    axes[1, 0].legend()
    axes[1, 0].set(xlabel="Nondimensional time", ylabel="Dimensionless value",
                   title="Positivity audit")

    axes[1, 1].plot(times, mass_drift, "o-", lw=2, label="Mass inventory")
    axes[1, 1].plot(times, energy_drift, "s-", lw=2, label="Energy inventory")
    axes[1, 1].plot(times, entropy_drift, "^-", lw=2, label="H functional")
    axes[1, 1].legend()
    axes[1, 1].set(xlabel="Nondimensional time", ylabel="Change from t=0 [%]",
                   title="Open-domain inventories")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def paired_snapshots(run: Path) -> list[tuple[float, Path, Path]]:
    distributions = {output_time(path): path for path in run.glob("dist_dgfs_fig14b-*.frfss")}
    bulks = {output_time(path): path for path in run.glob("bulksol_dgfs_fig14b-*.frfss")}
    times = sorted(distributions.keys() & bulks.keys())
    if not times:
        raise FileNotFoundError("no paired distribution and bulk snapshots found")
    return [(time, distributions[time], bulks[time]) for time in times]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("dgfs_fig14b.ini"))
    parser.add_argument("--mesh", type=Path, default=Path("mesh.frfsm"))
    parser.add_argument("--output-json", type=Path, default=Path("dgfs_diagnostics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("dgfs_diagnostics.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("dgfs_diagnostics.png"))
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--velocity-chunk", type=int, default=2048)
    args = parser.parse_args()

    run = args.run_dir.resolve()
    resolve = lambda path: path if path.is_absolute() else run / path
    cfg = read_config(resolve(args.config))
    left, right, order = load_mesh(resolve(args.mesh))
    xweights = spatial_weights(left, right)
    cv, cw = velocity_mesh(cfg)
    bounds = plateau_bounds(cfg)
    h0 = cfg.getfloat("non-dim", "H0")

    snapshots = []
    distributions = []
    for time, dist_path, bulk_path in paired_snapshots(run):
        soln = read_distribution(dist_path, order)
        moments, index = read_bulk(bulk_path, order)
        inventory = phase_space_inventory(soln, xweights, cv, cw)
        bulk = bulk_audit(moments, index, left, right, bounds)
        center_m = bulk["shock_center_nondim"] * h0
        snapshots.append({
            "time": time,
            "distribution_file": dist_path.name,
            "bulk_file": bulk_path.name,
            "shock_center_mm": center_m * 1.0e3,
            "shock_center_lambda": center_m / MEAN_FREE_PATH_M,
            "phase_space_inventory": inventory,
            "bulk": bulk,
        })
        distributions.append(soln)

    reference = snapshots[0]["phase_space_inventory"]
    for snapshot in snapshots:
        inventory = snapshot["phase_space_inventory"]
        inventory["relative_change_from_t0"] = {
            key: relative_change(inventory[key], reference[key])
            for key in ("mass", "kinetic_energy", "H")
        }

    transitions = []
    for idx in range(1, len(snapshots)):
        residual = distribution_residual(
            distributions[idx - 1], distributions[idx],
            snapshots[idx - 1]["bulk"]["shock_center_nondim"],
            snapshots[idx]["bulk"]["shock_center_nondim"],
            left, right, cw, h0, chunk_size=args.velocity_chunk,
        )
        transitions.append({
            "from_time": snapshots[idx - 1]["time"],
            "to_time": snapshots[idx]["time"],
            **residual,
        })

    latest = snapshots[-1]
    report = {
        "schema_version": 1,
        "case": "Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14b",
        "interpretation": {
            "inventories_are_open_domain_quantities": True,
            "time_conservation_claimed": False,
            "collision_invariant_defect_measured": False,
            "mean_free_path_m": MEAN_FREE_PATH_M,
        },
        "velocity_grid": {
            "Nv": cfg.getint("constants", "Nv"),
            "number_of_velocities": int(cv.shape[1]),
            "velocity_weight": cw,
        },
        "snapshots": snapshots,
        "transitions": transitions,
        "latest_summary": {
            "time": latest["time"],
            "negative_count": latest["phase_space_inventory"]["negative_count"],
            "negative_mass_fraction": latest["phase_space_inventory"]["negative_mass_fraction"],
            "raw_temperature_overshoot_fraction": latest["bulk"]["raw_DG"]["T"]["max_overshoot_fraction_of_jump"],
            "cell_temperature_overshoot_fraction": latest["bulk"]["cell_average"]["T"]["max_overshoot_fraction_of_jump"],
            "raw_heat_flux_positive_lobe_fraction": latest["bulk"]["raw_DG"]["q"]["positive_lobe_fraction_of_peak"],
            "cell_heat_flux_positive_lobe_fraction": latest["bulk"]["cell_average"]["q"]["positive_lobe_fraction_of_peak"],
        },
    }

    json_path = resolve(args.output_json)
    csv_path = resolve(args.output_csv)
    figure_path = resolve(args.output_figure)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(csv_path, snapshots)
    if not args.no_figure:
        plot_report(figure_path, snapshots, transitions)

    print(f"DGFS_DIAGNOSTICS_JSON={json_path}")
    print(f"DGFS_DIAGNOSTICS_CSV={csv_path}")
    if not args.no_figure:
        print(f"DGFS_DIAGNOSTICS_FIGURE={figure_path}")
    print(f"LATEST_TIME={latest['time']:.12g}")
    print(f"LATEST_NEGATIVE_COUNT={latest['phase_space_inventory']['negative_count']}")
    print(
        "LATEST_CELL_T_OVERSHOOT_FRACTION="
        f"{latest['bulk']['cell_average']['T']['max_overshoot_fraction_of_jump']:.12e}"
    )


if __name__ == "__main__":
    main()
