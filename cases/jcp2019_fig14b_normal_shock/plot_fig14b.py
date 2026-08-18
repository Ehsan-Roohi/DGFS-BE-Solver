#!/usr/bin/env python3
"""Plot paper-normalized raw DG polynomials and exact GLL cell averages."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np


FIELD_NAMES = {"rho": "rho", "u": "U:x", "T": "T", "q": "Q:x"}


def ini_from_h5(value: object) -> configparser.ConfigParser:
    raw = value.decode() if isinstance(value, bytes) else str(value)
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read_string(raw)
    return cfg


def lagrange_gll2(r: np.ndarray) -> np.ndarray:
    return np.vstack((0.5 * r * (r - 1), 1 - r * r, 0.5 * r * (r + 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("."))
    ap.add_argument("--bulk-soln", type=Path)
    ap.add_argument("--output", type=Path, default=Path("fig14b_raw_and_cell_average.png"))
    args = ap.parse_args()
    run = args.run_dir.resolve()

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(run / "dgfs_fig14b.ini")
    h0 = cfg.getfloat("non-dim", "H0")
    bounds = {
        "rho": (cfg.getfloat("soln-bcs-left", "rho"), cfg.getfloat("soln-bcs-right", "rho")),
        "u": (cfg.getfloat("soln-bcs-left", "ux"), cfg.getfloat("soln-bcs-right", "ux")),
        "T": (cfg.getfloat("soln-bcs-left", "T"), cfg.getfloat("soln-bcs-right", "T")),
    }

    with h5py.File(run / "mesh.frfsm", "r") as h5:
        mesh = h5["spt_line_p0"][()]
    if args.bulk_soln:
        bulk_soln = args.bulk_soln
        if not bulk_soln.is_absolute():
            bulk_soln = run / bulk_soln
    else:
        candidates = list(run.glob("bulksol_dgfs_fig14b-*.frfss"))
        if not candidates:
            raise FileNotFoundError("no bulksol_dgfs_fig14b-*.frfss file found")

        def output_time(path: Path) -> float:
            match = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.frfss$", path.name)
            return float(match.group(1)) if match else float("-inf")

        bulk_soln = max(candidates, key=output_time)

    with h5py.File(bulk_soln, "r") as h5:
        moments = h5["moments_line_p0"][()]
        stats = ini_from_h5(h5["stats"][()])

    fields = [item.strip() for item in stats["data"]["fields"].split(",")]
    fidx = {key: fields.index(name) for key, name in FIELD_NAMES.items()}
    x_left = mesh[:, :, 0].min(axis=0)
    x_right = mesh[:, :, 0].max(axis=0)
    order = np.argsort(x_left)
    x_left, x_right, moments = x_left[order], x_right[order], moments[:, :, order]

    # Three-point GLL weights integrate the degree-two DG representation exactly.
    weights = np.array([1.0, 4.0, 1.0]) / 6.0
    nodal_x = moments
    cell_avg = np.einsum("i,ive->ve", weights, nodal_x)
    x_center = 0.5 * (x_left + x_right)

    r = np.linspace(-1, 1, 241)
    basis = lagrange_gll2(r)
    dense_x: list[np.ndarray] = []
    dense = {key: [] for key in FIELD_NAMES}
    for e in range(moments.shape[2]):
        dense_x.append(0.5 * ((1 - r) * x_left[e] + (1 + r) * x_right[e]))
        for key in FIELD_NAMES:
            dense[key].append(basis.T @ nodal_x[:, fidx[key], e])

    rho_mid = 0.5 * sum(bounds["rho"])
    crossings: list[float] = []
    for xe, rho in zip(dense_x, dense["rho"]):
        hit = np.flatnonzero((rho[:-1] - rho_mid) * (rho[1:] - rho_mid) <= 0)
        for j in hit:
            if rho[j + 1] == rho[j]:
                crossings.append(float(xe[j]))
            else:
                crossings.append(float(xe[j] + (rho_mid - rho[j]) *
                                       (xe[j + 1] - xe[j]) / (rho[j + 1] - rho[j])))
    if not crossings:
        raise RuntimeError("density-midpoint crossing not found")
    x_shock = min(crossings, key=abs)

    def normalize(key: str, value: np.ndarray) -> np.ndarray:
        left, right = bounds[key]
        return (value - right) / (left - right) if key == "u" else (value - left) / (right - left)

    colors = {"rho": "#C83D2B", "u": "#202124", "T": "#2864DC", "q": "#7B3FB2"}
    markers = {"rho": "o", "u": "s", "T": "^"}
    labels = {"rho": r"$\rho$", "u": r"$u_x$", "T": r"$T$"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    for key in ("u", "T", "rho"):
        for xe, value in zip(dense_x, dense[key]):
            xi = (xe - x_shock) * h0 / 1.648e-3
            ax.plot(xi, normalize(key, value), color=colors[key], lw=2)
        ax.scatter(
            (x_center - x_shock) * h0 / 1.648e-3,
            normalize(key, cell_avg[fidx[key]]),
            color=colors[key], marker=markers[key], edgecolor="white", s=48,
            label=labels[key], zorder=4,
        )
    ax.set(xlim=(-8, 6), xlabel=r"$(x-x_s)/\lambda$", ylabel="Normalized property",
           title="JCP 2019 Figure 14(b) coordinates")
    ax.axvline(0, color="0.5", ls="--", lw=1)
    ax.legend()

    specs = [
        (axes[0, 1], "rho", r"$\rho$ [kg m$^{-3}$]"),
        (axes[1, 0], "T", r"$T$ [K]"),
        (axes[1, 1], "q", r"$q_x$ [W m$^{-2}$]"),
    ]
    for panel, key, ylabel in specs:
        for xe, value in zip(dense_x, dense[key]):
            panel.plot(xe * h0 * 1e3, value, color=colors[key], lw=2)
        panel.scatter(x_center * h0 * 1e3, cell_avg[fidx[key]], marker="D", s=45,
                      color="#0F766E", edgecolor="white", zorder=4)
        panel.set(xlabel="$x$ [mm]", ylabel=ylabel,
                  title=f"{key}: raw DG curve and exact cell average")
        for edge in x_right[:-1]:
            panel.axvline(edge * h0 * 1e3, color="0.85", lw=0.7)

    for panel in axes.flat:
        panel.grid(alpha=0.25)
    fig.suptitle("Helium normal shock, M=1.59 — exact JCP Figure 14(b) mesh", fontsize=16)
    output = args.output if args.output.is_absolute() else run / args.output
    fig.savefig(output, dpi=240, facecolor="white")

    report = {
        "output": str(output),
        "bulk_solution": str(bulk_soln),
        "shock_center_mm": x_shock * h0 * 1e3,
        "raw_extrema": {key: [float(np.min(dense[key])), float(np.max(dense[key]))] for key in FIELD_NAMES},
        "cell_average_extrema": {
            key: [float(np.min(cell_avg[fidx[key]])), float(np.max(cell_avg[fidx[key]]))]
            for key in FIELD_NAMES
        },
    }
    (run / "fig14b_extrema.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
