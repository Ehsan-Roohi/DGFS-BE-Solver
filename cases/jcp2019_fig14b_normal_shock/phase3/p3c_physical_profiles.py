#!/usr/bin/env python3
"""Plot P3C shock profiles in SI units using exact GLL cell averages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_ORDER = ("run_M16_raw", "run_M16_fplus", "run_M24_raw")
LABELS = {
    "reference": r"inherited $t=30$",
    "run_M16_raw": r"$M=16$, raw",
    "run_M16_fplus": r"$M=16$, conservative $f^+$",
    "run_M24_raw": r"$M=24$, raw",
}
STYLES = {
    "reference": dict(color="#808080", ls="--", marker="D"),
    "run_M16_raw": dict(color="#BC6C25", ls="-", marker="^"),
    "run_M16_fplus": dict(color="#355FA3", ls="-", marker="o"),
    "run_M24_raw": dict(color="#111111", ls="-", marker="s"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=Path("p3c_comparison.json"))
    ap.add_argument("--png", type=Path, default=Path("p3c_physical_profiles.png"))
    ap.add_argument("--pdf", type=Path, default=Path("p3c_physical_profiles.pdf"))
    ap.add_argument("--csv", type=Path, default=Path("p3c_physical_profiles.csv"))
    args = ap.parse_args()

    report = json.loads(args.comparison.read_text())
    runs = {r["run"]: r for r in report["runs"]}
    missing = set(RUN_ORDER) - set(runs)
    if missing:
        raise SystemExit(f"missing runs: {sorted(missing)}")

    nd = report["nondim"]
    rho0, u0, T0 = float(nd["rho0"]), float(nd["u0"]), float(nd["T0"])
    p0 = rho0 * u0**2
    q0 = rho0 * u0**3
    records = {"reference": report["reference"], **{name: runs[name] for name in RUN_ORDER}}
    ne = len(report["reference"]["cell_averages"]["rho"])
    edges = np.linspace(-15.0, 15.0, ne + 1)
    x = 0.5 * (edges[:-1] + edges[1:])

    panels = (
        ("rho", rho0, r"Density, $\rho$ [kg m$^{-3}$]"),
        ("ux", u0, r"Streamwise velocity, $u_x$ [m s$^{-1}$]"),
        ("T", T0, r"Temperature, $T$ [K]"),
        ("qx", q0, r"Heat flux, $q_x$ [W m$^{-2}$]"),
        ("Pxx_minus_p", p0, r"Normal stress, $P_{xx}-p$ [Pa]"),
        ("uz", u0, r"Transverse velocity, $u_z$ [m s$^{-1}$]"),
    )

    with args.csv.open("w", newline="") as stream:
        fields = ["case", "x_mm", "rho_kg_m3", "ux_m_s", "T_K",
                  "qx_W_m2", "Pxx_minus_p_Pa", "uz_m_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, record in records.items():
            vals = {key: np.asarray(record["cell_averages"][key], dtype=float) * scale
                    for key, scale, _ in panels}
            for i, xi in enumerate(x):
                writer.writerow({
                    "case": name, "x_mm": xi,
                    "rho_kg_m3": vals["rho"][i], "ux_m_s": vals["ux"][i],
                    "T_K": vals["T"][i], "qx_W_m2": vals["qx"][i],
                    "Pxx_minus_p_Pa": vals["Pxx_minus_p"][i],
                    "uz_m_s": vals["uz"][i],
                })

    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 10, "lines.linewidth": 1.8,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True,
                             constrained_layout=True)
    for letter, ax, (key, scale, ylabel) in zip("abcdef", axes.flat, panels):
        for name, record in records.items():
            y = np.asarray(record["cell_averages"][key], dtype=float) * scale
            ax.plot(x, y, ms=4.2, label=LABELS[name], **STYLES[name])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.grid(True, color="#D8D8D8", lw=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-15.5, 15.5)
        if key == "uz":
            ax.axhline(0.0, color="#777777", lw=0.8)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$x$ [mm]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", frameon=False)
    fig.savefig(args.png, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    print(f"RHO_SCALE={rho0:.16e} kg/m^3")
    print(f"VELOCITY_SCALE={u0:.16e} m/s")
    print(f"TEMPERATURE_SCALE={T0:.16e} K")
    print(f"STRESS_SCALE={p0:.16e} Pa")
    print(f"HEAT_FLUX_SCALE={q0:.16e} W/m^2")
    print(f"WROTE_PNG={args.png}")
    print(f"WROTE_PDF={args.pdf}")
    print(f"WROTE_CSV={args.csv}")


if __name__ == "__main__":
    main()
