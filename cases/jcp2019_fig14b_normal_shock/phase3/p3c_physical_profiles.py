#!/usr/bin/env python3
"""Plot the Ma=1.59 P3C normal-shock profiles and angular-rule differences.

All 24 DG solution values are retained (8 elements x 3 GLL nodes). Each
element is drawn as a separate three-node segment, so discontinuous traces at
element interfaces are never joined by an artificial vertical line.
"""
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
    "run_M16_raw": r"angular $M_\Omega=16$, raw",
    "run_M16_fplus": r"angular $M_\Omega=16$, conservative $f^+$",
    "run_M24_raw": r"angular $M_\Omega=24$, raw",
}
COLORS = {"run_M16_raw": "#BC6C25", "run_M16_fplus": "#355FA3", "run_M24_raw": "#111111"}


def nodal_profile(record: dict, key: str, scale: float) -> np.ndarray:
    """Return array with shape (3 GLL nodes, number of elements)."""
    return np.asarray([[record["points"][u][e][key]
                        for e in range(len(record["points"][0]))]
                       for u in range(len(record["points"]))], dtype=float) * scale


def plot_element_segments(ax, xnodes, values, *, color, label, marker=None,
                          ms=0.0, lw=1.6, zorder=2, markerfacecolor=None):
    """Plot within-element traces only; never bridge a DG interface."""
    for e in range(xnodes.shape[1]):
        ax.plot(xnodes[:, e], values[:, e], color=color, lw=lw,
                marker=marker, ms=ms, markeredgewidth=1.25,
                markerfacecolor=markerfacecolor, zorder=zorder,
                label=label if e == 0 else None)


def style_axes(axes, edges):
    for ax in axes.flat:
        for xb in edges[1:-1]:
            ax.axvline(xb, color="#E5E5E5", lw=0.55, zorder=0)
        ax.grid(True, axis="y", color="#D8D8D8", lw=0.65, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(edges[0] - 0.5, edges[-1] + 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=Path("p3c_comparison.json"))
    ap.add_argument("--mach", type=float, default=1.59)
    ap.add_argument("--profiles-png", type=Path, default=Path("p3c_physical_profiles_24points.png"))
    ap.add_argument("--profiles-pdf", type=Path, default=Path("p3c_physical_profiles_24points.pdf"))
    ap.add_argument("--differences-png", type=Path, default=Path("p3c_physical_differences.png"))
    ap.add_argument("--differences-pdf", type=Path, default=Path("p3c_physical_differences.pdf"))
    ap.add_argument("--csv", type=Path, default=Path("p3c_physical_profiles_24points.csv"))
    args = ap.parse_args()

    report = json.loads(args.comparison.read_text())
    tend = float(report.get("tend", float("nan")))
    runs = {r["run"]: r for r in report["runs"]}
    missing = set(RUN_ORDER) - set(runs)
    if missing:
        raise SystemExit(f"missing runs: {sorted(missing)}")

    nd = report["nondim"]
    rho0, u0, T0 = float(nd["rho0"]), float(nd["u0"]), float(nd["T0"])
    p0, q0 = rho0 * u0**2, rho0 * u0**3
    panels = (
        ("rho", rho0, r"Density, $\rho$ [kg m$^{-3}$]"),
        ("ux", u0, r"Streamwise velocity, $u_x$ [m s$^{-1}$]"),
        ("T", T0, r"Temperature, $T$ [K]"),
        ("qx", q0, r"Heat flux, $q_x$ [W m$^{-2}$]"),
        ("Pxx_minus_p", p0, r"Normal stress, $P_{xx}-p$ [Pa]"),
        ("uz", u0, r"Transverse velocity, $u_z$ [m s$^{-1}$]"),
    )

    ne = len(runs[RUN_ORDER[0]]["points"][0])
    edges = np.linspace(-15.0, 15.0, ne + 1)
    xnodes = np.vstack((edges[:-1], 0.5 * (edges[:-1] + edges[1:]), edges[1:]))
    values = {name: {key: nodal_profile(runs[name], key, scale)
                     for key, scale, _ in panels} for name in RUN_ORDER}

    with args.csv.open("w", newline="") as stream:
        fields = ["case", "Mach", "angular_Momega", "element", "gll_node", "x_mm",
                  "rho_kg_m3", "ux_m_s", "T_K", "qx_W_m2", "Pxx_minus_p_Pa", "uz_m_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in RUN_ORDER:
            for e in range(ne):
                for u in range(3):
                    writer.writerow({
                        "case": name, "Mach": args.mach, "angular_Momega": runs[name]["M"],
                        "element": e, "gll_node": u, "x_mm": xnodes[u, e],
                        "rho_kg_m3": values[name]["rho"][u, e],
                        "ux_m_s": values[name]["ux"][u, e], "T_K": values[name]["T"][u, e],
                        "qx_W_m2": values[name]["qx"][u, e],
                        "Pxx_minus_p_Pa": values[name]["Pxx_minus_p"][u, e],
                        "uz_m_s": values[name]["uz"][u, e],
                    })

    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11.5,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 10, "lines.linewidth": 1.7})

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.35), sharex=True, constrained_layout=True)
    for letter, ax, (key, _, ylabel) in zip("abcdef", axes.flat, panels):
        plot_element_segments(ax, xnodes, values["run_M24_raw"][key],
                              color=COLORS["run_M24_raw"], label=LABELS["run_M24_raw"],
                              lw=2.0, zorder=2)
        plot_element_segments(ax, xnodes, values["run_M16_raw"][key],
                              color=COLORS["run_M16_raw"], label=LABELS["run_M16_raw"],
                              marker="^", ms=6.0, lw=0.8, zorder=3, markerfacecolor="none")
        plot_element_segments(ax, xnodes, values["run_M16_fplus"][key],
                              color=COLORS["run_M16_fplus"], label=LABELS["run_M16_fplus"],
                              marker="o", ms=3.8, lw=0.8, zorder=4, markerfacecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        if key == "uz":
            ax.axhline(0.0, color="#777777", lw=0.8)
    style_axes(axes, edges)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$x$ [mm]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3,
               loc="outside upper center", frameon=False,
               title=rf"Normal shock: $Ma={args.mach:g}$; final state $t={tend:g}$; "
                     rf"8 elements $\times$ 3 GLL nodes")
    fig.savefig(args.profiles_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.profiles_pdf, bbox_inches="tight")

    fig2, axes2 = plt.subplots(2, 3, figsize=(13.2, 7.35), sharex=True, constrained_layout=True)
    benchmark = values["run_M24_raw"]
    for letter, ax, (key, _, ylabel) in zip("abcdef", axes2.flat, panels):
        for name, marker, ms in (("run_M16_raw", "^", 5.5), ("run_M16_fplus", "o", 4.0)):
            diff = values[name][key] - benchmark[key]
            plot_element_segments(ax, xnodes, diff, color=COLORS[name],
                                  label=LABELS[name] + r" minus $M_\Omega=24$",
                                  marker=marker, ms=ms, lw=1.3, zorder=3,
                                  markerfacecolor="white")
        ax.axhline(0.0, color="#777777", lw=0.8)
        ax.set_ylabel(r"Difference in " + ylabel[0].lower() + ylabel[1:])
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
    style_axes(axes2, edges)
    for ax in axes2[1, :]:
        ax.set_xlabel(r"$x$ [mm]")
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, ncol=2, loc="outside upper center", frameon=False,
                title=rf"Pointwise angular-rule difference; normal shock $Ma={args.mach:g}$; "
                      rf"{ne * 3} DG values")
    fig2.savefig(args.differences_png, dpi=300, bbox_inches="tight")
    fig2.savefig(args.differences_pdf, bbox_inches="tight")

    print(f"CASE_MACH={args.mach:g}")
    print("ANGULAR_RULES=Momega16,Momega24")
    print(f"DG_VALUES_PER_PROFILE={ne * 3}")
    for path in (args.profiles_png, args.profiles_pdf, args.differences_png,
                 args.differences_pdf, args.csv):
        print(f"WROTE={path}")


if __name__ == "__main__":
    main()
