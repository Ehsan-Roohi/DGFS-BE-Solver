#!/usr/bin/env python3
"""Create publication-scale physical time-history figures from P4B CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = ("run_M16_raw", "run_M16_fplus", "run_M24_raw")
LABELS = {
    "run_M16_raw": r"$M_\Omega=16$, raw",
    "run_M16_fplus": r"$M_\Omega=16$, conservative $f^+$",
    "run_M24_raw": r"$M_\Omega=24$, raw",
}
COLORS = {"run_M16_raw": "#BC6C25", "run_M16_fplus": "#355FA3", "run_M24_raw": "#111111"}
MARKERS = {"run_M16_raw": "^", "run_M16_fplus": "o", "run_M24_raw": "s"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by = {name: sorted((r for r in rows if r["run"] == name), key=lambda r: float(r["time"])) for name in RUNS}

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    panels = (
        ("rms_rho_vs_M24_kg_m3", r"RMS $\rho$ difference [kg m$^{-3}$]"),
        ("rms_ux_vs_M24_m_s", r"RMS $u_x$ difference [m s$^{-1}$]"),
        ("rms_T_vs_M24_K", r"RMS $T$ difference [K]"),
        ("rms_qx_vs_M24_W_m2", r"RMS $q_x$ difference [W m$^{-2}$]"),
        ("rms_Pxx_minus_p_vs_M24_Pa", r"RMS $(P_{xx}-p)$ difference [Pa]"),
        ("rms_uz_vs_M24_m_s", r"RMS $u_z$ difference [m s$^{-1}$]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, constrained_layout=True)
    for letter, ax, (field, ylabel) in zip("abcdef", axes.flat, panels):
        for name in RUNS[:2]:
            rr = by[name]
            ax.plot([float(r["time"]) for r in rr], [float(r[field]) for r in rr],
                    color=COLORS[name], marker=MARKERS[name], ms=6, lw=1.8,
                    markerfacecolor="white", label=LABELS[name])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.grid(True, color="#DDDDDD", lw=0.7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for ax in axes[1, :]:
        ax.set_xlabel(r"time, $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center", frameon=False,
               title=r"Normal shock $Ma=1.59$; differences relative to $M_\Omega=24$")
    fig.savefig(args.out_dir / "p4b_rms_time_history.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / "p4b_rms_time_history.pdf", bbox_inches="tight")

    quality = (
        ("max_abs_uz_m_s", r"max $|u_z|$ [m s$^{-1}$]", "absolute"),
        ("min_f", r"minimum distribution value", "absolute"),
        ("max_negative_mass_fraction", r"negative-mass fraction", "absolute"),
        ("shock_position_mm", r"$x_s-x_{s,24}$ [$\\mu$m]", "delta_um"),
        ("qx_min_W_m2", r"$q_{x,\\min}-q_{x,\\min,24}$ [W m$^{-2}$]", "delta"),
        ("stress_max_Pa", r"$(P_{xx}-p)_{\\max}-(P_{xx}-p)_{\\max,24}$ [Pa]", "delta"),
    )
    fig2, axes2 = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, constrained_layout=True)
    reference = {r["time"]: r for r in by["run_M24_raw"]}
    for letter, ax, (field, ylabel, mode) in zip("abcdef", axes2.flat, quality):
        names = RUNS if mode == "absolute" else RUNS[:2]
        for name in names:
            rr = by[name]
            values = [float(r[field]) for r in rr]
            if mode.startswith("delta"):
                values = [value - float(reference[r["time"]][field])
                          for r, value in zip(rr, values)]
                if mode == "delta_um":
                    values = [1000.0*value for value in values]
            ax.plot([float(r["time"]) for r in rr], values,
                    color=COLORS[name], marker=MARKERS[name], ms=5.5, lw=1.7,
                    markerfacecolor="white", label=LABELS[name])
        if mode.startswith("delta"):
            ax.axhline(0.0, color="#555555", lw=0.9, ls=":", zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.grid(True, color="#DDDDDD", lw=0.7)
        if field in ("max_abs_uz_m_s", "min_f", "max_negative_mass_fraction"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for ax in axes2[1, :]:
        ax.set_xlabel(r"time, $t$")
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, ncol=3, loc="outside upper center", frameon=False,
                title=r"Normal shock $Ma=1.59$; clean-start physical diagnostics")
    fig2.savefig(args.out_dir / "p4b_quality_time_history.png", dpi=300, bbox_inches="tight")
    fig2.savefig(args.out_dir / "p4b_quality_time_history.pdf", bbox_inches="tight")
    print("P4B_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()
