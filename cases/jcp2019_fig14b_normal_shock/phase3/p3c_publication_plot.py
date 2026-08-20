#!/usr/bin/env python3
"""Make a publication-quality P3C transverse-velocity diagnostic.

The comparison JSON contains discontinuous-Galerkin nodal values.  Connecting
the last node of one element to the first node of the next creates artificial
vertical teeth.  This plot therefore uses exact GLL cell averages for all
cross-element curves and reports errors against M24 separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "run_M16_raw": r"$M=16$, raw",
    "run_M16_fplus": r"$M=16$, conservative $f^+$",
    "run_M24_raw": r"$M=24$, raw",
}
COLORS = {
    "run_M16_raw": "#C17C43",
    "run_M16_fplus": "#3E5F9E",
    "run_M24_raw": "#111111",
}
MARKERS = {"run_M16_raw": "^", "run_M16_fplus": "o", "run_M24_raw": "s"}


def cell_profile(record: dict, key: str, scale: float) -> np.ndarray:
    return np.asarray(record["cell_averages"][key], dtype=float) * scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=Path("p3c_comparison.json"))
    ap.add_argument("--png", type=Path, default=Path("p3c_uz_diagnostic_corrected.png"))
    ap.add_argument("--pdf", type=Path, default=Path("p3c_uz_diagnostic_corrected.pdf"))
    args = ap.parse_args()

    report = json.loads(args.comparison.read_text())
    runs = {r["run"]: r for r in report["runs"]}
    required = ("run_M16_raw", "run_M16_fplus", "run_M24_raw")
    missing = set(required) - set(runs)
    if missing:
        raise SystemExit(f"missing runs: {sorted(missing)}")

    ne = len(report["reference"]["cell_averages"]["uz"])
    xedge = np.linspace(-15.0, 15.0, ne + 1)
    xc = 0.5 * (xedge[:-1] + xedge[1:])
    u0 = float(report["nondim"]["u0"])
    ref = cell_profile(report["reference"], "uz", u0)
    final = {name: cell_profile(runs[name], "uz", u0) for name in required}
    benchmark = final["run_M24_raw"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.8,
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]},
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(xc, ref, "--", color="#888888", marker="D", ms=3.5,
            label=r"inherited state, $t=30$")
    for name in required:
        ax.plot(xc, final[name], color=COLORS[name], marker=MARKERS[name], ms=4,
                label=LABELS[name])
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.set_ylabel(r"cell-averaged $u_z$ [m s$^{-1}$]")
    ax.set_title("(a) Decay of inherited transverse velocity, t = 30 to 31", loc="left")
    ax.legend(ncol=2, frameon=False, loc="lower left")

    ax = axes[1]
    for name in required:
        ax.plot(xc, final[name] * 1e3, color=COLORS[name], marker=MARKERS[name], ms=5,
                label=LABELS[name])
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.set_ylabel(r"cell-averaged $u_z$ [mm s$^{-1}$]")
    ax.set_title("(b) Final profiles at t = 31 (magnified)", loc="left")
    ax.legend(ncol=3, frameon=False, loc="lower left")

    ax = axes[2]
    for name in ("run_M16_raw", "run_M16_fplus"):
        err = (final[name] - benchmark) * 1e3
        ax.plot(xc, err, color=COLORS[name], marker=MARKERS[name], ms=5,
                label=LABELS[name] + r" $-$ $M=24$")
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.set_ylabel(r"difference from $M=24$ [mm s$^{-1}$]")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_title("(c) Angular-resolution error in the cell averages", loc="left")
    ax.legend(frameon=False, loc="best")

    for ax in axes:
        ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-15.5, 15.5)

    fig.savefig(args.png, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    print(f"WROTE_PNG={args.png}")
    print(f"WROTE_PDF={args.pdf}")


if __name__ == "__main__":
    main()
