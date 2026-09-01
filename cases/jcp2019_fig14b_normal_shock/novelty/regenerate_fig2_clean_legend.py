#!/usr/bin/env python3
"""Regenerate only the Mach-1.59 reference-profile figure with a clean manual legend.

Uses existing DGFS data only.  No solver run is launched.  Outputs are written
straight to --output-dir.
"""
from __future__ import annotations

import argparse
import configparser
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RUNS = ("M6_raw", "M6_fplus", "M16_raw", "M16_fplus")
LABELS = {
    "M6_raw": r"$M_\omega=6$ raw",
    "M6_fplus": r"$M_\omega=6$ projected",
    "M16_raw": r"$M_\omega=16$ raw",
    "M16_fplus": r"$M_\omega=16$ projected",
}
COL = {
    "M6_raw": "#1f4e79",
    "M6_fplus": "#1f4e79",
    "M16_raw": "#b22222",
    "M16_fplus": "#2e8b57",
}
LS = {"M6_raw": "-", "M6_fplus": "--", "M16_raw": "-", "M16_fplus": "--"}


def prepare_cases(closeout: Path, steady: Path, tmp: Path) -> Path:
    paper = tmp / "paper_cases"
    paper.mkdir()
    for name in ("M6_raw", "M6_fplus"):
        src = closeout / "paper_cases" / name
        dst = paper / name
        dst.mkdir()
        shutil.copy2(src / "dgfs.ini", dst / "dgfs.ini")
        shutil.copy2(src / "mesh.frfsm", dst / "mesh.frfsm")
        shutil.copy2(src / "bulk-final.frfss", dst / "bulk-final.frfss")
    for name in ("M16_raw", "M16_fplus"):
        src = steady / "stage_1" / name
        dst = paper / name
        dst.mkdir()
        shutil.copy2(src / f"p3b_{name}.ini", dst / "dgfs.ini")
        shutil.copy2(src / "mesh.frfsm", dst / "mesh.frfsm")
        shutil.copy2(src / f"bulksol_p3b_{name}-340.25.frfss", dst / "bulk-final.frfss")
    return paper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closeout", type=Path, required=True)
    ap.add_argument("--steady", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()

    closeout = args.closeout.resolve()
    steady = args.steady.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    validation = steady / "src/cases/jcp2019_fig14b_validation"
    sys.path.insert(0, str(validation))
    import compare_fig14 as cf

    with tempfile.TemporaryDirectory(prefix="dgfs_fig2_clean_") as td:
        paper = prepare_cases(closeout, steady, Path(td))
        cases = {name: cf.load_case(paper / name) for name in RUNS}
        refs, symbols = cf.load_reference(validation / "fig14_digitized.csv")

    props = ("rho", "T", "u")
    titles = {"rho": r"Density $\rho'$", "T": r"Temperature $T'$", "u": r"Velocity $u'$"}

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.72, wspace=0.08)

    for ax, prop in zip(axs, props):
        # Reference line: draw every digitized segment, but never attach labels.
        for seg in refs[(8, prop)]:
            x = np.asarray([q[0] for q in seg], float)
            y = np.asarray([q[1] for q in seg], float)
            ax.plot(x, y, color="0.55", lw=1.35, zorder=1)

        sx = np.asarray([q[0] for q in symbols[prop]], float)
        sy = np.asarray([q[1] for q in symbols[prop]], float)
        ax.scatter(sx, sy, s=24, facecolors="white", edgecolors="black",
                   linewidths=0.9, zorder=5)

        for name in RUNS:
            for x, y in cases[name]["segments"][prop]:
                ax.plot(x, y, color=COL[name], ls=LS[name], lw=1.9, zorder=3)

        ax.set_title(titles[prop], fontsize=13, pad=8)
        ax.set_xlim(-8, 6)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.18)
        ax.set_xlabel(r"$(x-x_s)/\lambda_1$", fontsize=11)
        ax.tick_params(labelsize=10)

    axs[0].set_ylabel("Normalized property", fontsize=11)

    legend_handles = [
        Line2D([0], [0], color="0.55", lw=1.5, label="Alexeenko DGFS"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor="black", markersize=5.5, label="Ohwada"),
        Line2D([0], [0], color=COL["M6_raw"], ls="-", lw=2.0, label=LABELS["M6_raw"]),
        Line2D([0], [0], color=COL["M6_fplus"], ls="--", lw=2.0, label=LABELS["M6_fplus"]),
        Line2D([0], [0], color=COL["M16_raw"], ls="-", lw=2.0, label=LABELS["M16_raw"]),
        Line2D([0], [0], color=COL["M16_fplus"], ls="--", lw=2.0, label=LABELS["M16_fplus"]),
    ]

    fig.suptitle(
        r"Helium normal shock, Mach 1.59, Kn $\approx 0.055$: reference-profile validation",
        y=0.985, fontsize=15,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"FIG2_MACH1P59_REFERENCE_PROFILES.{ext}",
            dpi=320 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"FIG2_PNG={out/'FIG2_MACH1P59_REFERENCE_PROFILES.png'}")
    print(f"FIG2_PDF={out/'FIG2_MACH1P59_REFERENCE_PROFILES.pdf'}")


if __name__ == "__main__":
    main()
