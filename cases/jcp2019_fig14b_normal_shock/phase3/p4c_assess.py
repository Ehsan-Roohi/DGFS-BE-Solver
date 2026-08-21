#!/usr/bin/env python3
"""Assess the P4C early-time projection-weighting scan."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

FIELDS = ("rms_rho_vs_M24", "rms_ux_vs_M24", "rms_T_vs_M24",
          "rms_qx_vs_M24", "rms_stress_vs_M24", "rms_uz_vs_M24")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a=ap.parse_args()
    rows={r["run"].removeprefix("run_"):r for r in csv.DictReader(a.csv.open())}
    needed=("M16_raw","M16_euclidean","M16_fplus","M24_raw")
    missing=[n for n in needed if n not in rows]
    if missing:
        raise SystemExit("P4C_MISSING_RUNS="+",".join(missing))
    raw=rows["M16_raw"]
    candidates=("M16_euclidean","M16_fplus")
    lines=["# P4C early-time weighting scan", "",
           "| method | max |u_z| [m/s] | raw/method u_z RMS | mean normalized core error | overshoot |",
           "|---|---:|---:|---:|---:|"]
    scores={}
    for name in candidates:
        r=rows[name]
        ratios=[float(raw[f])/max(float(r[f]),1e-300) for f in FIELDS]
        core=sum(float(r[f])/max(float(raw[f]),1e-300) for f in FIELDS[:5])/5
        over=max(float(r[k]) for k in ("rho_overshoot","ux_overshoot","T_overshoot"))
        scores[name]=(ratios[-1],core,over)
        lines.append(f"| {name} | {float(r['max_abs_uz_m_per_s']):.6e} | {ratios[-1]:.4f} | {core:.4f} | {over:.3e} |")
    # Pareto-oriented choice: suppress transverse error first, then minimize core penalty.
    eligible=[n for n in candidates if scores[n][0] > 1.0 and scores[n][2] <= 1e-10]
    if not eligible:
        raise SystemExit("P4C_NO_SAFE_UZ_IMPROVEMENT")
    selected=min(eligible, key=lambda n:(scores[n][1],-scores[n][0]))
    lines += ["", f"SELECTED_WEIGHTING={selected.removeprefix('M16_')}",
              "Selection rule: no macroscopic overshoot, improved u_z RMS, then minimum mean core-error ratio."]
    a.out.write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    print("P4C_ASSESSMENT_PASS")

if __name__=="__main__":
    main()
