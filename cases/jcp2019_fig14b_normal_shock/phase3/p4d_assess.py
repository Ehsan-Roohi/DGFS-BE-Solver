#!/usr/bin/env python3
"""Assess the P4D symmetry-aware transverse-moment projection experiment."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

CORE=("rms_rho_vs_M24","rms_ux_vs_M24","rms_T_vs_M24",
      "rms_qx_vs_M24","rms_stress_vs_M24")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rows={r["run"].removeprefix("run_"):r for r in csv.DictReader(a.csv.open())}
    needed=("M16_raw","M16_fplus","M16_transverse","M24_raw")
    missing=[n for n in needed if n not in rows]
    if missing: raise SystemExit("P4D_MISSING_RUNS="+",".join(missing))
    raw=rows["M16_raw"]
    lines=["# P4D symmetry-aware projection", "",
      "| method | mean core error/raw | raw/method u_z RMS | max |u_z| [m/s] | negmass/raw | wall/raw | overshoot |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    metrics={}
    for name in ("M16_fplus","M16_transverse"):
        r=rows[name]
        core=sum(float(r[f])/max(float(raw[f]),1e-300) for f in CORE)/len(CORE)
        uz=float(raw["rms_uz_vs_M24"])/max(float(r["rms_uz_vs_M24"]),1e-300)
        neg=float(r["max_negmass_frac"])/max(float(raw["max_negmass_frac"]),1e-300)
        wall=float(r["wall_seconds_job"])/float(raw["wall_seconds_job"])
        over=max(float(r[k]) for k in ("rho_overshoot","ux_overshoot","T_overshoot"))
        metrics[name]=(core,uz,neg,wall,over)
        lines.append(f"| {name} | {core:.4f} | {uz:.4f} | {float(r['max_abs_uz_m_per_s']):.6e} | {neg:.4f} | {wall:.4f} | {over:.3e} |")
    core,uz,neg,wall,over=metrics["M16_transverse"]
    supported=(uz>1.0 and core<metrics["M16_fplus"][0] and over<=1e-10)
    lines += ["",f"TRANSVERSE_HYPOTHESIS_SUPPORTED={'yes' if supported else 'no'}",
      "Interpretation: the diagnostic is successful if transverse-only correction suppresses spurious u_z while reducing the five-moment fplus core-profile penalty.",
      "This mode enforces only symmetry-forbidden transverse collision moments; it is a diagnostic, not yet the production five-invariant conservative scheme."]
    a.out.write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    print("P4D_ASSESSMENT_COMPLETE")

if __name__=="__main__": main()
