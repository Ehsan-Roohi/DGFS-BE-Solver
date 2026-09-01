#!/usr/bin/env python3
"""Build the final J14 paper package with no bar charts.

Main figures: method schematic, reference-profile validation, and
projection-sensitive profiles. Scalar metrics and stationarity are tables.
Residual history is supplementary. No DGFS solve is launched.
"""
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import plot_final_paper_figures_v2 as v2

OLD=(
"FIG1_METHOD_CONSERVATIVE_PROJECTION.png","FIG1_METHOD_CONSERVATIVE_PROJECTION.pdf",
"FIG2_MACH1P59_REFERENCE_PROFILES.svg",
"FIG3_COLLISION_INVARIANT_DEFECT.png","FIG3_COLLISION_INVARIANT_DEFECT.pdf",
"FIG3_CONSERVATION_DEFECT.png","FIG3_CONSERVATION_DEFECT.pdf",
"FIG4_ACCURACY_AND_OVERHEAD.png","FIG4_ACCURACY_AND_OVERHEAD.pdf",
"FIG5_STEADY_STATE_EVIDENCE.png","FIG5_STEADY_STATE_EVIDENCE.pdf",
"FIG6_MW16_SENSITIVE_QX_UY.png","FIG6_MW16_SENSITIVE_QX_UY.pdf",
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--closeout",type=Path,required=True)
    ap.add_argument("--steady",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=Path.cwd())
    a=ap.parse_args()
    out=a.output_dir.resolve(); close=a.closeout.resolve(); steady=a.steady.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for n in OLD: (out/n).unlink(missing_ok=True)

    report=json.loads((out/"novelty_report.json").read_text())
    audits={6:json.loads((close/"audit/M6.json").read_text()),16:json.loads((close/"audit/M16.json").read_text())}

    v2.fig1(out)
    v2.fig2(out,close,steady)
    v2.table1(out,report,audits)
    v2.table2(out,steady)
    v2.supp_residual(out,close,steady)
    v2.fig4(out,steady)

    # In the final manuscript the sensitive-profile panel is Figure 3.
    for ext in ("png","pdf"):
        src=out/f"FIG4_PROJECTION_SENSITIVE_PROFILES.{ext}"
        dst=out/f"FIG3_PROJECTION_SENSITIVE_PROFILES.{ext}"
        src.replace(dst)

    names=[
      "FIG1_CONSERVATIVE_PROJECTION.png","FIG1_CONSERVATIVE_PROJECTION.pdf",
      "FIG2_MACH1P59_REFERENCE_PROFILES.png","FIG2_MACH1P59_REFERENCE_PROFILES.pdf",
      "FIG3_PROJECTION_SENSITIVE_PROFILES.png","FIG3_PROJECTION_SENSITIVE_PROFILES.pdf",
      "TABLE1_CORE_METRICS.csv","TABLE1_CORE_METRICS.md","TABLE1_CORE_METRICS.tex",
      "TABLE2_FINAL_STATIONARITY.csv","TABLE2_FINAL_STATIONARITY.md",
      "SUPP_FIG_S1_RESIDUAL_HISTORY.png","SUPP_FIG_S1_RESIDUAL_HISTORY.pdf",
      "FINAL_STEADY_GATE_SUMMARY.md","novelty_report.json","CLOSEOUT_STATUS.json",
    ]
    bundle=out/"DGFS_PAPER_FINAL_NO_BARS.zip"; bundle.unlink(missing_ok=True)
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for n in names:
            p=out/n
            if p.exists(): z.write(p,n)
    print("FINAL_PACKAGE_NO_BARS")
    for n in names:
        p=out/n
        if p.exists(): print(n)
    print(f"BUNDLE={bundle}")

if __name__=="__main__": main()
