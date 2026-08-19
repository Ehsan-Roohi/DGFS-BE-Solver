#!/usr/bin/env python3
"""Assess Phase 3C transverse-velocity decay and emit a dependency-free SVG."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def ordered_profile(record, key):
    return [record["points"][u][e][key] for e in range(len(record["points"][0])) for u in range(len(record["points"]))]

def svg_plot(path, report):
    width, height = 1000, 620
    left, right, top, bottom = 105, 35, 55, 90
    pw, ph = width-left-right, height-top-bottom
    ne = len(report["reference"]["points"][0])
    xs = []
    for e in range(ne):
        xl, xr = -15.0+30.0*e/ne, -15.0+30.0*(e+1)/ne
        xs.extend((xl, 0.5*(xl+xr), xr))
    u0 = report["nondim"]["u0"]
    curves = [("t=30 reference", ordered_profile(report["reference"], "uz"), "#777777", "5,5")]
    colors = {"run_M16_raw":"#d07a3e", "run_M16_fplus":"#355da8", "run_M24_raw":"#111111"}
    for run in report["runs"]:
        curves.append((run["run"], ordered_profile(run, "uz"), colors[run["run"]], ""))
    curves = [(n,[v*u0 for v in y],c,d) for n,y,c,d in curves]
    ymin, ymax = min(min(y) for _,y,_,_ in curves), max(max(y) for _,y,_,_ in curves)
    pad=max(.05*(ymax-ymin),.02); ymin-=pad; ymax+=pad
    xp=lambda x:left+(x+15.0)/30.0*pw
    yp=lambda y:top+(ymax-y)/(ymax-ymin)*ph
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#111}.tick{font-size:17px}.label{font-size:22px}.legend{font-size:18px}</style>']
    for x in (-15,-10,-5,0,5,10,15):
        out += [f'<line x1="{xp(x):.2f}" y1="{top}" x2="{xp(x):.2f}" y2="{top+ph}" stroke="#ddd"/>',
                f'<text class="tick" x="{xp(x):.2f}" y="{top+ph+30}" text-anchor="middle">{x}</text>']
    for i in range(6):
        y=ymin+i*(ymax-ymin)/5
        out += [f'<line x1="{left}" y1="{yp(y):.2f}" x2="{left+pw}" y2="{yp(y):.2f}" stroke="#ddd"/>',
                f'<text class="tick" x="{left-12}" y="{yp(y)+6:.2f}" text-anchor="end">{y:.3f}</text>']
    out += [f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#111"/>',
            f'<text class="label" x="{left+pw/2}" y="{height-25}" text-anchor="middle">x [mm]</text>',
            f'<text class="label" transform="translate(28 {top+ph/2}) rotate(-90)" text-anchor="middle">u_z [m s^-1]</text>']
    for n,y,c,dash in curves:
        pts=" ".join(f"{xp(x):.2f},{yp(v):.2f}" for x,v in zip(xs,y))
        ds=f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.6"{ds}/>')
    for i,(n,_,c,dash) in enumerate(curves):
        ly=top+25+i*27; ds=f' stroke-dasharray="{dash}"' if dash else ""
        out += [f'<line x1="{left+18}" y1="{ly}" x2="{left+56}" y2="{ly}" stroke="{c}" stroke-width="3"{ds}/>',
                f'<text class="legend" x="{left+66}" y="{ly+6}">{n.replace("run_","")}</text>']
    out.append("</svg>"); path.write_text("\n".join(out)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comparison",type=Path,default=Path("p3c_comparison.json"))
    ap.add_argument("--json",type=Path,default=Path("p3c_metrics.json"))
    ap.add_argument("--csv",type=Path,default=Path("p3c_metrics.csv"))
    ap.add_argument("--md",type=Path,default=Path("p3c_metrics.md"))
    ap.add_argument("--svg",type=Path,default=Path("p3c_uz_decay.svg"))
    a=ap.parse_args(); report=json.loads(a.comparison.read_text())
    refuz=report["reference"]["max_abs_uz_m_per_s"]
    required={"run_M16_raw","run_M16_fplus","run_M24_raw"}
    if {r["run"] for r in report["runs"]} != required: raise SystemExit("P3C_RUN_SET_INVALID")
    rows=[]
    for r in report["runs"]:
        maxuz=r["max_abs_uz_m_per_s"]
        overs=max(r["monotone_overshoot"][k]["fraction_of_jump"] for k in ("rho","ux","T"))
        row={"run":r["run"],"M":r["M"],"projection":r["projection"],"tcurr":r["tcurr"],
             "wall_seconds":r["wall_seconds_job"],"max_abs_uz_m_per_s":maxuz,
             "uz_decay_fraction_vs_t30":1.0-maxuz/refuz,"max_rho_ux_T_overshoot_fraction":overs,
             "rms_uz_vs_M24":r["profile_rms_diff_vs_M24_raw"]["uz"],
             "rms_qx_vs_M24":r["profile_rms_diff_vs_M24_raw"]["qx"],
             "rms_stress_vs_M24":r["profile_rms_diff_vs_M24_raw"]["Pxx_minus_p"]}
        if not all(math.isfinite(float(v)) for k,v in row.items() if k not in ("run","projection")): raise SystemExit(f"P3C_NONFINITE {r['run']}")
        if abs(float(r["tcurr"])-31.0)>2e-8: raise SystemExit(f"P3C_BAD_FINAL_TIME {r['run']}")
        if overs>1e-8: raise SystemExit(f"P3C_MONOTONE_OVERSHOOT {r['run']} {overs}")
        rows.append(row)
    raw=next(x for x in rows if x["run"]=="run_M16_raw")
    plus=next(x for x in rows if x["run"]=="run_M16_fplus")
    metrics={"reference_max_abs_uz_m_per_s":refuz,"runs":rows,
             "fplus_to_raw_uz_rms_ratio":plus["rms_uz_vs_M24"]/max(raw["rms_uz_vs_M24"],1e-300),
             "fplus_closer_to_M24_in_uz":plus["rms_uz_vs_M24"]<raw["rms_uz_vs_M24"]}
    a.json.write_text(json.dumps(metrics,indent=2)+"\n")
    with a.csv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=["| run | max abs uz [m/s] | decay vs t30 | max overshoot | RMS uz vs M24 | wall [s] |",
           "|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"| {x['run']} | {x['max_abs_uz_m_per_s']:.6e} | {x['uz_decay_fraction_vs_t30']:.3%} | {x['max_rho_ux_T_overshoot_fraction']:.3e} | {x['rms_uz_vs_M24']:.3e} | {x['wall_seconds']:.1f} |")
    a.md.write_text("\n".join(lines)+"\n"); svg_plot(a.svg,report)
    print("\n".join(lines)); print(f"FPLUS_CLOSER_TO_M24_IN_UZ={str(metrics['fplus_closer_to_M24_in_uz']).lower()}"); print("P3C_ASSESSMENT_PASS")
if __name__=="__main__": main()
