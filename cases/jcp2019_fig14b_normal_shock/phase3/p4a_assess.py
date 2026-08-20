#!/usr/bin/env python3
"""Assess clean-start transverse-velocity generation in Phase 4A."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def profile(r,k):
    return [r["points"][u][e][k] for e in range(len(r["points"][0])) for u in range(len(r["points"]))]

def svg(path,report):
    W,H=1000,620; L,R,T,B=110,35,55,90; pw,ph=W-L-R,H-T-B
    ne=len(report["reference"]["points"][0]); xs=[]
    for e in range(ne):
        a,b=-15+30*e/ne,-15+30*(e+1)/ne; xs += [a,(a+b)/2,b]
    u0=report["nondim"]["u0"]; colors={"run_M6_raw":"#aa5149","run_M16_fplus":"#355da8","run_M24_raw":"#111111"}
    curves=[("t=0 reference",[v*u0 for v in profile(report["reference"],"uz")],"#888","5,5")]
    curves += [(r["run"].replace("run_",""),[v*u0 for v in profile(r,"uz")],colors[r["run"]],"") for r in report["runs"]]
    lo=min(min(y) for _,y,_,_ in curves); hi=max(max(y) for _,y,_,_ in curves); p=max(.07*(hi-lo),1e-6); lo-=p; hi+=p
    xp=lambda x:L+(x+15)/30*pw; yp=lambda y:T+(hi-y)/(hi-lo)*ph
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial;fill:#111}.t{font-size:17px}.l{font-size:22px}.g{font-size:18px}</style>']
    for x in (-15,-10,-5,0,5,10,15):
        out += [f'<line x1="{xp(x):.2f}" y1="{T}" x2="{xp(x):.2f}" y2="{T+ph}" stroke="#ddd"/>',f'<text class="t" x="{xp(x):.2f}" y="{T+ph+30}" text-anchor="middle">{x}</text>']
    for i in range(6):
        y=lo+i*(hi-lo)/5; out += [f'<line x1="{L}" y1="{yp(y):.2f}" x2="{L+pw}" y2="{yp(y):.2f}" stroke="#ddd"/>',f'<text class="t" x="{L-12}" y="{yp(y)+6:.2f}" text-anchor="end">{y:.3e}</text>']
    out += [f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#111"/>',f'<text class="l" x="{L+pw/2}" y="{H-25}" text-anchor="middle">x [mm]</text>',f'<text class="l" transform="translate(28 {T+ph/2}) rotate(-90)" text-anchor="middle">u_z [m s^-1]</text>']
    for n,y,c,d in curves:
        pts=" ".join(f"{xp(x):.2f},{yp(v):.2f}" for x,v in zip(xs,y)); ds=f' stroke-dasharray="{d}"' if d else ""; out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.6"{ds}/>')
    for i,(n,_,c,d) in enumerate(curves):
        y=T+25+27*i; ds=f' stroke-dasharray="{d}"' if d else ""; out += [f'<line x1="{L+18}" y1="{y}" x2="{L+56}" y2="{y}" stroke="{c}" stroke-width="3"{ds}/>',f'<text class="g" x="{L+66}" y="{y+6}">{n}</text>']
    out.append("</svg>"); path.write_text("\n".join(out)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--comparison",type=Path,default=Path("p4a_comparison.json")); p.add_argument("--json",type=Path,default=Path("p4a_metrics.json")); p.add_argument("--csv",type=Path,default=Path("p4a_metrics.csv")); p.add_argument("--md",type=Path,default=Path("p4a_metrics.md")); p.add_argument("--svg",type=Path,default=Path("p4a_uz_generation.svg")); a=p.parse_args()
    report=json.loads(a.comparison.read_text()); required={"run_M6_raw","run_M16_fplus","run_M24_raw"}
    if {r["run"] for r in report["runs"]} != required: raise SystemExit("P4A_RUN_SET_INVALID")
    rows=[]
    for r in report["runs"]:
        ov=max(r["monotone_overshoot"][k]["fraction_of_jump"] for k in ("rho","ux","T"))
        row={"run":r["run"],"M":r["M"],"projection":r["projection"],"tcurr":r["tcurr"],"wall_seconds":r["wall_seconds_job"],"max_abs_uz_m_per_s":r["max_abs_uz_m_per_s"],"max_rho_ux_T_overshoot_fraction":ov,"rms_rho_vs_M24":r["profile_rms_diff_vs_M24_raw"]["rho"],"rms_ux_vs_M24":r["profile_rms_diff_vs_M24_raw"]["ux"],"rms_T_vs_M24":r["profile_rms_diff_vs_M24_raw"]["T"],"rms_qx_vs_M24":r["profile_rms_diff_vs_M24_raw"]["qx"],"rms_stress_vs_M24":r["profile_rms_diff_vs_M24_raw"]["Pxx_minus_p"],"rms_uz_vs_M24":r["profile_rms_diff_vs_M24_raw"]["uz"]}
        if not all(math.isfinite(float(v)) for k,v in row.items() if k not in ("run","projection")): raise SystemExit(f"P4A_NONFINITE {r['run']}")
        if abs(float(r["tcurr"])-1)>2e-8: raise SystemExit(f"P4A_BAD_FINAL_TIME {r['run']}")
        if ov>1e-8: raise SystemExit(f"P4A_MONOTONE_OVERSHOOT {r['run']} {ov}")
        rows.append(row)
    m6=next(x for x in rows if x["run"]=="run_M6_raw"); plus=next(x for x in rows if x["run"]=="run_M16_fplus")
    metrics={"runs":rows,"M6_to_fplus_max_uz_ratio":m6["max_abs_uz_m_per_s"]/max(plus["max_abs_uz_m_per_s"],1e-300),"fplus_closer_to_M24_in_uz":plus["rms_uz_vs_M24"]<m6["rms_uz_vs_M24"]}
    a.json.write_text(json.dumps(metrics,indent=2)+"\n")
    with a.csv.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=["| run | max abs uz [m/s] | max overshoot | RMS uz vs M24 | wall [s] |","|---|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['run']} | {r['max_abs_uz_m_per_s']:.6e} | {r['max_rho_ux_T_overshoot_fraction']:.3e} | {r['rms_uz_vs_M24']:.3e} | {r['wall_seconds']:.1f} |")
    a.md.write_text("\n".join(lines)+"\n"); svg(a.svg,report); print("\n".join(lines)); print(f"M6_TO_FPLUS_MAX_UZ_RATIO={metrics['M6_to_fplus_max_uz_ratio']:.8e}"); print(f"FPLUS_CLOSER_TO_M24_IN_UZ={str(metrics['fplus_closer_to_M24_in_uz']).lower()}"); print("P4A_ASSESSMENT_PASS")
if __name__=="__main__": main()
