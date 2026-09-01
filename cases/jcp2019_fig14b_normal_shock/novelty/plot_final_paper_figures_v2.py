#!/usr/bin/env python3
"""Paper-facing J14 figures/tables from existing data only.

No solver is run.  This version deliberately avoids bar charts.  Scalar
metrics are written as manuscript tables; only profiles, convergence histories,
and the conservation-defect trend are plotted.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

RUNS=("M6_raw","M6_fplus","M16_raw","M16_fplus")
LABELS={
    "M6_raw":r"$M_\omega=6$ raw",
    "M6_fplus":r"$M_\omega=6$ projected",
    "M16_raw":r"$M_\omega=16$ raw",
    "M16_fplus":r"$M_\omega=16$ projected",
}
COL={"M6_raw":"#1f4e79","M6_fplus":"#1f4e79","M16_raw":"#b22222","M16_fplus":"#2e8b57"}
LS={"M6_raw":"-","M6_fplus":"--","M16_raw":"-","M16_fplus":"--"}


def _cfg(path):
    c=configparser.ConfigParser(); c.optionxform=str; c.read(path); return c


def _ini_from_h5(v):
    c=configparser.ConfigParser(); c.optionxform=str
    c.read_string(v.decode() if isinstance(v,bytes) else str(v)); return c


def _basis(r): return np.vstack((0.5*r*(r-1),1-r*r,0.5*r*(r+1)))


def dense(ini,mesh_file,bulk):
    cfg=_cfg(ini)
    h0=cfg.getfloat("non-dim","H0")
    rho_lr=(cfg.getfloat("soln-bcs-left","rho"),cfg.getfloat("soln-bcs-right","rho"))
    with h5py.File(mesh_file,"r") as h: mesh=np.asarray(h["spt_line_p0"])
    with h5py.File(bulk,"r") as h:
        mom=np.asarray(h["moments_line_p0"],float); stats=_ini_from_h5(h["stats"][()])
    fields=[s.strip() for s in stats["data"]["fields"].split(",")]
    left=mesh[:,:,0].min(axis=0); right=mesh[:,:,0].max(axis=0); order=np.argsort(left)
    left,right,mom=left[order],right[order],mom[:,:,order]
    r=np.linspace(-1,1,241); B=_basis(r); xs=[]; vals={f:[] for f in fields}
    for e in range(mom.shape[2]):
        x=.5*((1-r)*left[e]+(1+r)*right[e]); xs.append(x)
        for j,f in enumerate(fields): vals[f].append(B.T@mom[:,j,e])
    x=np.concatenate(xs); vals={k:np.concatenate(v) for k,v in vals.items()}; ii=np.argsort(x)
    x=x[ii]; vals={k:v[ii] for k,v in vals.items()}
    target=.5*sum(rho_lr); y=vals["rho"]-target; hit=np.flatnonzero(y[:-1]*y[1:]<=0)
    if len(hit):
        j=hit[np.argmin(np.abs(x[hit]))]
        shock=float(x[j]+(target-vals["rho"][j])*(x[j+1]-x[j])/(vals["rho"][j+1]-vals["rho"][j]))
    else: shock=float(x[np.argmin(np.abs(y))])
    return x,vals,shock,h0


def residual(path):
    rows=[]
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try: q=(float(row["t"]),float(row["f"]),float(row["f_normalized"]))
            except (KeyError,ValueError,TypeError): continue
            if all(math.isfinite(v) for v in q): rows.append(q)
    a=np.asarray(rows,float)
    if len(a)>2:
        keep=np.ones(len(a),bool)
        for i in range(1,len(a)-1):
            if a[i,1] < 1e-3*min(a[i-1,1],a[i+1,1]): keep[i]=False
        a=a[keep]
    return a


def fig1(out):
    fig,ax=plt.subplots(figsize=(11.5,4.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(.5,.93,r"$B=[1,v_x,v_y,v_z,\frac{1}{2}|v|^2],\qquad w=\max(f,0)$",ha="center",fontsize=17)
    boxes=[(.04,.52,.20,.25,"Raw spectral collision",r"$Q(\mathbf{v})$"),
           (.30,.52,.20,.25,"Invariant defect",r"$m=c_wBQ$"),
           (.56,.52,.25,.25,"Weighted moment solve",r"$G\lambda=m$"),
           (.38,.10,.28,.25,"Conservative correction",r"$Q_c=Q-wB^T\lambda$")]
    for x,y,w,h,title,eq in boxes:
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",fc="white",ec="black",lw=1.5)
        ax.add_patch(p); ax.text(x+w/2,y+.68*h,title,ha="center",va="center",fontsize=11.5,fontweight="bold")
        ax.text(x+w/2,y+.30*h,eq,ha="center",va="center",fontsize=15)
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=15,lw=1.6))
    arrow((.24,.645),(.30,.645)); arrow((.50,.645),(.56,.645)); arrow((.685,.52),(.54,.35))
    ax.text(.5,.015,"The projection acts on the collision term Q; it neither clips f nor implies global positivity.",ha="center",fontsize=10.5)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG1_CONSERVATIVE_PROJECTION.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def prepare_cases(closeout,steady,tmp):
    paper=tmp/"paper_cases"; paper.mkdir()
    for name in ("M6_raw","M6_fplus"):
        src=closeout/"paper_cases"/name; dst=paper/name; dst.mkdir()
        shutil.copy2(src/"dgfs.ini",dst/"dgfs.ini"); shutil.copy2(src/"mesh.frfsm",dst/"mesh.frfsm"); shutil.copy2(src/"bulk-final.frfss",dst/"bulk-final.frfss")
    for name in ("M16_raw","M16_fplus"):
        src=steady/"stage_1"/name; dst=paper/name; dst.mkdir()
        shutil.copy2(src/f"p3b_{name}.ini",dst/"dgfs.ini"); shutil.copy2(src/"mesh.frfsm",dst/"mesh.frfsm"); shutil.copy2(src/f"bulksol_p3b_{name}-340.25.frfss",dst/"bulk-final.frfss")
    return paper


def fig2(out,closeout,steady):
    source=steady/"src"; val=source/"cases/jcp2019_fig14b_validation"
    import sys
    sys.path.insert(0,str(val)); import compare_fig14 as cf
    with tempfile.TemporaryDirectory(prefix="dgfs_fig2_") as td:
        paper=prepare_cases(closeout,steady,Path(td))
        cases={n:cf.load_case(paper/n) for n in RUNS}
        refs,symbols=cf.load_reference(val/"fig14_digitized.csv")
    props=("rho","T","u"); titles={"rho":r"Density $\rho'$","T":r"Temperature $T'$","u":r"Velocity $u'$"}
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.25),sharex=True,sharey=True,constrained_layout=True)
    for ax,p in zip(axs,props):
        for seg in refs[(8,p)]:
            x=np.array([q[0] for q in seg]); y=np.array([q[1] for q in seg]); ax.plot(x,y,color="0.55",lw=1.4,label="Alexeenko DGFS" if p=="rho" else None)
        sx=np.array([q[0] for q in symbols[p]]); sy=np.array([q[1] for q in symbols[p]])
        ax.scatter(sx,sy,s=22,facecolors="white",edgecolors="black",linewidths=.9,label="Ohwada" if p=="rho" else None,zorder=5)
        for name in RUNS:
            first=True
            for x,y in cases[name]["segments"][p]:
                ax.plot(x,y,color=COL[name],ls=LS[name],lw=1.8,label=LABELS[name] if p=="rho" and first else None)
                first=False
        ax.set_title(titles[p]); ax.set_xlim(-8,6); ax.set_ylim(-.03,1.03); ax.grid(alpha=.18); ax.set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[0].set_ylabel("Normalized property")
    handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.08),fontsize=9)
    fig.suptitle(r"Helium normal shock, Mach 1.59, Kn $\approx 0.055$: reference-profile validation",y=1.15,fontsize=15)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG2_MACH1P59_REFERENCE_PROFILES.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def fig3(out,audits):
    raw=np.array([audits[6]["summary"]["raw_max_defect"],audits[16]["summary"]["raw_max_defect"]])
    fp=np.array([audits[6]["summary"]["fplus_max_defect"],audits[16]["summary"]["fplus_max_defect"]])
    x=np.array([6,16],float)
    fig,ax=plt.subplots(figsize=(7.1,4.8),constrained_layout=True)
    ax.semilogy(x,raw,"o-",lw=2,ms=7,label="Raw collision")
    ax.semilogy(x,fp,"s--",lw=2,ms=7,label="Projected collision")
    for xi,a,b in zip(x,raw,fp): ax.annotate("",xy=(xi,b*2.2),xytext=(xi,a/2.2),arrowprops=dict(arrowstyle="->",lw=1.2,color="0.35"))
    ax.axhline(5e-12,ls=":",lw=1.5,color="0.35",label=r"Acceptance level $5\times10^{-12}$")
    ax.set_xticks([6,16],[r"$M_\omega=6$",r"$M_\omega=16$"]); ax.set_ylabel("Maximum collision-invariant defect"); ax.set_title("Conservative projection reduces invariant defects to roundoff")
    ax.grid(alpha=.2,which="both"); ax.legend(frameon=False)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG3_CONSERVATION_DEFECT.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def table1(out,report,audits):
    ss=report["steady_state"]
    header=["run","M_omega","mode","paper_mean_rms","max_collision_defect","projection_overhead_pct","max_overshoot","negative_mass_fraction","final_normalized_residual"]
    rows=[]
    for name in RUNS:
        r=report["runs"][name]["summary"]; M=6 if name.startswith("M6") else 16; mode="projected" if name.endswith("fplus") else "raw"
        over=0.0 if mode=="raw" else 100*(audits[M]["summary"]["median_projection_overhead_ratio"]-1)
        ov=max(r["rho_overshoot"],r["u_overshoot"],r["T_overshoot"])
        rows.append([name,M,mode,r["paper_mean_rms"],r["collision_max_defect"],over,ov,r["negative_mass_fraction"],ss[name]["normalized_residual"]])
    with (out/"TABLE1_CORE_METRICS.csv").open("w",newline="") as f: w=csv.writer(f); w.writerow(header); w.writerows(rows)
    md=["| Run | $M_\\omega$ | Mode | Mean RMS | Max invariant defect | Projection overhead | Max overshoot | Negative mass | Final $r_n/r_1$ |","|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4e} | {r[4]:.3e} | {r[5]:.2f}% | {r[6]:.3e} | {r[7]:.3e} | {r[8]:.4f} |")
    (out/"TABLE1_CORE_METRICS.md").write_text("\n".join(md)+"\n")
    tex=[r"\begin{tabular}{lrrrrrr}",r"\hline",r"Case & RMS & defect & overhead (\%) & overshoot & neg. mass & final $r_n/r_1$ \\",r"\hline"]
    for r in rows: tex.append(f"{r[0].replace('_',r'\_')} & {r[3]:.4e} & {r[4]:.3e} & {r[5]:.2f} & {r[6]:.3e} & {r[7]:.3e} & {r[8]:.4f} \\")
    tex += [r"\hline",r"\end{tabular}"]
    (out/"TABLE1_CORE_METRICS.tex").write_text("\n".join(tex)+"\n")


def table2(out,steady):
    fields=["rho","U:x","T","Q:x","p"]; names=["rho","u_x","T","q_x","p"]
    rows=[]
    for case in ("M16_raw","M16_fplus"):
        base=steady/"stage_1"/case
        _,a,_,_=dense(base/f"p3b_{case}.ini",base/"mesh.frfsm",base/f"bulksol_p3b_{case}-335.25.frfss")
        _,b,_,_=dense(base/f"p3b_{case}.ini",base/"mesh.frfsm",base/f"bulksol_p3b_{case}-340.25.frfss")
        for f,n in zip(fields,names): rows.append([case,n,float(np.linalg.norm(b[f]-a[f])/max(np.linalg.norm(a[f]),1e-300))])
    with (out/"TABLE2_FINAL_STATIONARITY.csv").open("w",newline="") as q: w=csv.writer(q); w.writerow(["case","field","relative_L2_335p25_to_340p25"]); w.writerows(rows)
    md=["| Case | Field | Relative $L_2$ change, 335.25→340.25 |","|---|---|---:|"]+[f"| {a} | {b} | {c:.4e} |" for a,b,c in rows]
    (out/"TABLE2_FINAL_STATIONARITY.md").write_text("\n".join(md)+"\n")


def supp_residual(out,closeout,steady):
    paths={"M6_raw":closeout/"final_runs/run_M6_raw/kinetic_residual_p3b.csv","M6_fplus":closeout/"final_runs/run_M6_fplus/kinetic_residual_p3b.csv","M16_raw":steady/"stage_1/M16_raw/kinetic_residual_p3b.csv","M16_fplus":steady/"stage_1/M16_fplus/kinetic_residual_p3b.csv"}
    fig,ax=plt.subplots(figsize=(7.5,4.7),constrained_layout=True)
    for name,p in paths.items():
        a=residual(p); ax.semilogy(a[:,0],a[:,2],color=COL[name],ls=LS[name],lw=1.8,label=LABELS[name])
    ax.axhline(1,color="0.35",ls=":",label="steady-state threshold"); ax.set_xlabel("Time"); ax.set_ylabel(r"Normalized kinetic residual $r_n/r_1$"); ax.set_title("Steady-state closeout"); ax.grid(alpha=.2,which="both"); ax.legend(frameon=False,fontsize=9)
    for ext in ("png","pdf"): fig.savefig(out/f"SUPP_FIG_S1_RESIDUAL_HISTORY.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def fig4(out,steady):
    data={}
    for name in ("M16_raw","M16_fplus"):
        b=steady/"stage_1"/name; data[name]=dense(b/f"p3b_{name}.ini",b/"mesh.frfsm",b/f"bulksol_p3b_{name}-340.25.frfss")
    xs=data["M16_raw"][2]; h0=data["M16_raw"][3]
    fig,ax=plt.subplots(1,2,figsize=(11.4,4.5),constrained_layout=True)
    for name in ("M16_raw","M16_fplus"):
        x,v,_,_=data[name]; X=(x-xs)*h0/h0
        ax[0].plot(X,v["Q:x"],color=COL[name],ls=LS[name],lw=2,label=LABELS[name]); ax[1].plot(X,v["U:y"],color=COL[name],ls=LS[name],lw=2,label=LABELS[name])
    ax[0].set_title("Heat flux"); ax[0].set_ylabel(r"$q_x$ [W m$^{-2}$]"); ax[1].set_title("Transverse-velocity diagnostic"); ax[1].set_ylabel(r"$u_y$ [m s$^{-1}$]")
    for a in ax: a.set_xlabel(r"$(x-x_{s,raw})/H_0$"); a.grid(alpha=.18)
    ax[1].legend(frameon=False,fontsize=9); fig.suptitle(r"Mach 1.59, $M_\omega=16$: projection-sensitive quantities")
    for ext in ("png","pdf"): fig.savefig(out/f"FIG4_PROJECTION_SENSITIVE_PROFILES.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--closeout",type=Path,required=True); ap.add_argument("--steady",type=Path,required=True); ap.add_argument("--output-dir",type=Path,default=Path.cwd()); a=ap.parse_args()
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); close=a.closeout.resolve(); steady=a.steady.resolve()
    report=json.loads((out/"novelty_report.json").read_text()); audits={6:json.loads((close/"audit/M6.json").read_text()),16:json.loads((close/"audit/M16.json").read_text())}
    fig1(out); fig2(out,close,steady); fig3(out,audits); table1(out,report,audits); table2(out,steady); supp_residual(out,close,steady); fig4(out,steady)
    bundle=out/"DGFS_PAPER_FIGURES_TABLES_V2.zip"; bundle.unlink(missing_ok=True)
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and (p.name.startswith("FIG") or p.name.startswith("SUPP_FIG") or p.name.startswith("TABLE") or p.name in {"FINAL_STEADY_GATE_SUMMARY.md","novelty_report.json","CLOSEOUT_STATUS.json"}): z.write(p,p.name)
    print(f"BUNDLE={bundle}")
    for p in sorted(out.glob("FIG*")): print(p.name)
    for p in sorted(out.glob("TABLE*")): print(p.name)

if __name__=="__main__": main()
