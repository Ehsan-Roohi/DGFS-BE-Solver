#!/usr/bin/env python3
"""Create paper-facing figures from the existing J14/M16 data only.

No DGFS solve is launched.  FIG2 is produced by finalize_existing_steady_gate.py;
this script creates FIG1 and FIG3--FIG6 directly in --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = ("M6_raw", "M6_fplus", "M16_raw", "M16_fplus")
PRETTY = {
    "M6_raw": r"$M_\omega=6$ raw", "M6_fplus": r"$M_\omega=6$ fplus",
    "M16_raw": r"$M_\omega=16$ raw", "M16_fplus": r"$M_\omega=16$ fplus",
}
COLORS = {"M6_raw":"#222222", "M6_fplus":"#2b6cb0", "M16_raw":"#c53030", "M16_fplus":"#2f855a"}
LINES = {"M6_raw":"-", "M6_fplus":"--", "M16_raw":"-", "M16_fplus":"--"}


def ini_from_h5(value):
    import configparser
    cfg = configparser.ConfigParser(); cfg.optionxform = str
    cfg.read_string(value.decode() if isinstance(value, bytes) else str(value))
    return cfg


def basis(r):
    return np.vstack((0.5*r*(r-1.0), 1.0-r*r, 0.5*r*(r+1.0)))


def dense(ini: Path, mesh_file: Path, bulk: Path):
    import configparser
    cfg = configparser.ConfigParser(); cfg.optionxform = str; cfg.read(ini)
    rho_lr = (cfg.getfloat("soln-bcs-left","rho"), cfg.getfloat("soln-bcs-right","rho"))
    with h5py.File(mesh_file,"r") as h: mesh = np.asarray(h["spt_line_p0"])
    with h5py.File(bulk,"r") as h:
        mom = np.asarray(h["moments_line_p0"], float); stats = ini_from_h5(h["stats"][()])
    fields = [x.strip() for x in stats["data"]["fields"].split(",")]
    left = mesh[:,:,0].min(axis=0); right = mesh[:,:,0].max(axis=0); order = np.argsort(left)
    left,right,mom = left[order],right[order],mom[:,:,order]
    r = np.linspace(-1,1,241); b = basis(r); xx=[]; vals={f:[] for f in fields}
    for e in range(mom.shape[2]):
        x = 0.5*((1-r)*left[e] + (1+r)*right[e]); xx.append(x)
        for j,f in enumerate(fields): vals[f].append(b.T @ mom[:,j,e])
    x = np.concatenate(xx); vals = {k:np.concatenate(v) for k,v in vals.items()}; ii=np.argsort(x)
    x=x[ii]; vals={k:v[ii] for k,v in vals.items()}
    target=0.5*sum(rho_lr); y=vals["rho"]-target; hits=np.flatnonzero(y[:-1]*y[1:]<=0)
    if len(hits):
        j=hits[np.argmin(np.abs(x[hits]))]
        xs=float(x[j]+(target-vals["rho"][j])*(x[j+1]-x[j])/(vals["rho"][j+1]-vals["rho"][j]))
    else: xs=float(x[np.argmin(np.abs(y))])
    return x,vals,xs


def residual(path: Path):
    rows=[]
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try: item=(float(r["t"]),float(r["f"]),float(r["f_normalized"]))
            except (KeyError,TypeError,ValueError): continue
            if all(math.isfinite(v) for v in item): rows.append(item)
    a=np.asarray(rows,float)
    if len(a)>2:
        keep=np.ones(len(a),dtype=bool)
        for i in range(1,len(a)-1):
            if a[i,1] < 1e-3*min(a[i-1,1],a[i+1,1]): keep[i]=False
        a=a[keep]
    return a


def temporal_l2(steady: Path, case: str, field: str):
    base=steady/"stage_1"/case
    _,a,_=dense(base/f"p3b_{case}.ini",base/"mesh.frfsm",base/f"bulksol_p3b_{case}-335.25.frfss")
    _,b,_=dense(base/f"p3b_{case}.ini",base/"mesh.frfsm",base/f"bulksol_p3b_{case}-340.25.frfss")
    return float(np.linalg.norm(b[field]-a[field])/max(np.linalg.norm(a[field]),1e-300))


def fig1(out: Path):
    fig,ax=plt.subplots(figsize=(11,4.2)); ax.axis("off")
    boxes=[(0.03,0.58,0.20,0.25,"Raw spectral collision","Q(v)"),(0.30,0.58,0.23,0.25,"Invariant defect",r"$m=c_wBQ$"),(0.60,0.58,0.30,0.25,"Weighted 5×5 solve",r"$G\lambda=m$"),(0.34,0.12,0.31,0.25,"Conservative correction",r"$Q_c=Q-wB^T\lambda$")]
    for x,y,w,h,t,s in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,lw=1.8)); ax.text(x+w/2,y+h*.67,t,ha="center",va="center",fontweight="bold",fontsize=12); ax.text(x+w/2,y+h*.30,s,ha="center",va="center",fontsize=13)
    for a,b in [((.23,.705),(.30,.705)),((.53,.705),(.60,.705)),((.75,.58),(.56,.37))]: ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.8))
    ax.text(.5,.94,r"$B=[1,v_x,v_y,v_z,\frac12|v|^2],\quad w=\max(f,0)$",ha="center",fontsize=14)
    ax.text(.5,.02,"Projection acts on Q; it is not clipping and does not imply global positivity of f.",ha="center",fontsize=10)
    fig.savefig(out/"FIG1_METHOD_CONSERVATIVE_PROJECTION.png",dpi=240,bbox_inches="tight"); fig.savefig(out/"FIG1_METHOD_CONSERVATIVE_PROJECTION.pdf",bbox_inches="tight"); plt.close(fig)


def fig3(out: Path, audits):
    x=np.arange(2); w=.34
    raw=[audits[6]["summary"]["raw_max_defect"],audits[16]["summary"]["raw_max_defect"]]
    fp=[audits[6]["summary"]["fplus_max_defect"],audits[16]["summary"]["fplus_max_defect"]]
    fig,ax=plt.subplots(figsize=(7.2,5.0),constrained_layout=True)
    ax.bar(x-w/2,raw,w,label="raw"); ax.bar(x+w/2,fp,w,label="fplus"); ax.set_yscale("log"); ax.axhline(5e-12,color="0.4",ls="--",label=r"gate $5\times10^{-12}$")
    ax.set_xticks(x,[r"$M_\omega=6$",r"$M_\omega=16$"]); ax.set_ylabel("Maximum collision-invariant defect"); ax.set_title("Discrete conservation of the collision term"); ax.grid(alpha=.2,axis="y",which="both"); ax.legend()
    fig.savefig(out/"FIG3_COLLISION_INVARIANT_DEFECT.png",dpi=240); fig.savefig(out/"FIG3_COLLISION_INVARIANT_DEFECT.pdf"); plt.close(fig)


def fig4(out: Path, report, audits):
    vals={k:v["summary"] for k,v in report["runs"].items()}; x=np.arange(2); w=.34
    raw=[vals["M6_raw"]["paper_mean_rms"],vals["M16_raw"]["paper_mean_rms"]]; fp=[vals["M6_fplus"]["paper_mean_rms"],vals["M16_fplus"]["paper_mean_rms"]]
    over=[100*(audits[6]["summary"]["median_projection_overhead_ratio"]-1),100*(audits[16]["summary"]["median_projection_overhead_ratio"]-1)]
    fig,ax=plt.subplots(1,2,figsize=(11.0,4.5),constrained_layout=True)
    ax[0].bar(x-w/2,raw,w,label="raw"); ax[0].bar(x+w/2,fp,w,label="fplus"); ax[0].set_xticks(x,[r"$M_\omega=6$",r"$M_\omega=16$"]); ax[0].set_ylabel("Mean RMS vs Alexeenko + Ohwada"); ax[0].set_title("Reference-profile accuracy"); ax[0].legend(); ax[0].grid(alpha=.2,axis="y")
    ax[1].bar(x,over,.5); ax[1].axhline(50,color="0.4",ls="--",label="50% gate"); ax[1].set_xticks(x,[r"$M_\omega=6$",r"$M_\omega=16$"]); ax[1].set_ylabel("Measured projection overhead [%]"); ax[1].set_title("Collision-kernel overhead"); ax[1].legend(); ax[1].grid(alpha=.2,axis="y")
    fig.savefig(out/"FIG4_ACCURACY_AND_OVERHEAD.png",dpi=240); fig.savefig(out/"FIG4_ACCURACY_AND_OVERHEAD.pdf"); plt.close(fig)


def fig5(out: Path, closeout: Path, steady: Path):
    paths={"M6_raw":closeout/"final_runs/run_M6_raw/kinetic_residual_p3b.csv","M6_fplus":closeout/"final_runs/run_M6_fplus/kinetic_residual_p3b.csv","M16_raw":steady/"stage_1/M16_raw/kinetic_residual_p3b.csv","M16_fplus":steady/"stage_1/M16_fplus/kinetic_residual_p3b.csv"}
    fig,ax=plt.subplots(1,2,figsize=(12.0,4.6),constrained_layout=True)
    for name,p in paths.items():
        a=residual(p); ax[0].semilogy(a[:,0],a[:,2],color=COLORS[name],ls=LINES[name],lw=1.8,label=PRETTY[name])
    ax[0].axhline(1,color="0.4",ls="--",label="steady threshold"); ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$r_n/r_1$"); ax[0].set_title("Normalized kinetic residual"); ax[0].grid(alpha=.2,which="both"); ax[0].legend(fontsize=8)
    fields=["rho","U:x","T","Q:x","p"]; labels=[r"$\rho$",r"$u_x$",r"$T$",r"$q_x$",r"$p$"]; x=np.arange(len(fields)); w=.36
    raw=[temporal_l2(steady,"M16_raw",f) for f in fields]; fp=[temporal_l2(steady,"M16_fplus",f) for f in fields]
    ax[1].bar(x-w/2,raw,w,label=r"$M_\omega=16$ raw"); ax[1].bar(x+w/2,fp,w,label=r"$M_\omega=16$ fplus"); ax[1].set_yscale("log"); ax[1].set_xticks(x,labels); ax[1].set_ylabel(r"Relative $L_2$ change, 335.25→340.25"); ax[1].set_title("Final temporal stationarity"); ax[1].grid(alpha=.2,axis="y",which="both"); ax[1].legend(fontsize=8)
    fig.suptitle("Steady-state evidence — helium normal shock, Mach 1.59")
    fig.savefig(out/"FIG5_STEADY_STATE_EVIDENCE.png",dpi=240); fig.savefig(out/"FIG5_STEADY_STATE_EVIDENCE.pdf"); plt.close(fig)


def fig6(out: Path, steady: Path):
    dat={}
    for name in ("M16_raw","M16_fplus"):
        base=steady/"stage_1"/name; dat[name]=dense(base/f"p3b_{name}.ini",base/"mesh.frfsm",base/f"bulksol_p3b_{name}-340.25.frfss")
    xs=dat["M16_raw"][2]; fig,ax=plt.subplots(1,2,figsize=(11.0,4.5),constrained_layout=True)
    for name in ("M16_raw","M16_fplus"):
        x,v,_=dat[name]; ax[0].plot(x-xs,v["Q:x"],color=COLORS[name],ls=LINES[name],lw=2,label=PRETTY[name]); ax[1].plot(x-xs,v["U:y"],color=COLORS[name],ls=LINES[name],lw=2,label=PRETTY[name])
    ax[0].set_xlabel(r"$(x-x_{s,raw})/H_0$"); ax[0].set_ylabel(r"$q_x$ [W m$^{-2}$]"); ax[0].set_title("Heat flux"); ax[0].grid(alpha=.2)
    ax[1].set_xlabel(r"$(x-x_{s,raw})/H_0$"); ax[1].set_ylabel(r"$u_y$ [m s$^{-1}$]"); ax[1].set_title("Spurious transverse velocity"); ax[1].grid(alpha=.2); ax[1].legend(fontsize=8)
    fig.suptitle(r"Mach 1.59, $M_\omega=16$: projection-sensitive quantities")
    fig.savefig(out/"FIG6_MW16_SENSITIVE_QX_UY.png",dpi=240); fig.savefig(out/"FIG6_MW16_SENSITIVE_QX_UY.pdf"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--closeout",type=Path,required=True); ap.add_argument("--steady",type=Path,required=True); ap.add_argument("--output-dir",type=Path,default=Path.cwd()); args=ap.parse_args()
    closeout=args.closeout.resolve(); steady=args.steady.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    report=json.loads((out/"novelty_report.json").read_text()); audits={6:json.loads((closeout/"audit/M6.json").read_text()),16:json.loads((closeout/"audit/M16.json").read_text())}
    fig1(out); fig3(out,audits); fig4(out,report,audits); fig5(out,closeout,steady); fig6(out,steady)
    bundle=out/"DGFS_PAPER_STAGE2_3.zip"
    if bundle.exists(): bundle.unlink()
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and (p.name.startswith("FIG") or p.name in {"novelty_report.json","novelty_metrics.csv","SUMMARY.md","CLOSEOUT_STATUS.json","FINAL_STEADY_GATE_SUMMARY.md","FINAL_STEADY_RESIDUALS.json","CLAIM_GATE_PASS","CLAIM_GATE_FAIL"}): z.write(p,p.name)
    print(f"BUNDLE={bundle}")
    for p in sorted(out.glob("FIG*")): print(p.name)

if __name__=="__main__": main()
