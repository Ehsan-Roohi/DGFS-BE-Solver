#!/usr/bin/env python3
"""Shock-aligned steady-state comparison for the J14 normal shock.

Pure CPU post-processing. Reads existing DGFS distribution snapshots for
M6 raw / Euclidean / f+-weighted and M16 raw / f+-weighted. Each case is
centered independently at the density midpoint before profile comparisons,
so translation of the shock is not misreported as a kinetic-profile error.
No solver is run and no bar charts are produced.
"""
from __future__ import annotations

import argparse, configparser, csv, json, math, zipfile
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R0 = 8.3144598
TINY = np.finfo(float).tiny
ALL = ("M6_raw","M6_euclidean","M6_fplus","M16_raw","M16_fplus")
M6 = ("M6_raw","M6_euclidean","M6_fplus")
FIELDS = ("rho","ux","T","qx","Pdev_xx","c4","uz","qz","Pxz")
LABEL = {
    "M6_raw": r"$M_\omega=6$ raw",
    "M6_euclidean": r"$M_\omega=6$ Euclidean",
    "M6_fplus": r"$M_\omega=6$ $f^+$-weighted",
    "M16_raw": r"$M_\omega=16$ raw",
    "M16_fplus": r"$M_\omega=16$ $f^+$-weighted",
}
LS={"M6_raw":"-","M6_euclidean":"--","M6_fplus":"-.","M16_raw":":","M16_fplus":":"}


def cfgread(path):
    c=configparser.ConfigParser(inline_comment_prefixes=(";","#")); c.optionxform=str
    with Path(path).open() as f: c.read_file(f)
    return c


def velocity_grid(cfg):
    T0=cfg.getfloat("non-dim","T0"); mm=cfg.getfloat("non-dim","molarMass0")
    u0=math.sqrt(2*R0/mm*T0); Nv=cfg.getint("constants","Nv")
    cmax=cfg.getfloat("velocity-mesh","cmax")/u0
    Tmax=cfg.getfloat("velocity-mesh","Tmax")/T0
    dev=cfg.getfloat("velocity-mesh","dev")
    L=cmax+dev*math.sqrt(Tmax); cw=(2*L/Nv)**3
    c0=np.linspace(-L+L/Nv,L-L/Nv,Nv)
    X,Y,Z=np.meshgrid(c0,c0,c0,indexing="ij")
    return np.vstack((X.ravel(),Y.ravel(),Z.ravel())),cw,L


def load_mesh(path):
    with h5py.File(path,"r") as h: m=np.asarray(h["spt_line_p0"],float)
    left=m[:,:,0].min(axis=0); right=m[:,:,0].max(axis=0)
    return np.vstack((left,.5*(left+right),right))


def point_metrics(f,cv,cw):
    rho=cw*float(np.sum(f)); mom=cw*(cv@f); u=mom/max(rho,TINY)
    c=cv-u[:,None]; c2=np.sum(c*c,axis=0)
    T=(2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    sigma=math.sqrt(max(.5*T,TINY)); neg=np.maximum(-f,0); pos=np.maximum(f,0)
    tail=np.sqrt(c2)>3*sigma
    return {
        "rho":rho,"ux":float(u[0]),"uy":float(u[1]),"uz":float(u[2]),"T":T,
        "qx":.5*cw*float(np.dot(c2*c[0],f)),
        "qy":.5*cw*float(np.dot(c2*c[1],f)),
        "qz":.5*cw*float(np.dot(c2*c[2],f)),
        "Pdev_xx":cw*float(np.dot(c[0]*c[0]-c2/3,f)),
        "Pxz":cw*float(np.dot(c[0]*c[2],f)),
        "c4":cw*float(np.dot(c2*c2,f)),
        "min_f":float(np.min(f)),
        "negative_mass_fraction":cw*float(np.sum(neg))/max(cw*float(np.sum(pos)),TINY),
        "negative_node_fraction":float(np.mean(f<0)),
        "tail_abs_mass_fraction":cw*float(np.sum(np.abs(f[tail])))/max(cw*float(np.sum(np.abs(f))),TINY),
        "max_abs_f_tail":float(np.max(np.abs(f[tail])) if np.any(tail) else 0),
    }


def load_case(label,cfg_path,mesh_path,snapshot):
    cfg=cfgread(cfg_path); cv,cw,L=velocity_grid(cfg); x=load_mesh(mesh_path)
    with h5py.File(snapshot,"r") as h: f=np.asarray(h["soln_line_p0"],float)
    if f.shape[1]!=cv.shape[1]: raise ValueError(f"{label}: velocity size mismatch")
    rows=[]
    for e in range(f.shape[2]):
        for u in range(f.shape[0]):
            rows.append({"case":label,"element":e,"solution_point":u,"x":float(x[u,e]),**point_metrics(f[u,:,e],cv,cw)})
    return {"rows":rows,"L":L,"cw":cw,"snapshot":str(snapshot)}


def dense(case,field,n=121):
    by={}
    for r in case["rows"]: by.setdefault(r["element"],[]).append(r)
    rr=np.linspace(-1,1,n); B=np.vstack((.5*rr*(rr-1),1-rr*rr,.5*rr*(rr+1))).T
    xs=[]; ys=[]
    for e in sorted(by):
        q=sorted(by[e],key=lambda z:z["solution_point"])
        xv=np.array([r["x"] for r in q]); yv=np.array([r[field] for r in q])
        xs.append(.5*((1-rr)*xv[0]+(1+rr)*xv[2])); ys.append(B@yv)
    x=np.concatenate(xs); y=np.concatenate(ys); ii=np.argsort(x)
    return x[ii],y[ii]


def shock_center(case):
    x,rho=dense(case,"rho",181)
    n=max(12,len(x)//20)
    rl=float(np.mean(rho[:n])); rr=float(np.mean(rho[-n:])); target=.5*(rl+rr)
    y=rho-target; hit=np.flatnonzero(y[:-1]*y[1:]<=0)
    if len(hit):
        j=hit[np.argmin(np.abs(x[hit]))]; dr=rho[j+1]-rho[j]
        return float(x[j] if abs(dr)<TINY else x[j]+(target-rho[j])*(x[j+1]-x[j])/dr)
    return float(x[np.argmin(np.abs(y))])


def aligned_profile(case,field):
    x,y=dense(case,field); return x-case["xs"],y


def aligned_rel_l2(a,b,field):
    xa,ya=aligned_profile(a,field); xb,yb=aligned_profile(b,field)
    lo=max(xa.min(),xb.min()); hi=min(xa.max(),xb.max())
    grid=np.linspace(lo,hi,1201)
    ai=np.interp(grid,xa,ya); bi=np.interp(grid,xb,yb)
    return float(np.linalg.norm(ai-bi)/max(np.linalg.norm(bi),TINY))


def plot_group(out,cases,specs,filename,title,show):
    fig,axs=plt.subplots(1,len(specs),figsize=(4.45*len(specs),4.3),constrained_layout=True)
    if len(specs)==1: axs=[axs]
    for ax,(field,ylabel) in zip(axs,specs):
        for name in show:
            x,y=aligned_profile(cases[name],field)
            ax.plot(x,y,ls=LS[name],lw=1.8,label=LABEL[name])
        ax.set_xlabel(r"shock-aligned $x-x_s$"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=min(4,len(show)),frameon=False,bbox_to_anchor=(.5,1.06))
    fig.suptitle(title,y=1.13)
    for ext in ("png","pdf"):
        fig.savefig(out/f"{filename}.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def plot_negativity(out,cases):
    fig,axs=plt.subplots(1,2,figsize=(9.4,4.2),constrained_layout=True)
    for name in M6:
        r=sorted(cases[name]["rows"],key=lambda q:q["x"])
        x=np.array([q["x"]-cases[name]["xs"] for q in r])
        mn=np.array([q["min_f"] for q in r]); nm=np.array([q["negative_mass_fraction"] for q in r])
        axs[0].semilogy(x,np.maximum(-mn,1e-300),ls=LS[name],marker="o",ms=3,label=LABEL[name])
        axs[1].semilogy(x,np.maximum(nm,1e-300),ls=LS[name],marker="o",ms=3,label=LABEL[name])
    axs[0].set_ylabel(r"$-\min_{\bf v}f$"); axs[1].set_ylabel("negative-mass fraction")
    for ax in axs: ax.set_xlabel(r"shock-aligned $x-x_s$"); ax.grid(alpha=.2,which="both")
    h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.06))
    fig.suptitle("Steady velocity-space negativity",y=1.13)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG_STEADY_NEGATIVITY_ALIGNED.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    for n in ALL:
        ap.add_argument(f"--{n}-config",dest=f"{n}_config",type=Path,required=True)
        ap.add_argument(f"--{n}-mesh",dest=f"{n}_mesh",type=Path,required=True)
        ap.add_argument(f"--{n}-snapshot",dest=f"{n}_snapshot",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path(".")); a=ap.parse_args(); out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    cases={n:load_case(n,getattr(a,f"{n}_config"),getattr(a,f"{n}_mesh"),getattr(a,f"{n}_snapshot")) for n in ALL}
    for n in ALL: cases[n]["xs"]=shock_center(cases[n])

    summary={"schema_version":2,"comparison":"shock-aligned","shock_centers":{n:cases[n]["xs"] for n in ALL},"snapshots":{n:cases[n]["snapshot"] for n in ALL},"aligned_relative_l2_vs_M6_raw":{},"aligned_relative_l2_vs_M16_raw":{},"negativity":{}}
    for n in ("M6_euclidean","M6_fplus"):
        summary["aligned_relative_l2_vs_M6_raw"][n]={f:aligned_rel_l2(cases[n],cases["M6_raw"],f) for f in FIELDS}
    for n in ("M6_raw","M6_euclidean","M6_fplus","M16_fplus"):
        summary["aligned_relative_l2_vs_M16_raw"][n]={f:aligned_rel_l2(cases[n],cases["M16_raw"],f) for f in FIELDS}
    for n in ALL:
        r=cases[n]["rows"]
        summary["negativity"][n]={
            "min_f":min(q["min_f"] for q in r),
            "max_negative_mass_fraction":max(q["negative_mass_fraction"] for q in r),
            "max_abs_uz":max(abs(q["uz"]) for q in r),
            "max_abs_qz":max(abs(q["qz"]) for q in r),
            "max_abs_Pxz":max(abs(q["Pxz"]) for q in r),
        }
    (out/"STEADY_PROJECTION_ALIGNED_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

    rows=[]
    for n in ALL:
        for q in cases[n]["rows"]: rows.append({**q,"x_aligned":q["x"]-cases[n]["xs"]})
    with (out/"STEADY_PROJECTION_ALIGNED_POINTS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    md=["# Shock-aligned steady projection comparison","","> Profiles are centered independently at the density midpoint before L2 comparison; M16 raw is a higher-angular-order numerical reference, not exact truth.","","## Shock centers","","| case | x_s |","|---|---:|"]
    for n in ALL: md.append(f"| {n} | {cases[n]['xs']:.8e} |")
    md += ["","## Aligned relative L2 versus M6 raw","","| case | rho | ux | T | qx | Pdev_xx | c4 | uz | qz | Pxz |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in ("M6_euclidean","M6_fplus"):
        s=summary["aligned_relative_l2_vs_M6_raw"][n]; md.append("| "+n+" | "+" | ".join(f"{s[f]:.4e}" for f in FIELDS)+" |")
    md += ["","## Aligned relative L2 versus M16 raw","","| case | rho | ux | T | qx | Pdev_xx | c4 | uz | qz | Pxz |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in ("M6_raw","M6_euclidean","M6_fplus","M16_fplus"):
        s=summary["aligned_relative_l2_vs_M16_raw"][n]; md.append("| "+n+" | "+" | ".join(f"{s[f]:.4e}" for f in FIELDS)+" |")
    md += ["","## Negativity and transverse symmetry","","| case | min(f) | max negative mass | max |uz| | max |qz| | max |Pxz| |","|---|---:|---:|---:|---:|---:|"]
    for n in ALL:
        s=summary["negativity"][n]; md.append(f"| {n} | {s['min_f']:.4e} | {s['max_negative_mass_fraction']:.4e} | {s['max_abs_uz']:.4e} | {s['max_abs_qz']:.4e} | {s['max_abs_Pxz']:.4e} |")
    (out/"STEADY_PROJECTION_ALIGNED_SUMMARY.md").write_text("\n".join(md)+"\n")

    show=("M6_raw","M6_euclidean","M6_fplus","M16_raw")
    plot_group(out,cases,(("rho",r"$\rho$"),("ux",r"$u_x$"),("T",r"$T$")),"FIG_STEADY_HYDRO_ALIGNED","Steady hydrodynamic shock profiles",show)
    plot_group(out,cases,(("qx",r"$q_x$"),("Pdev_xx",r"$P_{xx}^{dev}$"),("c4",r"$\langle c^4f\rangle$")),"FIG_STEADY_KINETIC_ALIGNED","Steady non-conserved kinetic moments",show)
    plot_group(out,cases,(("uz",r"$u_z$"),("qz",r"$q_z$"),("Pxz",r"$P_{xz}$")),"FIG_STEADY_SYMMETRY_ALIGNED","Steady transverse-symmetry diagnostics",M6)
    plot_negativity(out,cases)

    zp=out/"DGFS_STEADY_PROJECTION_ALIGNED.zip"; zp.unlink(missing_ok=True)
    names=["STEADY_PROJECTION_ALIGNED_SUMMARY.json","STEADY_PROJECTION_ALIGNED_SUMMARY.md","STEADY_PROJECTION_ALIGNED_POINTS.csv",
           "FIG_STEADY_HYDRO_ALIGNED.png","FIG_STEADY_HYDRO_ALIGNED.pdf","FIG_STEADY_KINETIC_ALIGNED.png","FIG_STEADY_KINETIC_ALIGNED.pdf",
           "FIG_STEADY_SYMMETRY_ALIGNED.png","FIG_STEADY_SYMMETRY_ALIGNED.pdf","FIG_STEADY_NEGATIVITY_ALIGNED.png","FIG_STEADY_NEGATIVITY_ALIGNED.pdf"]
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for n in names:
            p=out/n
            if p.exists(): z.write(p,p.name)
    print((out/"STEADY_PROJECTION_ALIGNED_SUMMARY.md").read_text())
    print("BUNDLE="+str(zp))

if __name__=="__main__": main()
