#!/usr/bin/env python3
"""Steady-state comparison of raw, Euclidean and distribution-weighted projections.

Pure post-processing.  Reads existing DGFS distribution snapshots and compares
state-level kinetic moments on the common 1-D DG mesh.  No solver is run.
"""
from __future__ import annotations

import argparse
import configparser
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

R0 = 8.3144598
TINY = np.finfo(float).tiny
FIELDS = ("rho", "ux", "T", "qx", "Pdev_xx", "c4", "uz", "qz", "Pxz")
M6 = ("M6_raw", "M6_euclidean", "M6_fplus")
ALL = ("M6_raw", "M6_euclidean", "M6_fplus", "M16_raw", "M16_fplus")
LABEL = {
    "M6_raw": r"$M_\omega=6$ raw",
    "M6_euclidean": r"$M_\omega=6$ Euclidean",
    "M6_fplus": r"$M_\omega=6$ $f^+$-weighted",
    "M16_raw": r"$M_\omega=16$ raw",
    "M16_fplus": r"$M_\omega=16$ $f^+$-weighted",
}
LS = {"M6_raw":"-", "M6_euclidean":"--", "M6_fplus":"-.", "M16_raw":":", "M16_fplus":":"}


def cfgread(path: Path):
    c = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    c.optionxform = str
    with path.open() as f:
        c.read_file(f)
    return c


def velocity_grid(cfg):
    T0 = cfg.getfloat("non-dim", "T0")
    mm = cfg.getfloat("non-dim", "molarMass0")
    u0 = math.sqrt(2*R0/mm*T0)
    Nv = cfg.getint("constants", "Nv")
    cmax = cfg.getfloat("velocity-mesh", "cmax")/u0
    Tmax = cfg.getfloat("velocity-mesh", "Tmax")/T0
    dev = cfg.getfloat("velocity-mesh", "dev")
    L = cmax + dev*math.sqrt(Tmax)
    cw = (2*L/Nv)**3
    c0 = np.linspace(-L + L/Nv, L - L/Nv, Nv)
    X,Y,Z = np.meshgrid(c0,c0,c0,indexing="ij")
    return np.vstack((X.ravel(),Y.ravel(),Z.ravel())), cw, L


def load_mesh(path: Path):
    with h5py.File(path,"r") as h:
        m = np.asarray(h["spt_line_p0"],float)
    left = m[:,:,0].min(axis=0)
    right = m[:,:,0].max(axis=0)
    return np.vstack((left,0.5*(left+right),right))


def point_metrics(f,cv,cw):
    rho = cw*float(np.sum(f))
    mom = cw*(cv@f)
    u = mom/max(rho,TINY)
    c = cv-u[:,None]
    c2 = np.sum(c*c,axis=0)
    T = (2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    sigma = math.sqrt(max(0.5*T,TINY))
    neg = np.maximum(-f,0.0); pos = np.maximum(f,0.0)
    tail = np.sqrt(c2) > 3*sigma
    return {
        "rho":rho, "ux":float(u[0]), "uy":float(u[1]), "uz":float(u[2]), "T":T,
        "qx":0.5*cw*float(np.dot(c2*c[0],f)),
        "qy":0.5*cw*float(np.dot(c2*c[1],f)),
        "qz":0.5*cw*float(np.dot(c2*c[2],f)),
        "Pdev_xx":cw*float(np.dot(c[0]*c[0]-c2/3,f)),
        "Pxz":cw*float(np.dot(c[0]*c[2],f)),
        "c4":cw*float(np.dot(c2*c2,f)),
        "min_f":float(np.min(f)),
        "negative_mass_fraction":cw*float(np.sum(neg))/max(cw*float(np.sum(pos)),TINY),
        "negative_node_fraction":float(np.mean(f<0)),
        "tail_abs_mass_fraction":cw*float(np.sum(np.abs(f[tail])))/max(cw*float(np.sum(np.abs(f))),TINY),
        "max_abs_f_tail":float(np.max(np.abs(f[tail])) if np.any(tail) else 0.0),
    }


def load_case(label,cfg_path,mesh_path,snapshot):
    cfg=cfgread(cfg_path); cv,cw,L=velocity_grid(cfg); x=load_mesh(mesh_path)
    with h5py.File(snapshot,"r") as h: f=np.asarray(h["soln_line_p0"],float)
    if f.shape[1] != cv.shape[1]:
        raise ValueError(f"{label}: velocity size mismatch {f.shape[1]} != {cv.shape[1]}")
    rows=[]
    for e in range(f.shape[2]):
        for u in range(f.shape[0]):
            rows.append({"case":label,"element":e,"solution_point":u,"x":float(x[u,e]),**point_metrics(f[u,:,e],cv,cw)})
    # Preserve element/GLL ordering for apples-to-apples L2 comparisons.
    arrays={k:np.array([r[k] for r in rows],float) for k in rows[0] if k not in ("case","element","solution_point")}
    return {"rows":rows,"arrays":arrays,"L":L,"cw":cw,"snapshot":str(snapshot)}


def rel_l2(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),TINY))


def dense_segments(case,field,n=81):
    rows=case["rows"]
    by={}
    for r in rows: by.setdefault(r["element"],[]).append(r)
    out=[]
    rr=np.linspace(-1,1,n)
    B=np.vstack((0.5*rr*(rr-1),1-rr*rr,0.5*rr*(rr+1))).T
    for e in sorted(by):
        q=sorted(by[e],key=lambda z:z["solution_point"])
        xv=np.array([r["x"] for r in q]); yv=np.array([r[field] for r in q])
        x=.5*((1-rr)*xv[0]+(1+rr)*xv[2]); y=B@yv
        out.append((x,y))
    return out


def plot_profiles(out,cases):
    specs=[("qx",r"$q_x$"),("Pdev_xx",r"$P_{xx}-p$"),("c4",r"$\langle c^4 f\rangle$")]
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.3),constrained_layout=True)
    show=("M6_raw","M6_euclidean","M6_fplus","M16_raw")
    for ax,(field,ylabel) in zip(axs,specs):
        for name in show:
            first=True
            for x,y in dense_segments(cases[name],field):
                ax.plot(x,y,ls=LS[name],lw=1.8,label=LABEL[name] if first else None)
                first=False
        ax.set_xlabel("x (nondimensional)"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.5,1.08))
    fig.suptitle("Steady kinetic moments: raw vs conservative projections",y=1.14)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG_STEADY_KINETIC_MOMENTS.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def plot_symmetry(out,cases):
    specs=[("uz",r"$u_z$"),("qz",r"$q_z$"),("Pxz",r"$P_{xz}$")]
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.3),constrained_layout=True)
    for ax,(field,ylabel) in zip(axs,specs):
        for name in M6:
            first=True
            for x,y in dense_segments(cases[name],field):
                ax.plot(x,y,ls=LS[name],lw=1.8,label=LABEL[name] if first else None)
                first=False
        ax.axhline(0,lw=.8); ax.set_xlabel("x (nondimensional)"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.08))
    fig.suptitle("Steady transverse-symmetry diagnostics",y=1.14)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG_STEADY_SYMMETRY.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def plot_negativity(out,cases):
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.2),constrained_layout=True)
    for name in M6:
        a=cases[name]["arrays"]; x=a["x"]
        ii=np.argsort(x)
        axs[0].semilogy(x[ii],np.maximum(-a["min_f"][ii],1e-300),ls=LS[name],marker="o",ms=3,label=LABEL[name])
        axs[1].semilogy(x[ii],np.maximum(a["negative_mass_fraction"][ii],1e-300),ls=LS[name],marker="o",ms=3,label=LABEL[name])
    axs[0].set_ylabel(r"$-\min_{\bf v} f$"); axs[1].set_ylabel("negative-mass fraction")
    for ax in axs: ax.set_xlabel("x (nondimensional)"); ax.grid(alpha=.2,which="both")
    handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.08))
    fig.suptitle("Steady distribution negativity",y=1.14)
    for ext in ("png","pdf"): fig.savefig(out/f"FIG_STEADY_NEGATIVITY.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    for n in ALL:
        ap.add_argument(f"--{n}-config",dest=f"{n}_config",type=Path,required=True)
        ap.add_argument(f"--{n}-mesh",dest=f"{n}_mesh",type=Path,required=True)
        ap.add_argument(f"--{n}-snapshot",dest=f"{n}_snapshot",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("."))
    args=ap.parse_args(); out=args.out_dir; out.mkdir(parents=True,exist_ok=True)
    cases={}
    for n in ALL:
        cases[n]=load_case(n,getattr(args,f"{n}_config"),getattr(args,f"{n}_mesh"),getattr(args,f"{n}_snapshot"))

    # Point table.
    allrows=[]
    for n in ALL: allrows.extend(cases[n]["rows"])
    with (out/"STEADY_PROJECTION_COMPARE_POINTS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)

    summary={"schema_version":1,"snapshots":{n:cases[n]["snapshot"] for n in ALL},"relative_l2_vs_M16_raw":{},"relative_l2_vs_M6_raw":{},"negativity":{}}
    ref16=cases["M16_raw"]["arrays"]; ref6=cases["M6_raw"]["arrays"]
    for n in ("M6_raw","M6_euclidean","M6_fplus","M16_fplus"):
        summary["relative_l2_vs_M16_raw"][n]={k:rel_l2(cases[n]["arrays"][k],ref16[k]) for k in FIELDS}
    for n in ("M6_euclidean","M6_fplus"):
        summary["relative_l2_vs_M6_raw"][n]={k:rel_l2(cases[n]["arrays"][k],ref6[k]) for k in FIELDS}
    for n in ALL:
        a=cases[n]["arrays"]
        summary["negativity"][n]={
            "min_f":float(np.min(a["min_f"])),
            "max_negative_mass_fraction":float(np.max(a["negative_mass_fraction"])),
            "max_abs_uz":float(np.max(np.abs(a["uz"]))),
            "max_abs_qz":float(np.max(np.abs(a["qz"]))),
            "max_abs_Pxz":float(np.max(np.abs(a["Pxz"]))),
        }
    (out/"STEADY_PROJECTION_COMPARE_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

    md=["# Steady projection comparison","",
        "All quantities below are moments of the steady distribution f itself. M16 raw is used only as a higher-angular-order numerical reference, not as exact truth.","",
        "## Relative L2 difference versus M16 raw","",
        "| case | rho | ux | T | qx | Pdev_xx | c4 | uz | qz | Pxz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in ("M6_raw","M6_euclidean","M6_fplus","M16_fplus"):
        s=summary["relative_l2_vs_M16_raw"][n]; md.append("| "+n+" | "+" | ".join(f"{s[k]:.4e}" for k in FIELDS)+" |")
    md += ["","## Relative L2 difference versus M6 raw","",
           "| case | rho | ux | T | qx | Pdev_xx | c4 | uz | qz | Pxz |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in ("M6_euclidean","M6_fplus"):
        s=summary["relative_l2_vs_M6_raw"][n]; md.append("| "+n+" | "+" | ".join(f"{s[k]:.4e}" for k in FIELDS)+" |")
    md += ["","## Negativity and symmetry","",
           "| case | min(f) | max negative mass | max |uz| | max |qz| | max |Pxz| |",
           "|---|---:|---:|---:|---:|---:|"]
    for n in ALL:
        s=summary["negativity"][n]; md.append(f"| {n} | {s['min_f']:.4e} | {s['max_negative_mass_fraction']:.4e} | {s['max_abs_uz']:.4e} | {s['max_abs_qz']:.4e} | {s['max_abs_Pxz']:.4e} |")
    (out/"STEADY_PROJECTION_COMPARE_SUMMARY.md").write_text("\n".join(md)+"\n")

    plot_profiles(out,cases); plot_symmetry(out,cases); plot_negativity(out,cases)

    zp=out/"DGFS_STEADY_PROJECTION_COMPARE.zip"; zp.unlink(missing_ok=True)
    keep=["STEADY_PROJECTION_COMPARE_POINTS.csv","STEADY_PROJECTION_COMPARE_SUMMARY.json","STEADY_PROJECTION_COMPARE_SUMMARY.md",
          "FIG_STEADY_KINETIC_MOMENTS.png","FIG_STEADY_KINETIC_MOMENTS.pdf","FIG_STEADY_SYMMETRY.png","FIG_STEADY_SYMMETRY.pdf","FIG_STEADY_NEGATIVITY.png","FIG_STEADY_NEGATIVITY.pdf"]
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for n in keep:
            p=out/n
            if p.exists(): z.write(p,p.name)
    print((out/"STEADY_PROJECTION_COMPARE_SUMMARY.md").read_text())
    print(f"BUNDLE={zp.resolve()}")

if __name__=="__main__": main()
