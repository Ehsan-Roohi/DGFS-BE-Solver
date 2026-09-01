#!/usr/bin/env python3
"""Plot and quantify the existing M16 raw/fplus DGFS steady data.

No solver run is performed.  The script reads only the existing mesh, bulk
moment files, and residual CSVs from an M16 steady directory.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = ("M16_raw", "M16_fplus")
TIMES = (330.25, 335.25, 340.25)
FIELD_LABELS = {
    "rho": r"$\rho$ [kg m$^{-3}$]",
    "U:x": r"$u_x$ [m s$^{-1}$]",
    "U:y": r"$u_y$ [m s$^{-1}$]",
    "T": r"$T$ [K]",
    "Q:x": r"$q_x$ [W m$^{-2}$]",
    "p": r"$p$ [Pa]",
}


def ini_from_h5(value: object) -> configparser.ConfigParser:
    raw = value.decode() if isinstance(value, bytes) else str(value)
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read_string(raw)
    return cfg


def lagrange_gll2(r: np.ndarray) -> np.ndarray:
    return np.vstack((0.5*r*(r - 1.0), 1.0 - r*r, 0.5*r*(r + 1.0)))


def read_case_config(stage: Path, case: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(stage/case/f"p3b_{case}.ini")
    return cfg


def mesh_edges(mesh_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(mesh_file, "r") as h5:
        mesh = np.asarray(h5["spt_line_p0"])
    x_left = mesh[:, :, 0].min(axis=0)
    x_right = mesh[:, :, 0].max(axis=0)
    order = np.argsort(x_left)
    return x_left[order], x_right[order], order


def load_dense(stage: Path, case: str, time: float, x_left, x_right, order):
    path = stage/case/f"bulksol_p3b_{case}-{time:.2f}.frfss"
    with h5py.File(path, "r") as h5:
        moments = np.asarray(h5["moments_line_p0"], dtype=float)[:, :, order]
        stats = ini_from_h5(h5["stats"][()])
    fields = [s.strip() for s in stats["data"]["fields"].split(",")]
    fidx = {name: fields.index(name) for name in fields}

    r = np.linspace(-1.0, 1.0, 241)
    basis = lagrange_gll2(r)
    xs, vals = [], {name: [] for name in fields}
    for e in range(moments.shape[2]):
        xe = 0.5*((1.0-r)*x_left[e] + (1.0+r)*x_right[e])
        xs.append(xe)
        for name in fields:
            vals[name].append(basis.T @ moments[:, fidx[name], e])
    x = np.concatenate(xs)
    out = {name: np.concatenate(v) for name, v in vals.items()}
    ii = np.argsort(x)
    return x[ii], {k: v[ii] for k, v in out.items()}, path


def shock_center(x: np.ndarray, rho: np.ndarray, rho_left: float, rho_right: float) -> float:
    target = 0.5*(rho_left + rho_right)
    y = rho - target
    hits = np.flatnonzero(y[:-1]*y[1:] <= 0.0)
    if hits.size == 0:
        return float(x[np.argmin(np.abs(y))])
    candidates = []
    for j in hits:
        if rho[j+1] == rho[j]:
            candidates.append(float(x[j]))
        else:
            candidates.append(float(x[j] + (target-rho[j])*(x[j+1]-x[j])/(rho[j+1]-rho[j])))
    return min(candidates, key=abs)


def temporal_metrics(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    ans = {}
    for name in a:
        aa, bb = a[name], b[name]
        d = bb-aa
        rel = float(np.linalg.norm(d)/max(np.linalg.norm(aa), 1e-300))
        rng = float(max(np.ptp(aa), 1e-300))
        ans[name] = {
            "relative_l2": rel,
            "max_delta_over_range": float(np.max(np.abs(d))/rng),
            "max_abs_delta": float(np.max(np.abs(d))),
        }
    return ans


def read_residual(path: Path):
    rows=[]
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                t=float(r["t"]); raw=float(r["f"]); norm=float(r["f_normalized"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(raw) and np.isfinite(norm):
                rows.append((t,raw,norm))
    a=np.asarray(rows,float)
    keep=np.ones(a.shape[0],dtype=bool)
    # Writer/checkpoint synchronization can create isolated machine-zero rows.
    # Remove only points that are >1000x below both immediate neighbours.
    for i in range(1,a.shape[0]-1):
        if a[i,1] < 1e-3*min(a[i-1,1],a[i+1,1]):
            keep[i]=False
    return a, a[keep]


def sustained_crossing(clean: np.ndarray, threshold: float=1.0, hold: float=0.5):
    t,n=clean[:,0],clean[:,2]
    for i in np.flatnonzero(n <= threshold):
        j=np.searchsorted(t,t[i]+hold,side="left")
        if j < len(t) and np.all(n[i:j+1] <= threshold):
            return float(t[i]), float(n[i])
    return None


def write_profile_csv(path: Path, x, vals, xs, h0):
    fields=[k for k in ("rho","U:x","U:y","T","Q:x","p") if k in vals]
    with path.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["x_over_H0","x_minus_xs_over_H0","x_mm",*fields])
        for i in range(len(x)):
            w.writerow([x[i],x[i]-xs,x[i]*h0*1e3,*[vals[k][i] for k in fields]])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steady-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=None)
    args=ap.parse_args()
    steady=args.steady_dir.resolve()
    stage=steady/"stage_1"
    out=(args.output_dir or steady/"analysis_m16_existing").resolve()
    out.mkdir(parents=True,exist_ok=True)

    mesh=stage/"M16_raw"/"mesh.frfsm"
    x_left,x_right,order=mesh_edges(mesh)
    cfg=read_case_config(stage,"M16_raw")
    h0=cfg.getfloat("non-dim","H0")
    rho_lr=(cfg.getfloat("soln-bcs-left","rho"),cfg.getfloat("soln-bcs-right","rho"))
    bc={
        "rho": rho_lr,
        "U:x": (cfg.getfloat("soln-bcs-left","ux"),cfg.getfloat("soln-bcs-right","ux")),
        "T": (cfg.getfloat("soln-bcs-left","T"),cfg.getfloat("soln-bcs-right","T")),
    }

    data={}
    centers={}
    for case in CASES:
        data[case]={}
        for t in TIMES:
            x,v,p=load_dense(stage,case,t,x_left,x_right,order)
            data[case][t]=(x,v,p)
        x,v,_=data[case][340.25]
        centers[case]=shock_center(x,v["rho"],*rho_lr)
        write_profile_csv(out/f"{case}_profile_340p25.csv",x,v,centers[case],h0)

    # Figure 1: dimensional raw vs fplus at the same physical x, centered on raw shock.
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.3),constrained_layout=True)
    fields=("rho","U:x","T","Q:x","p","U:y")
    styles={"M16_raw":("-",2.1),"M16_fplus":("--",2.0)}
    for ax,name in zip(axes.flat,fields):
        for case in CASES:
            x,v,_=data[case][340.25]
            ls,lw=styles[case]
            ax.plot(x-centers["M16_raw"],v[name],ls=ls,lw=lw,label=case.replace("M16_",""))
        ax.set_xlabel(r"$(x-x_{s,raw})/H_0$")
        ax.set_ylabel(FIELD_LABELS[name])
        ax.grid(alpha=0.25)
    axes[0,0].legend()
    fig.suptitle("DGFS M=16 existing steady bulk solution, t=340.25")
    fig.savefig(out/"M16_RAW_VS_FPLUS_340p25.png",dpi=240,facecolor="white")
    fig.savefig(out/"M16_RAW_VS_FPLUS_340p25.pdf")
    plt.close(fig)

    # Figure 2: normalized shock shape, each case centered on its own density midpoint.
    fig,ax=plt.subplots(figsize=(8.4,6.2),constrained_layout=True)
    for case in CASES:
        x,v,_=data[case][340.25]
        for name,label in (("rho",r"$\rho$"),("U:x",r"$u_x$"),("T",r"$T$")):
            left,right=bc[name]
            y=(v[name]-left)/(right-left)
            ax.plot(x-centers[case],y,lw=2,label=f"{case.replace('M16_','')} {label}")
    ax.set_xlabel(r"$(x-x_s)/H_0$")
    ax.set_ylabel("Normalized property")
    ax.set_title("M=16 normalized shock structure at t=340.25")
    ax.grid(alpha=0.25); ax.legend(ncol=2)
    fig.savefig(out/"M16_NORMALIZED_SHOCK_340p25.png",dpi=240,facecolor="white")
    fig.savefig(out/"M16_NORMALIZED_SHOCK_340p25.pdf")
    plt.close(fig)

    # Figure 3: temporal stationarity for the most informative quantities.
    fig,axes=plt.subplots(2,2,figsize=(12.4,8.2),constrained_layout=True)
    for ax,name in zip(axes.flat,("rho","T","Q:x","p")):
        for case in CASES:
            for t,ls in zip(TIMES,(":","--","-")):
                x,v,_=data[case][t]
                ax.plot(x-centers[case],v[name],ls=ls,lw=1.6,
                        label=f"{case.replace('M16_','')} t={t:.2f}")
        ax.set_xlabel(r"$(x-x_s)/H_0$"); ax.set_ylabel(FIELD_LABELS[name]); ax.grid(alpha=0.25)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle("M=16 temporal stationarity from existing bulk snapshots")
    fig.savefig(out/"M16_TEMPORAL_STATIONARITY.png",dpi=240,facecolor="white")
    fig.savefig(out/"M16_TEMPORAL_STATIONARITY.pdf")
    plt.close(fig)

    # Figure 4: cleaned residual history. Raw CSV is retained untouched.
    residual_report={}
    fig,ax=plt.subplots(figsize=(8.4,5.8),constrained_layout=True)
    for case in CASES:
        raw,clean=read_residual(stage/case/"kinetic_residual_p3b.csv")
        cross=sustained_crossing(clean)
        residual_report[case]={
            "rows_total":int(len(raw)),"rows_clean":int(len(clean)),
            "removed_isolated_artifacts":int(len(raw)-len(clean)),
            "sustained_threshold_crossing":cross,
            "final_clean_row":clean[-1].tolist(),
        }
        ax.semilogy(clean[:,0],clean[:,2],lw=1.8,label=case.replace("M16_",""))
        if cross:
            ax.scatter([cross[0]],[cross[1]],s=38,zorder=4)
    ax.axhline(1.0,color="0.35",ls="--",lw=1.2,label="steady threshold")
    ax.set_xlabel("t"); ax.set_ylabel(r"$r_n/r_1$"); ax.set_title("M=16 DGFS residual history (isolated checkpoint artifacts removed)")
    ax.grid(alpha=0.25,which="both"); ax.legend()
    fig.savefig(out/"M16_RESIDUAL_HISTORY_CLEAN.png",dpi=240,facecolor="white")
    fig.savefig(out/"M16_RESIDUAL_HISTORY_CLEAN.pdf")
    plt.close(fig)

    report={
        "steady_dir":str(steady),"H0_m":h0,
        "shock_center_x_over_H0":centers,
        "shock_center_mm":{k:float(v*h0*1e3) for k,v in centers.items()},
        "shock_center_difference_over_H0":float(centers["M16_fplus"]-centers["M16_raw"]),
        "temporal_stationarity":{},"raw_vs_fplus_340p25":{},
        "residual":residual_report,
    }
    for case in CASES:
        report["temporal_stationarity"][case]={}
        for ta,tb in ((330.25,335.25),(335.25,340.25)):
            report["temporal_stationarity"][case][f"{ta:.2f}_to_{tb:.2f}"]=temporal_metrics(data[case][ta][1],data[case][tb][1])
    report["raw_vs_fplus_340p25"]=temporal_metrics(data["M16_raw"][340.25][1],data["M16_fplus"][340.25][1])
    (out/"M16_EXISTING_STEADY_ANALYSIS.json").write_text(json.dumps(report,indent=2)+"\n")

    print(f"OUTPUT_DIR={out}")
    for p in sorted(out.iterdir()):
        print(p.name)
    print(json.dumps({"shock_centers":centers,"residual":residual_report},indent=2))


if __name__ == "__main__":
    main()
