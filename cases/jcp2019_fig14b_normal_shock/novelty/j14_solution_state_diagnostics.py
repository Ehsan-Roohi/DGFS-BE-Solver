#!/usr/bin/env python3
"""State-level diagnostics for short M6 projection comparison runs.

Pure post-processing: reads existing DGFS distribution snapshots and reports
moments of f itself (not Q), including negativity, heat flux, deviatoric stress,
fourth-order moment, symmetry diagnostics, velocity-tail content, and global
mass/momentum/energy integrals over the 1-D DG mesh.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np

R0 = 8.3144598
NA = 6.0221409e23
TINY = np.finfo(float).tiny


def cfgread(path: Path):
    c = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    c.optionxform = str
    c.read(path)
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
    cv = np.vstack((X.ravel(),Y.ravel(),Z.ravel()))
    return cv, cw, L


def load_mesh(path: Path):
    with h5py.File(path,"r") as h:
        m = np.asarray(h["spt_line_p0"],float)
    left = m[:,:,0].min(axis=0)
    right = m[:,:,0].max(axis=0)
    x = np.vstack((left,0.5*(left+right),right))
    # 3-point GLL weights mapped from [-1,1] to each element.
    wg = np.array([1/3,4/3,1/3],float)[:,None]*(right-left)[None,:]/2
    return x,wg


def time_from_name(path: Path):
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.frfss$",path.name)
    return float(m.group(1)) if m else float("nan")


def point_metrics(f,cv,cw):
    rho = cw*float(np.sum(f))
    mom = cw*(cv@f)
    u = mom/max(rho,TINY)
    c = cv-u[:,None]
    c2 = np.sum(c*c,axis=0)
    T = (2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    sigma = math.sqrt(max(0.5*T,TINY))  # exp(-c^2/T): component variance=T/2
    neg = np.maximum(-f,0.0); pos=np.maximum(f,0.0)
    tail = np.sqrt(c2)>3*sigma
    pdev_xx = cw*float(np.dot(c[0]*c[0]-c2/3,f))
    pxz = cw*float(np.dot(c[0]*c[2],f))
    qx = 0.5*cw*float(np.dot(c2*c[0],f))
    qy = 0.5*cw*float(np.dot(c2*c[1],f))
    qz = 0.5*cw*float(np.dot(c2*c[2],f))
    c4 = cw*float(np.dot(c2*c2,f))
    return {
        "rho":rho,"ux":float(u[0]),"uy":float(u[1]),"uz":float(u[2]),"T":T,
        "Pdev_xx":pdev_xx,"Pxz":pxz,"qx":qx,"qy":qy,"qz":qz,"c4":c4,
        "min_f":float(np.min(f)),
        "negative_mass_fraction":cw*float(np.sum(neg))/max(cw*float(np.sum(pos)),TINY),
        "negative_node_fraction":float(np.mean(f<0)),
        "tail_abs_mass_fraction":cw*float(np.sum(np.abs(f[tail])))/max(cw*float(np.sum(np.abs(f))),TINY),
        "max_abs_f_tail":float(np.max(np.abs(f[tail])) if np.any(tail) else 0.0),
        "mass":rho,"momx":float(mom[0]),"momy":float(mom[1]),"momz":float(mom[2]),
        "energy":0.5*cw*float(np.dot(np.sum(cv*cv,axis=0),f)),
    }


def analyze_snapshot(path,mesh,cv,cw,label,t0):
    with h5py.File(path,"r") as h:
        f=np.asarray(h["soln_line_p0"],float)
    if f.shape[1] != cv.shape[1]:
        raise ValueError(f"{path}: velocity size mismatch {f.shape[1]} vs {cv.shape[1]}")
    x,wg=mesh
    rows=[]
    totals={k:0.0 for k in ("mass","momx","momy","momz","energy")}
    for e in range(f.shape[2]):
        for u in range(f.shape[0]):
            q=point_metrics(f[u,:,e],cv,cw)
            for k in totals: totals[k]+=wg[u,e]*q[k]
            rows.append({"label":label,"time":time_from_name(path),"elapsed":time_from_name(path)-t0,
                         "element":e,"solution_point":u,"x":float(x[u,e]),**q})
    g={"label":label,"time":time_from_name(path),"elapsed":time_from_name(path)-t0,
       **{f"global_{k}":float(v) for k,v in totals.items()},
       "min_f":min(r["min_f"] for r in rows),
       "max_negative_mass_fraction":max(r["negative_mass_fraction"] for r in rows),
       "max_abs_uy":max(abs(r["uy"]) for r in rows),"max_abs_uz":max(abs(r["uz"]) for r in rows),
       "max_abs_qz":max(abs(r["qz"]) for r in rows),"max_abs_Pxz":max(abs(r["Pxz"]) for r in rows)}
    return rows,g


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True); ap.add_argument("--mesh",type=Path,required=True)
    ap.add_argument("--label",required=True); ap.add_argument("--t0",type=float,required=True)
    ap.add_argument("--snapshots",type=Path,nargs="+",required=True); ap.add_argument("--output-prefix",type=Path,required=True)
    a=ap.parse_args(); cfg=cfgread(a.config); cv,cw,L=velocity_grid(cfg); mesh=load_mesh(a.mesh)
    allrows=[]; globals_=[]
    for p in sorted(a.snapshots,key=time_from_name):
        r,g=analyze_snapshot(p,mesh,cv,cw,a.label,a.t0); allrows+=r; globals_.append(g)
    pref=a.output_prefix
    if allrows:
        with pref.with_suffix(".points.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    if globals_:
        with pref.with_suffix(".global.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(globals_[0])); w.writeheader(); w.writerows(globals_)
    out={"schema_version":1,"label":a.label,"tail_definition":"|c| > 3 sigma, sigma=sqrt(T/2)",
         "velocity_box_halfwidth":L,"velocity_weight":cw,"snapshots":[str(p) for p in a.snapshots],
         "global_series":globals_}
    pref.with_suffix(".json").write_text(json.dumps(out,indent=2)+"\n")
    print(f"DGFS_STATE_DIAGNOSTICS_COMPLETE label={a.label} snapshots={len(a.snapshots)}")

if __name__=="__main__": main()
