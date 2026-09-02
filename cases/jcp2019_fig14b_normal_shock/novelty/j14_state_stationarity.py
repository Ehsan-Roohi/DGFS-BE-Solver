#!/usr/bin/env python3
"""State-level stationarity gate for DGFS distribution snapshots."""
from __future__ import annotations

import argparse
import configparser
import json
import math
from pathlib import Path

import h5py
import numpy as np

R0 = 8.3144598
TINY = np.finfo(float).tiny
FIELDS = ("rho", "ux", "T", "qx", "Pdev_xx", "c4")


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
    return cv,cw


def point_metrics(f,cv,cw):
    rho = cw*float(np.sum(f))
    mom = cw*(cv@f)
    u = mom/max(rho,TINY)
    c = cv-u[:,None]
    c2 = np.sum(c*c,axis=0)
    T = (2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    qx = 0.5*cw*float(np.dot(c2*c[0],f))
    pdev = cw*float(np.dot(c[0]*c[0]-c2/3,f))
    c4 = cw*float(np.dot(c2*c2,f))
    neg = np.maximum(-f,0.0); pos = np.maximum(f,0.0)
    return {
        "rho":rho,"ux":float(u[0]),"T":T,"qx":qx,"Pdev_xx":pdev,"c4":c4,
        "min_f":float(np.min(f)),
        "negative_mass_fraction":cw*float(np.sum(neg))/max(cw*float(np.sum(pos)),TINY),
    }


def snapshot_metrics(path,cv,cw):
    with h5py.File(path,"r") as h:
        f=np.asarray(h["soln_line_p0"],float)
    rows=[]
    for e in range(f.shape[2]):
        for u in range(f.shape[0]):
            rows.append(point_metrics(f[u,:,e],cv,cw))
    return rows


def rel_l2(a,b):
    aa=np.asarray(a,float); bb=np.asarray(b,float)
    return float(np.linalg.norm(bb-aa)/max(np.linalg.norm(aa),TINY))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--old",type=Path,required=True)
    ap.add_argument("--new",type=Path,required=True)
    ap.add_argument("--tolerance",type=float,default=5e-4)
    ap.add_argument("--output-json",type=Path,required=True)
    a=ap.parse_args()
    cv,cw=velocity_grid(cfgread(a.config))
    old=snapshot_metrics(a.old,cv,cw); new=snapshot_metrics(a.new,cv,cw)
    if len(old)!=len(new): raise SystemExit("snapshot point counts differ")
    drifts={k:rel_l2([r[k] for r in old],[r[k] for r in new]) for k in FIELDS}
    worst=max(drifts.values())
    out={
        "schema_version":1,
        "old_snapshot":str(a.old),"new_snapshot":str(a.new),
        "tolerance":a.tolerance,"relative_l2":drifts,"max_relative_l2":worst,
        "final_min_f":min(r["min_f"] for r in new),
        "final_max_negative_mass_fraction":max(r["negative_mass_fraction"] for r in new),
        "pass":bool(worst<=a.tolerance),
    }
    a.output_json.write_text(json.dumps(out,indent=2)+"\n")
    print("DGFS_STATE_STATIONARITY="+json.dumps(out,sort_keys=True))
    print("DGFS_STATE_STATIONARITY_GATE="+("PASS" if out["pass"] else "CONTINUE"))


if __name__=="__main__": main()
