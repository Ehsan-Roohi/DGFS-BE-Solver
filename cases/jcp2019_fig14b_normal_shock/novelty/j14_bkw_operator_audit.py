#!/usr/bin/env python3
"""Exact-BKW operator audit for Maxwell molecules.

Constructs the positive Bobylev-Krook-Wu distribution on the solver velocity grid,
evaluates the same fast-spectral collision operator, then compares raw, Euclidean,
f-weighted (f+=f here), and local-Maxwellian-weighted conservative corrections
against the analytical BKW time derivative.

The DGFS physical prefactor need not match the standard BKW time normalization.
Therefore one scalar alpha is fitted ONCE from the raw numerical Q to Q_exact and
then held fixed for every projection mode.  Reported operator errors test shape and
high-order-moment fidelity, not absolute collision-frequency calibration.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

TINY=np.finfo(float).tiny
INV=("mass","momentum_x","momentum_y","momentum_z","energy")


def bkw_K(t):
    return 1.0 - 0.4*math.exp(-t/6.0)


def bkw_f(cv,t):
    K=bkw_K(t)
    v2=np.sum(cv*cv,axis=0)
    pref=1.0/(2.0*(2.0*math.pi*K)**1.5)
    poly=(5.0*K-3.0)/K + (1.0-K)*v2/(K*K)
    return pref*np.exp(-v2/(2.0*K))*poly


def exact_q(cv,t,dt):
    return (bkw_f(cv,t+dt)-bkw_f(cv,t-dt))/(2.0*dt)


def basis(cv):
    return np.vstack((np.ones(cv.shape[1]),cv,0.5*np.sum(cv*cv,axis=0)))


def invariant_defect(q,B,cw):
    signed=cw*(B@q); scale=cw*(np.abs(B)@np.abs(q))
    return float(np.max(np.abs(signed)/np.maximum(scale,TINY)))


def project(q,B,w):
    G=(B*w[None,:])@B.T; rhs=B@q
    lam=np.linalg.solve(G,rhs)
    delta=-w*(B.T@lam)
    d=np.sqrt(np.maximum(np.diag(G),TINY)); Gs=G/np.outer(d,d)
    return q+delta,delta,float(np.linalg.cond(Gs))


def local_maxwellian(f,cv,cw):
    rho=cw*float(np.sum(f)); u=cw*(cv@f)/rho
    c=cv-u[:,None]; c2=np.sum(c*c,axis=0)
    T=(2.0/3.0)*cw*float(np.dot(c2,f))/rho
    M=rho/(math.pi*T)**1.5*np.exp(-c2/T)
    return M,c,T


def rel_l2(a,b,mask=None):
    if mask is not None: a,b=a[mask],b[mask]
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),TINY))


def moment_rates(q,cv,cw):
    v2=np.sum(cv*cv,axis=0)
    return {
      "vx4":cw*float(np.dot(cv[0]**4,q)),
      "radial_c4":cw*float(np.dot(v2*v2,q)),
      "vx6":cw*float(np.dot(cv[0]**6,q)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--M",type=int,required=True)
    ap.add_argument("--Nrho",type=int,default=32)
    ap.add_argument("--time",type=float,default=0.0)
    ap.add_argument("--fd-dt",type=float,default=1e-6)
    ap.add_argument("--output-json",type=Path,required=True)
    a=ap.parse_args()

    import mpi4py.rc; mpi4py.rc.initialize=False
    from mpi4py import MPI
    if not MPI.Is_initialized(): MPI.Init()
    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.scattering import DGFSScatteringModel
    from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
    from frfs.util import subclass_where
    import pycuda.driver as cuda

    cfg=Inifile.load(str(a.config))
    cfg.set("spherical-design-rule","M",str(a.M))
    cfg.set("constants","Nrho",str(a.Nrho))
    # VHS omega=1 -> gamma=2(1-omega)=0: Maxwell-molecule constant-speed kernel.
    cfg.set("scattering-model","omega","1.0")

    backend=get_backend("cuda",cfg); vm=DGFSVelocityMesh(backend,cfg)
    cls=subclass_where(DGFSScatteringModel,scattering_model=cfg.get("scattering-model","type"))
    scat=cls(backend,cfg,vm); cuda.Context.synchronize()

    cv=np.asarray(vm.cv(),float); cw=float(vm.cw()); B=basis(cv)
    f=bkw_f(cv,a.time); qan=exact_q(cv,a.time,a.fd_dt)
    M,c,T=local_maxwellian(f,cv,cw)
    fp=np.maximum(f,0.0)

    shape=(1,vm.vsize(),1); zeros=np.zeros(shape,dtype=backend.fpdtype)
    df=backend.matrix(shape,zeros,tags={"align"}); dq=backend.matrix(shape,zeros,tags={"align"})
    df.set(f.astype(backend.fpdtype).reshape(shape)); dq.set(zeros)
    scat.fs(df,dq,0,0); cuda.Context.synchronize()
    qraw=np.asarray(dq.get()[0,:,0],float).copy()

    # One collision-frequency calibration from RAW only; reused for all modes.
    alpha=float(np.dot(qraw,qan)/max(np.dot(qraw,qraw),TINY))
    core=f > 1e-10*float(np.max(f))

    modes={"raw":(qraw,np.zeros_like(qraw),float("nan"))}
    for name,w in (("euclidean",np.ones_like(f)),("fplus",fp),("maxwellian",M)):
        modes[name]=project(qraw,B,w)

    exact_m=moment_rates(qan,cv,cw)
    results={}
    for name,(q,delta,cond) in modes.items():
        qs=alpha*q
        mr=moment_rates(qs,cv,cw)
        results[name]={
          "invariant_defect":invariant_defect(q,B,cw),
          "relative_operator_l2_full":rel_l2(qs,qan),
          "relative_operator_l2_core":rel_l2(qs,qan,core),
          "relative_correction_l2":float(np.linalg.norm(delta)/max(np.linalg.norm(qraw),TINY)),
          "scaled_gram_condition":cond,
          "moment_rates":mr,
          "moment_rate_relative_error":{k:float(abs(mr[k]-exact_m[k])/max(abs(exact_m[k]),TINY)) for k in exact_m},
        }

    out={
      "schema_version":1,
      "purpose":"exact BKW operator-shape audit: raw/euclidean/fplus/maxwellian",
      "BKW":{"K":bkw_K(a.time),"time":a.time,"fd_dt":a.fd_dt,
             "formula":"K=1-0.4 exp(-t/6); positive BKW benchmark"},
      "grid":{"Nv":int(vm.Nv()),"Nrho":int(vm.Nrho()),"M_omega":int(vm.M()),"L":float(vm.L()),"cw":cw},
      "collision":{"type":cfg.get("scattering-model","type"),"omega":1.0,"gamma":0.0,"alpha_raw_to_exact":alpha},
      "discrete_BKW":{"mass":cw*float(np.sum(f)),"min_f":float(np.min(f)),"temperature":float(T),"core_node_fraction":float(np.mean(core))},
      "exact_moment_rates":exact_m,
      "results":results,
      "interpretation_note":"alpha is fitted once from raw Q and held fixed; compare relative shape/moment errors between modes, not absolute relaxation time.",
    }
    a.output_json.write_text(json.dumps(out,indent=2)+"\n")
    print("BKW_OPERATOR_AUDIT_COMPLETE")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
