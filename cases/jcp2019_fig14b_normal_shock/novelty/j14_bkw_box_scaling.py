#!/usr/bin/env python3
"""Velocity-box scaling audit for conservative collision projections.

Uses the exact positive BKW state at its strongest nonequilibrium point (t=0,
K=0.6), evaluates the DGFS Maxwell-molecule fast-spectral collision operator,
and compares raw, Euclidean, f+-weighted and local-Maxwellian-weighted
conservative corrections as the velocity box is enlarged while approximately
holding velocity spacing fixed.

The goal is not to claim box independence of the raw spectral solver.  Instead,
all projection metrics are measured relative to the raw operator on the SAME
velocity grid, so this audit isolates how each correction behaves when extra
low-population velocity support is added.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

TINY=np.finfo(float).tiny


def bkw_K(t): return 1.0 - 0.4*math.exp(-t/6.0)

def bkw_f(cv,t):
    K=bkw_K(t); v2=np.sum(cv*cv,axis=0)
    pref=1.0/(2.0*(2.0*math.pi*K)**1.5)
    poly=(5.0*K-3.0)/K + (1.0-K)*v2/(K*K)
    return pref*np.exp(-v2/(2.0*K))*poly

def exact_q(cv,t,dt): return (bkw_f(cv,t+dt)-bkw_f(cv,t-dt))/(2.0*dt)

def basis(cv): return np.vstack((np.ones(cv.shape[1]),cv,0.5*np.sum(cv*cv,axis=0)))

def invariant_defect(q,B,cw):
    signed=cw*(B@q); scale=cw*(np.abs(B)@np.abs(q))
    return float(np.max(np.abs(signed)/np.maximum(scale,TINY)))

def project(q,B,w):
    G=(B*w[None,:])@B.T; rhs=B@q
    lam=np.linalg.solve(G,rhs); delta=-w*(B.T@lam)
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

def frac(delta,mask): return float(np.linalg.norm(delta[mask])/max(np.linalg.norm(delta),TINY))

def moment_rates(q,cv,cw):
    v2=np.sum(cv*cv,axis=0)
    return {"vx4":cw*float(np.dot(cv[0]**4,q)),
            "radial_c4":cw*float(np.dot(v2*v2,q)),
            "vx6":cw*float(np.dot(cv[0]**6,q))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--L',type=float,required=True)
    ap.add_argument('--Nv',type=int,required=True)
    ap.add_argument('--Nrho',type=int,required=True)
    ap.add_argument('--M',type=int,default=6)
    ap.add_argument('--time',type=float,default=0.0)
    ap.add_argument('--fd-dt',type=float,default=1e-6)
    ap.add_argument('--output-json',type=Path,required=True)
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
    cfg.set('constants','Nv',str(a.Nv)); cfg.set('constants','Nrho',str(a.Nrho))
    cfg.set('spherical-design-rule','M',str(a.M))
    # Base J14 config has cmax=0 and Tmax=T0, so dev equals nondimensional L.
    cfg.set('velocity-mesh','cmax','0'); cfg.set('velocity-mesh','Tmax',cfg.get('non-dim','T0'))
    cfg.set('velocity-mesh','dev',str(a.L))
    cfg.set('scattering-model','omega','1.0')

    backend=get_backend('cuda',cfg); vm=DGFSVelocityMesh(backend,cfg)
    cls=subclass_where(DGFSScatteringModel,scattering_model=cfg.get('scattering-model','type'))
    scat=cls(backend,cfg,vm); cuda.Context.synchronize()
    cv=np.asarray(vm.cv(),float); cw=float(vm.cw()); B=basis(cv)
    f=bkw_f(cv,a.time); qan=exact_q(cv,a.time,a.fd_dt); fp=np.maximum(f,0.0)
    M,c,T=local_maxwellian(f,cv,cw)

    shape=(1,vm.vsize(),1); zeros=np.zeros(shape,dtype=backend.fpdtype)
    df=backend.matrix(shape,zeros,tags={'align'}); dq=backend.matrix(shape,zeros,tags={'align'})
    df.set(f.astype(backend.fpdtype).reshape(shape)); dq.set(zeros)
    scat.fs(df,dq,0,0); cuda.Context.synchronize()
    qraw=np.asarray(dq.get()[0,:,0],float).copy()

    alpha=float(np.dot(qraw,qan)/max(np.dot(qraw,qraw),TINY))
    fmax=float(np.max(fp)); low=fp <= max(1e-10*fmax,TINY)
    c2=np.sum(c*c,axis=0); sigma=math.sqrt(max(0.5*T,TINY)); tail=np.sqrt(c2)>3*sigma
    outer=np.max(np.abs(cv),axis=0) > 0.75*float(vm.L())
    core=fp > 1e-10*fmax
    exact_m=moment_rates(qan,cv,cw)

    results={}
    modes={'raw':(qraw,np.zeros_like(qraw),float('nan'))}
    for name,w in [('euclidean',np.ones_like(f)),('fplus',fp),('maxwellian',M)]:
        modes[name]=project(qraw,B,w)
    raw_err=rel_l2(alpha*qraw,qan); raw_merr={k:abs(moment_rates(alpha*qraw,cv,cw)[k]-exact_m[k])/max(abs(exact_m[k]),TINY) for k in exact_m}
    for name,(q,delta,cond) in modes.items():
        qs=alpha*q; mr=moment_rates(qs,cv,cw)
        merr={k:float(abs(mr[k]-exact_m[k])/max(abs(exact_m[k]),TINY)) for k in exact_m}
        results[name]={
          'invariant_defect':invariant_defect(q,B,cw),
          'relative_operator_l2_full':rel_l2(qs,qan),
          'relative_operator_l2_core':rel_l2(qs,qan,core),
          'operator_error_increment_over_raw':float(rel_l2(qs,qan)-raw_err),
          'relative_correction_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(qraw),TINY)),
          'scaled_gram_condition':cond,
          'tail_correction_fraction':frac(delta,tail) if name!='raw' else 0.0,
          'low_support_correction_fraction':frac(delta,low) if name!='raw' else 0.0,
          'outer_box_correction_fraction':frac(delta,outer) if name!='raw' else 0.0,
          'moment_rate_relative_error':merr,
          'moment_error_ratio_to_raw':{k:float(merr[k]/max(raw_merr[k],TINY)) for k in merr},
        }

    out={'schema_version':1,'purpose':'BKW velocity-box scaling of conservative projections',
         'BKW':{'K':bkw_K(a.time),'time':a.time},
         'grid':{'L':float(vm.L()),'Nv':int(vm.Nv()),'Nrho':int(vm.Nrho()),'M_omega':int(vm.M()),
                 'dv':float(2*vm.L()/vm.Nv()),'cw':cw},
         'discrete_state':{'mass':cw*float(np.sum(f)),'min_f':float(np.min(f)),'temperature':float(T),
                           'low_support_node_fraction':float(np.mean(low)),'tail_node_fraction':float(np.mean(tail)),
                           'outer_box_node_fraction':float(np.mean(outer))},
         'collision':{'alpha_raw_to_exact':alpha,'raw_operator_l2_error':raw_err},
         'results':results,
         'interpretation_note':'Compare each projection to raw on the same grid; box changes also affect the raw spectral approximation, so projection increments/fractions are the primary box-scaling evidence.'}
    a.output_json.write_text(json.dumps(out,indent=2)+'\n')
    print('BKW_BOX_SCALING_COMPLETE')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
