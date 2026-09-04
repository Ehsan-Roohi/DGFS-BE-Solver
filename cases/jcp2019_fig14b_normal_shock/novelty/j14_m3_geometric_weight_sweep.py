#!/usr/bin/env python3
"""Mach-3 operator sweep for geometric interpolation between M[f] and f+.

For theta in [0, .25, .5, .75, 1], use
    w_theta = M[f]^(1-theta) * (f+)^theta.
For theta>0, nodes with f<=0 have zero weight, retaining the support restriction
of f+. A global normalization of w is applied for numerical range only; it does
not change the projection.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import h5py,numpy as np
TINY=np.finfo(float).tiny
THETAS=(0.0,0.25,0.5,0.75,1.0)
INV=("mass","momentum_x","momentum_y","momentum_z","energy")

def basis(cv): return np.vstack((np.ones(cv.shape[1]),cv,0.5*np.sum(cv*cv,axis=0)))
def invdef(q,B,cw):
    s=cw*(B@q); a=cw*(np.abs(B)@np.abs(q)); return float(np.max(np.abs(s)/np.maximum(a,TINY)))
def project(q,B,w):
    G=(B*w[None,:])@B.T; rhs=B@q
    d=np.sqrt(np.maximum(np.diag(G),TINY)); Gs=G/np.outer(d,d)
    lam=np.linalg.solve(G,rhs); delta=-w*(B.T@lam)
    return q+delta,delta,float(np.linalg.cond(Gs))
def load(snapshot,mesh):
    with h5py.File(snapshot,'r') as h: f=np.asarray(h['soln_line_p0'],float)
    with h5py.File(mesh,'r') as h: m=np.asarray(h['spt_line_p0'],float)
    left=m[:,:,0].min(axis=0); right=m[:,:,0].max(axis=0); x=np.vstack((left,.5*(left+right),right))
    return f,x
def geom(f,cv,cw):
    fp=np.maximum(f,0); rho=cw*float(np.sum(f)); u=cw*(cv@f)/max(rho,TINY)
    c=cv-u[:,None]; c2=np.sum(c*c,axis=0); T=(2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    if not np.isfinite(T) or T<=0: raise FloatingPointError('bad T')
    M=rho/(math.pi*T)**1.5*np.exp(-c2/T)
    sig=math.sqrt(.5*T); tail=np.sqrt(c2)>3*sig
    fmax=float(np.max(fp)); low=fp<=max(1e-8*fmax,TINY); neg=f<0
    return fp,M,c,T,tail,low,neg
def weight_theta(fp,M,theta):
    if theta==0: w=M.copy()
    elif theta==1: w=fp.copy()
    else:
        w=np.zeros_like(fp); mask=fp>0
        w[mask]=np.exp(theta*np.log(np.maximum(fp[mask],TINY))+(1-theta)*np.log(np.maximum(M[mask],TINY)))
    mx=float(np.max(w));
    if not np.isfinite(mx) or mx<=0: raise FloatingPointError('bad weight')
    return w/mx
def frac(d,m): return float(np.linalg.norm(d[m])/max(np.linalg.norm(d),TINY))
def homom(qraw,qc,c,cw):
    c2=np.sum(c*c,axis=0)
    K={'qx':.5*c2*c[0],'qz':.5*c2*c[2],'Pdev_xx':c[0]*c[0]-c2/3,'Pxz':c[0]*c[2],'c4':c2*c2}
    out={}
    for n,k in K.items():
        dr=cw*float(np.dot(k,qc-qraw)); sc=cw*float(np.sum(np.abs(k*qraw)))
        out[n]=float(abs(dr)/max(sc,TINY))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--snapshot',type=Path,required=True); ap.add_argument('--mesh',type=Path,required=True); ap.add_argument('--output-json',type=Path,required=True)
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
    cfg=Inifile.load(str(a.config)); backend=get_backend('cuda',cfg); vm=DGFSVelocityMesh(backend,cfg)
    cls=subclass_where(DGFSScatteringModel,scattering_model=cfg.get('scattering-model','type')); scat=cls(backend,cfg,vm); cuda.Context.synchronize()
    F,x=load(a.snapshot,a.mesh); cv=np.asarray(vm.cv(),float); cw=float(vm.cw()); B=basis(cv)
    shape=(1,vm.vsize(),1); zeros=np.zeros(shape,dtype=backend.fpdtype); df=backend.matrix(shape,zeros,tags={'align'}); dq=backend.matrix(shape,zeros,tags={'align'})
    rec=[]
    for e in range(F.shape[2]):
      for uidx in range(F.shape[0]):
        f=np.asarray(F[uidx,:,e],float); df.set(f.astype(backend.fpdtype).reshape(shape)); dq.set(zeros); scat.fs(df,dq,0,0); cuda.Context.synchronize(); q=np.asarray(dq.get()[0,:,0],float).copy()
        fp,M,c,T,tail,low,neg=geom(f,cv,cw); modes={}
        for th in THETAS:
            w=weight_theta(fp,M,th); qc,d,cond=project(q,B,w)
            modes[f'{th:.2f}']={'inv_defect':invdef(qc,B,cw),'corr_L2':float(np.linalg.norm(d)/max(np.linalg.norm(q),TINY)),'cond':cond,'tail':frac(d,tail),'low':frac(d,low),'negative':frac(d,neg),'mom':homom(q,qc,c,cw)}
        rec.append({'element':e,'solution_point':uidx,'x':float(x[uidx,e]),'raw_inv_defect':invdef(q,B,cw),'min_f':float(np.min(f)),'modes':modes})
    def vals(th,key,*sub):
        z=[]
        for r in rec:
            v=r['modes'][f'{th:.2f}'][key]
            for k in sub: v=v[k]
            z.append(float(v))
        return np.asarray(z)
    summary={'label':a.label,'points':len(rec),'raw_max_inv_defect':float(max(r['raw_inv_defect'] for r in rec)),'theta':{}}
    for th in THETAS:
        summary['theta'][f'{th:.2f}']={'max_inv_defect':float(np.max(vals(th,'inv_defect'))),'median_corr_L2':float(np.median(vals(th,'corr_L2'))),'median_cond':float(np.median(vals(th,'cond'))),'median_tail':float(np.median(vals(th,'tail'))),'median_low':float(np.median(vals(th,'low'))),'median_negative':float(np.median(vals(th,'negative'))),'median_mom':{k:float(np.median(vals(th,'mom',k))) for k in ['qx','qz','Pdev_xx','Pxz','c4']}}
    a.output_json.write_text(json.dumps({'schema_version':1,'definition':'w_theta=M^(1-theta)*(f+)^theta; weights globally normalized without changing projection','summary':summary,'records':rec},indent=2)+'\n')
    print('M3_GEOMETRIC_SWEEP_COMPLETE'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
