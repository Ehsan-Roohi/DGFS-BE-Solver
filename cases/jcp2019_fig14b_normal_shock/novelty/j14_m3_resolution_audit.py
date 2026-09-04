#!/usr/bin/env python3
from __future__ import annotations
import argparse, configparser, json, math
from pathlib import Path
import h5py, numpy as np
TINY=np.finfo(float).tiny

def basis(cv): return np.vstack((np.ones(cv.shape[1]),cv,0.5*np.sum(cv*cv,axis=0)))
def invdef(q,B,cw):
    s=cw*(B@q); sc=cw*(np.abs(B)@np.abs(q)); return float(np.max(np.abs(s)/np.maximum(sc,TINY)))
def project(q,B,w):
    G=(B*w[None,:])@B.T; rhs=B@q; lam=np.linalg.solve(G,rhs); d=-w*(B.T@lam); qc=q+d
    ds=np.sqrt(np.maximum(np.diag(G),TINY)); Gs=G/np.outer(ds,ds)
    return qc,d,float(np.linalg.cond(Gs))
def geom(f,M,theta):
    fp=np.maximum(f,0.0)
    if theta==0: return M
    w=np.zeros_like(f); mask=fp>0
    w[mask]=np.exp((1-theta)*np.log(np.maximum(M[mask],TINY))+theta*np.log(fp[mask]))
    return w
def local(f,cv,cw):
    fp=np.maximum(f,0); rho=cw*np.sum(f); u=cw*(cv@f)/max(rho,TINY); c=cv-u[:,None]; c2=np.sum(c*c,axis=0)
    T=(2/3)*cw*np.dot(c2,f)/max(rho,TINY); M=rho/(math.pi*T)**1.5*np.exp(-c2/T)
    return fp,M,c,T
def moments(q,c,cw):
    c2=np.sum(c*c,axis=0)
    K={'qx':0.5*c2*c[0],'qz':0.5*c2*c[2],'Pdev_xx':c[0]**2-c2/3,'Pxz':c[0]*c[2],'c4':c2*c2}
    return {k:float(cw*np.dot(v,q)) for k,v in K.items()}
def dist(mraw,mcorr):
    out={};
    for k in mraw: out[k]=abs(mcorr[k]-mraw[k])
    return out
def embed(f,n0,n1):
    if n1==n0: return f.copy()
    if n1<n0 or (n1-n0)%2: raise ValueError('target Nv must exceed source Nv by an even number')
    p=(n1-n0)//2; a=f.reshape(n0,n0,n0); b=np.zeros((n1,n1,n1),float); b[p:p+n0,p:p+n0,p:p+n0]=a; return b.ravel()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--label',required=True); ap.add_argument('--Nv',type=int,required=True); ap.add_argument('--Nrho',type=int,required=True); ap.add_argument('--M',type=int,required=True); ap.add_argument('--L',type=float,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    import mpi4py.rc; mpi4py.rc.initialize=False
    from mpi4py import MPI
    if not MPI.Is_initialized(): MPI.Init()
    from frfs.backends import get_backend
    from frfs.inifile import Inifile
    from frfs.solvers.dgfs.scattering import DGFSScatteringModel
    from frfs.solvers.dgfs.velocitymesh import DGFSVelocityMesh
    from frfs.util import subclass_where
    import pycuda.driver as cuda
    cfg=Inifile.load(str(a.config)); cfg.set('constants','Nv',str(a.Nv)); cfg.set('constants','Nrho',str(a.Nrho)); cfg.set('spherical-design-rule','M',str(a.M)); cfg.set('velocity-mesh','cmax','0'); cfg.set('velocity-mesh','Tmax',cfg.get('non-dim','T0')); cfg.set('velocity-mesh','dev',str(a.L))
    backend=get_backend('cuda',cfg); vm=DGFSVelocityMesh(backend,cfg); cls=subclass_where(DGFSScatteringModel,scattering_model=cfg.get('scattering-model','type')); scat=cls(backend,cfg,vm); cuda.Context.synchronize()
    cv=np.asarray(vm.cv(),float); cw=float(vm.cw()); B=basis(cv)
    with h5py.File(a.snapshot,'r') as h: F0=np.asarray(h['soln_line_p0'],float)
    n0=round(F0.shape[1]**(1/3)); dv0=2*8.75/n0; dvt=2*a.L/a.Nv
    if abs(dv0-dvt)>1e-12: raise ValueError(f'dv mismatch source={dv0} target={dvt}')
    shape=(1,vm.vsize(),1); zeros=np.zeros(shape,dtype=backend.fpdtype); df=backend.matrix(shape,zeros,tags={'align'}); dq=backend.matrix(shape,zeros,tags={'align'})
    rec=[]
    for e in range(F0.shape[2]):
      for uidx in range(F0.shape[0]):
        f=embed(np.asarray(F0[uidx,:,e],float),n0,a.Nv); fp,Mx,c,T=local(f,cv,cw); df.set(f.astype(backend.fpdtype).reshape(shape)); dq.set(zeros); scat.fs(df,dq,0,0); cuda.Context.synchronize(); q=np.asarray(dq.get()[0,:,0],float).copy(); mr=moments(q,c,cw)
        modes={}
        for name,w in [('euclidean',np.ones_like(f)),('fplus',fp),('maxwellian',Mx),('theta025',geom(f,Mx,0.25))]:
          qc,delta,cond=project(q,B,w); modes[name]={'inv_defect':invdef(qc,B,cw),'rel_corr_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(q),TINY)),'gram_cond':cond,'moment_abs_disturbance':dist(mr,moments(qc,c,cw))}
        rec.append({'element':e,'solution_point':uidx,'raw_inv_defect':invdef(q,B,cw),'raw_q_norm':float(np.linalg.norm(q)),'raw_moments':mr,'modes':modes})
    out={'label':a.label,'grid':{'Nv':a.Nv,'Nrho':a.Nrho,'M_omega':a.M,'L':a.L,'dv':dvt},'records':rec}
    out['summary']={'raw_max_inv_defect':max(r['raw_inv_defect'] for r in rec),'raw_median_inv_defect':float(np.median([r['raw_inv_defect'] for r in rec])),'median_rel_corr_l2':{m:float(np.median([r['modes'][m]['rel_corr_l2'] for r in rec])) for m in ['euclidean','fplus','maxwellian','theta025']},'median_gram_cond':{m:float(np.median([r['modes'][m]['gram_cond'] for r in rec])) for m in ['euclidean','fplus','maxwellian','theta025']}}
    a.output.write_text(json.dumps(out,indent=2)+'\n'); print('M3_RESOLUTION_AUDIT_COMPLETE'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
