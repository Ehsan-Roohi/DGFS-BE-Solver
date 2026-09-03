#!/usr/bin/env python3
"""Four-way operator audit on one existing DGFS state.

Evaluates the same raw fast-spectral Q(f,f), then compares:
  raw, Euclidean w=1, distribution w=f+=max(f,0), Maxwellian w=M[f].
No time integration is performed.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import h5py
import numpy as np

TINY=np.finfo(float).tiny
INV=("mass","momentum_x","momentum_y","momentum_z","energy")


def basis(cv):
    return np.vstack((np.ones(cv.shape[1]),cv,0.5*np.sum(cv*cv,axis=0)))


def inv_report(q,B,cw):
    signed=cw*(B@q); scale=cw*(np.abs(B)@np.abs(q))
    defect=np.abs(signed)/np.maximum(scale,TINY)
    return {"signed":dict(zip(INV,map(float,signed))),
            "max_relative_cancellation_defect":float(np.max(defect))}


def project(q,B,w):
    G=(B*w[None,:])@B.T; rhs=B@q
    d=np.sqrt(np.maximum(np.diag(G),TINY)); Gs=G/np.outer(d,d)
    try: lam=np.linalg.solve(G,rhs); solver="solve"
    except np.linalg.LinAlgError: lam=np.linalg.lstsq(G,rhs,rcond=None)[0]; solver="lstsq"
    delta=-w*(B.T@lam); qc=q+delta
    return qc,delta,{"solver":solver,"gram_condition_2":float(np.linalg.cond(G)),
                     "gram_condition_scaled_2":float(np.linalg.cond(Gs)),
                     "lambda_l2":float(np.linalg.norm(lam))}


def load(snapshot,mesh):
    with h5py.File(snapshot,"r") as h: f=np.asarray(h["soln_line_p0"],float)
    with h5py.File(mesh,"r") as h: m=np.asarray(h["spt_line_p0"],float)
    left=m[:,:,0].min(axis=0); right=m[:,:,0].max(axis=0)
    x=np.vstack((left,.5*(left+right),right))
    return f,x


def local_geometry(f,cv,cw):
    fp=np.maximum(f,0.0)
    rho=cw*float(np.sum(f)); mom=cw*(cv@f)
    u=mom/max(rho,TINY); c=cv-u[:,None]; c2=np.sum(c*c,axis=0)
    T=(2/3)*cw*float(np.dot(c2,f))/max(rho,TINY)
    if not np.isfinite(T) or T<=0: raise FloatingPointError("nonpositive Maxwellian T")
    M=rho/(math.pi*T)**1.5*np.exp(-c2/T)
    sigma=math.sqrt(0.5*T); speed=np.sqrt(c2)
    tail=speed>3*sigma
    fmax=float(np.max(fp)); low=fp<=max(1e-8*fmax,TINY); neg=f<0
    return fp,M,u,c,T,tail,low,neg


def fraction(delta,mask):
    return float(np.linalg.norm(delta[mask])/max(np.linalg.norm(delta),TINY))


def moments(qraw,qc,c,cw):
    c2=np.sum(c*c,axis=0)
    kernels={
      "heatflux_x":0.5*c2*c[0],
      "heatflux_z":0.5*c2*c[2],
      "deviatoric_stress_xx":c[0]*c[0]-c2/3,
      "stress_xz":c[0]*c[2],
      "fourth_scalar":c2*c2,
    }
    out={}
    for name,K in kernels.items():
        raw=cw*float(np.dot(K,qraw)); cor=cw*float(np.dot(K,qc)); delta=cor-raw
        scale=cw*float(np.sum(np.abs(K*qraw)))
        out[name]={"raw_signed":raw,"corrected_signed":cor,"delta_signed":delta,
                   "relative_disturbance_to_raw_scale":float(abs(delta)/max(scale,TINY)),
                   "relative_disturbance_to_raw_signed":float(abs(delta)/max(abs(raw),TINY))}
    return out


def metrics(qraw,qc,delta,info,B,cw,c,tail,low,neg):
    return {"invariants":inv_report(qc,B,cw),"projection":info,
      "relative_correction_l2":float(np.linalg.norm(delta)/max(np.linalg.norm(qraw),TINY)),
      "relative_correction_linf":float(np.max(np.abs(delta))/max(np.max(np.abs(qraw)),TINY)),
      "tail_correction_l2_fraction":fraction(delta,tail),
      "low_support_correction_l2_fraction":fraction(delta,low),
      "negative_node_correction_l2_fraction":fraction(delta,neg),
      "max_abs_correction_on_negative_nodes":float(np.max(np.abs(delta[neg])) if np.any(neg) else 0.0),
      "high_order_collision_moments":moments(qraw,qc,c,cw)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label",required=True); ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--snapshot",type=Path,required=True); ap.add_argument("--mesh",type=Path,required=True)
    ap.add_argument("--output-json",type=Path,required=True); ap.add_argument("--repeats",type=int,default=2)
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

    cfg=Inifile.load(str(a.config)); backend=get_backend("cuda",cfg); vm=DGFSVelocityMesh(backend,cfg)
    cls=subclass_where(DGFSScatteringModel,scattering_model=cfg.get("scattering-model","type"))
    scattering=cls(backend,cfg,vm); cuda.Context.synchronize()
    soln,x=load(a.snapshot,a.mesh); cv=np.asarray(vm.cv(),float); cw=float(vm.cw()); B=basis(cv)
    shape=(1,vm.vsize(),1); zeros=np.zeros(shape,dtype=backend.fpdtype)
    df=backend.matrix(shape,zeros,tags={"align"}); dq=backend.matrix(shape,zeros,tags={"align"})
    rows=[]
    for e in range(soln.shape[2]):
      for uidx in range(soln.shape[0]):
        f=np.asarray(soln[uidx,:,e],float); df.set(f.astype(backend.fpdtype).reshape(shape))
        dq.set(zeros); scattering.fs(df,dq,0,0); cuda.Context.synchronize()
        times=[]
        for _ in range(max(1,a.repeats)):
          dq.set(zeros); st=time.perf_counter(); scattering.fs(df,dq,0,0); cuda.Context.synchronize(); times.append(time.perf_counter()-st)
        qraw=np.asarray(dq.get()[0,:,0],float).copy()
        fp,M,umean,c,T,tail,low,neg=local_geometry(f,cv,cw)
        modes={}
        for name,w in (("euclidean",np.ones_like(f)),("fplus",fp),("maxwellian",M)):
          qc,delta,info=project(qraw,B,w)
          modes[name]=metrics(qraw,qc,delta,info,B,cw,c,tail,low,neg)
        rows.append({"element":e,"solution_point":uidx,"x_nondim":float(x[uidx,e]),
          "rho":float(cw*np.sum(f)),"T":float(T),"min_f":float(np.min(f)),
          "negative_mass_fraction":float(np.sum(np.maximum(-f,0))/max(np.sum(fp),TINY)),
          "raw":{"invariants":inv_report(qraw,B,cw),"collision_time_ms":1e3*float(np.median(times))},
          **modes})
        print(f"J14_FOURWAY_POINT e={e} u={uidx} raw={rows[-1]['raw']['invariants']['max_relative_cancellation_defect']:.3e} eu={modes['euclidean']['invariants']['max_relative_cancellation_defect']:.3e} fp={modes['fplus']['invariants']['max_relative_cancellation_defect']:.3e} M={modes['maxwellian']['invariants']['max_relative_cancellation_defect']:.3e}",flush=True)

    modes=("euclidean","fplus","maxwellian")
    def vals(mode,*keys):
      z=[]
      for r in rows:
        v=r[mode]
        for k in keys: v=v[k]
        z.append(float(v))
      return np.asarray(z)
    summary={"label":a.label,"M_omega":int(vm.M()),"points":len(rows),
      "definitions":{"tail":"|c| > 3 sqrt(T/2)","low_support":"f+ <= 1e-8 max(f+)","maxwellian":"local M[f] from signed discrete rho,u,T"},
      "max_invariant_defect":{},"median_relative_correction_l2":{},"median_scaled_gram_condition":{},
      "median_tail_correction_fraction":{},"median_low_support_correction_fraction":{},
      "median_negative_node_correction_fraction":{},"median_high_order_relative_disturbance":{}}
    summary["raw_max_invariant_defect"]=float(max(r["raw"]["invariants"]["max_relative_cancellation_defect"] for r in rows))
    for mode in modes:
      summary["max_invariant_defect"][mode]=float(np.max(vals(mode,"invariants","max_relative_cancellation_defect")))
      summary["median_relative_correction_l2"][mode]=float(np.median(vals(mode,"relative_correction_l2")))
      summary["median_scaled_gram_condition"][mode]=float(np.median(vals(mode,"projection","gram_condition_scaled_2")))
      summary["median_tail_correction_fraction"][mode]=float(np.median(vals(mode,"tail_correction_l2_fraction")))
      summary["median_low_support_correction_fraction"][mode]=float(np.median(vals(mode,"low_support_correction_l2_fraction")))
      summary["median_negative_node_correction_fraction"][mode]=float(np.median(vals(mode,"negative_node_correction_l2_fraction")))
      summary["median_high_order_relative_disturbance"][mode]={}
      for mom in ("heatflux_x","heatflux_z","deviatoric_stress_xx","stress_xz","fourth_scalar"):
        summary["median_high_order_relative_disturbance"][mode][mom]=float(np.median(vals(mode,"high_order_collision_moments",mom,"relative_disturbance_to_raw_scale")))
    out={"schema_version":1,"purpose":"raw vs Euclidean vs fplus vs Maxwellian operator audit","config":str(a.config),"snapshot":str(a.snapshot),"mesh":str(a.mesh),"summary":summary,"records":rows}
    a.output_json.write_text(json.dumps(out,indent=2)+"\n")
    print("FOURWAY_MAXWELLIAN_AUDIT_COMPLETE")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
