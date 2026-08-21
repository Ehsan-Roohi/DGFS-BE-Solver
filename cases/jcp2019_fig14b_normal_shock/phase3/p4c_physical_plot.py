#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER=("run_M16_raw","run_M16_euclidean","run_M16_fplus","run_M24_raw")
LABEL={
 "run_M16_raw":r"$M_\Omega=16$, raw",
 "run_M16_euclidean":r"$M_\Omega=16$, Euclidean",
 "run_M16_fplus":r"$M_\Omega=16$, conservative $f^+$",
 "run_M24_raw":r"$M_\Omega=24$, raw reference"}
COLOR={"run_M16_raw":"#C47A35","run_M16_euclidean":"#18856F",
       "run_M16_fplus":"#355FA3","run_M24_raw":"#111111"}
MARK={"run_M16_raw":"^","run_M16_euclidean":"D","run_M16_fplus":"o","run_M24_raw":"s"}

def prof(rec,key,scale):
 return np.asarray([[rec["points"][u][e][key] for e in range(len(rec["points"][0]))]
                    for u in range(3)])*scale

def segments(ax,x,y,name,lw=1.3):
 for e in range(x.shape[1]):
  ax.plot(x[:,e],y[:,e],color=COLOR[name],marker=MARK[name],ms=4.0,lw=lw,
          markerfacecolor="white",label=LABEL[name] if e==0 else None)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--json",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
 a=ap.parse_args(); d=json.loads(a.json.read_text()); runs={r["run"]:r for r in d["runs"]}
 nd=d["nondim"]; rho0=float(nd["rho0"]); u0=float(nd["u0"]); T0=float(nd["T0"])
 panels=(("rho",rho0,r"$\rho$ [kg m$^{-3}$]"),("ux",u0,r"$u_x$ [m s$^{-1}$]"),
         ("T",T0,r"$T$ [K]"),("qx",rho0*u0**3,r"$q_x$ [W m$^{-2}$]"),
         ("Pxx_minus_p",rho0*u0**2,r"$P_{xx}-p$ [Pa]"),("uz",u0,r"$u_z$ [m s$^{-1}$]"))
 ne=len(runs[ORDER[0]]["points"][0]); edge=np.linspace(-15,15,ne+1)
 x=np.vstack((edge[:-1],(edge[:-1]+edge[1:])/2,edge[1:])); vals={n:{k:prof(runs[n],k,s) for k,s,_ in panels} for n in ORDER}
 plt.rcParams.update({"font.size":10.5,"axes.labelsize":11.5,"legend.fontsize":9.5})
 for difference in (False,True):
  fig,axs=plt.subplots(2,3,figsize=(13.4,7.4),sharex=True,constrained_layout=True)
  for letter,ax,(k,s,ylab) in zip("abcdef",axs.flat,panels):
   names=ORDER if not difference else ORDER[:3]
   for n in names:
    y=vals[n][k] if not difference else vals[n][k]-vals["run_M24_raw"][k]
    segments(ax,x,y,n,2.0 if n=="run_M24_raw" else 1.15)
   if difference or k=="uz": ax.axhline(0,color="#666",lw=.8)
   for xb in edge[1:-1]: ax.axvline(xb,color="#e6e6e6",lw=.5,zorder=0)
   ax.grid(axis="y",alpha=.3); ax.set_ylabel((r"difference in " if difference else "")+ylab)
   ax.set_title(f"({letter})",loc="left",fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
  for ax in axs[1]: ax.set_xlabel(r"$x$ [mm]")
  h,l=axs[0,0].get_legend_handles_labels(); title=rf"Normal shock $Ma=1.59$, $t=0.25$; {3*ne} DG values"
  fig.suptitle(title, y=1.065, fontsize=12.5)
  fig.legend(h,l,ncol=len(h),loc="lower center",bbox_to_anchor=(0.5,1.005),frameon=False)
  stem="p4c_differences" if difference else "p4c_profiles"
  fig.savefig(a.out.with_name(stem+".png"),dpi=300,bbox_inches="tight")
  fig.savefig(a.out.with_name(stem+".pdf"),bbox_inches="tight")
 print("P4C_PHYSICAL_FIGURES_COMPLETE")
if __name__=="__main__": main()
