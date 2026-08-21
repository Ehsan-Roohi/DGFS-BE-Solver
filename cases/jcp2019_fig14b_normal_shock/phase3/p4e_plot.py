#!/usr/bin/env python3
"""Publication-style physical profiles and time history for P4E."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER=("run_M16_raw","run_M16_transverse","run_M16_fplus","run_M24_raw")
LABEL={
 "run_M16_raw":r"$M_\Omega=16$, raw",
 "run_M16_transverse":r"$M_\Omega=16$, transverse",
 "run_M16_fplus":r"$M_\Omega=16$, five-moment $f^+$",
 "run_M24_raw":r"$M_\Omega=24$, raw reference"}
COLOR={"run_M16_raw":"#C47A35","run_M16_transverse":"#18856F",
       "run_M16_fplus":"#355FA3","run_M24_raw":"#111111"}
MARK={"run_M16_raw":"^","run_M16_transverse":"D","run_M16_fplus":"o","run_M24_raw":"s"}
PANELS=(("rho_kg_m3",r"$\rho$ [kg m$^{-3}$]"),("ux_m_s",r"$u_x$ [m s$^{-1}$]"),
        ("T_K",r"$T$ [K]"),("qx_W_m2",r"$q_x$ [W m$^{-2}$]"),
        ("Pxx_minus_p_Pa",r"$P_{xx}-p$ [Pa]"),("uz_m_s",r"$u_z$ [m s$^{-1}$]"))

def read(path):
 with path.open(newline="") as f: return list(csv.DictReader(f))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--profiles",type=Path,required=True)
 ap.add_argument("--history",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
 a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
 profiles=read(a.profiles); history=read(a.history)
 final=[r for r in profiles if abs(float(r["time"])-1.0)<1e-12]
 values={n:sorted((r for r in final if r["run"]==n),key=lambda r:int(r["dg_point_index"])) for n in ORDER}
 plt.rcParams.update({"font.size":10.5,"axes.labelsize":11.5,"legend.fontsize":9})
 for difference in (False,True):
  fig,axs=plt.subplots(2,3,figsize=(13.4,7.4),sharex=True,constrained_layout=True)
  for letter,ax,(key,ylab) in zip("abcdef",axs.flat,PANELS):
   names=ORDER if not difference else ORDER[:3]
   ref=[float(r[key]) for r in values["run_M24_raw"]]
   for name in names:
    x=[float(r["x_mm"]) for r in values[name]]; y=[float(r[key]) for r in values[name]]
    if difference: y=[v-b for v,b in zip(y,ref)]
    ax.plot(x,y,color=COLOR[name],marker=MARK[name],ms=3.8,
            markerfacecolor="white",lw=2 if name=="run_M24_raw" else 1.25,label=LABEL[name])
   if difference or key=="uz_m_s": ax.axhline(0,color="#666",lw=.8)
   ax.grid(alpha=.28); ax.set_ylabel(("difference in " if difference else "")+ylab)
   ax.set_title(f"({letter})",loc="left",fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
  for ax in axs[1]: ax.set_xlabel(r"$x$ [mm]")
  h,l=axs[0,0].get_legend_handles_labels()
  fig.suptitle(r"Normal shock $Ma=1.59$, $t=1.00$; 24 DG values",y=1.065,fontsize=12.5)
  fig.legend(h,l,ncol=4,loc="lower center",bbox_to_anchor=(.5,1.005),frameon=False)
  stem="p4e_differences_t1" if difference else "p4e_profiles_t1"
  fig.savefig(a.out_dir/(stem+".png"),dpi=300,bbox_inches="tight")
  fig.savefig(a.out_dir/(stem+".pdf"),bbox_inches="tight")

 fig,ax=plt.subplots(figsize=(7.2,4.5),constrained_layout=True)
 for name in ORDER:
  rows=sorted((r for r in history if r["run"]==name),key=lambda r:float(r["time"]))
  ax.semilogy([float(r["time"]) for r in rows],
              [max(float(r["max_abs_uz_m_s"]),1e-16) for r in rows],
              color=COLOR[name],marker=MARK[name],markerfacecolor="white",label=LABEL[name])
 ax.set(xlabel=r"time $t$",ylabel=r"max $|u_z|$ [m s$^{-1}$]",
        title=r"Symmetry error history, normal shock $Ma=1.59$")
 ax.grid(alpha=.3,which="both"); ax.legend(frameon=False)
 fig.savefig(a.out_dir/"p4e_uz_time_history.png",dpi=300,bbox_inches="tight")
 fig.savefig(a.out_dir/"p4e_uz_time_history.pdf",bbox_inches="tight")
 print("P4E_PHYSICAL_FIGURES_COMPLETE")
if __name__=="__main__": main()
