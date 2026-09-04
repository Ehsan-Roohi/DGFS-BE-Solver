#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1] if len(sys.argv)>1 else '.')
out.mkdir(parents=True,exist_ok=True)

def tag(L,Nv): return f'L{L:04.2f}_Nv{Nv}'.replace('.','p')
spec=[(5.25,24),(7.00,32),(8.75,40),(10.50,48)]
box={(L,Nv):json.load(open(out/f'BKW_BOXSCALE_{tag(L,Nv)}.json')) for L,Nv in spec}
ops={M:json.load(open(out/f'BKW_OPERATOR_M{M}.json')) for M in (6,16)}

# 1) Support localization under box enlargement.  Use L>=7 for the main converged trend;
# L=5.25 is shown with an open marker as a truncation-stress case.
modes=['euclidean','fplus','maxwellian']
labels={'euclidean':'Euclidean','fplus':r'$f^+$ weighted','maxwellian':'Maxwellian weighted'}
metrics=[('tail_correction_fraction','Tail correction fraction'),('low_support_correction_fraction','Low-support correction fraction'),('outer_box_correction_fraction','Outer-box correction fraction')]
for key,ylabel in metrics:
    fig,ax=plt.subplots(figsize=(7.2,4.9))
    for m in modes:
        xs=np.array([L for L,Nv in spec]); ys=np.array([box[(L,Nv)]['results'][m][key] for L,Nv in spec])
        ax.semilogy(xs[1:],ys[1:],marker='o',linewidth=1.8,label=labels[m])
        ax.semilogy(xs[:2],ys[:2],linestyle=':',marker='o',linewidth=1.0,alpha=.6)
    ax.axvline(7.0,linestyle='--',linewidth=.9)
    ax.set_xlabel(r'Velocity-box half-width $L$')
    ax.set_ylabel(ylabel)
    ax.set_title('BKW box scaling at fixed velocity spacing, $M_\\omega=6$')
    ax.grid(alpha=.25,which='both'); ax.legend(frameon=False)
    fig.tight_layout()
    stem={'tail_correction_fraction':'FIG_SUPPORT_TAIL','low_support_correction_fraction':'FIG_SUPPORT_LOW','outer_box_correction_fraction':'FIG_SUPPORT_OUTER'}[key]
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# 2) Exact-BKW higher-order moment fidelity at the standard L=7 box.
for M in (6,16):
    fig,ax=plt.subplots(figsize=(7.2,4.9))
    names=['vx4','radial_c4','vx6']; x=np.arange(3)
    for m in ['raw','euclidean','fplus','maxwellian']:
        r=ops[M]['results'][m]['moment_rate_relative_error']
        ax.semilogy(x,[r[n] for n in names],marker='o',linewidth=1.8,label=labels.get(m,m.capitalize()))
    ax.set_xticks(x,[r'$v_x^4$',r'$|v|^4$',r'$v_x^6$'])
    ax.set_ylabel('Relative exact moment-rate error')
    ax.set_title(fr'Exact BKW moment-rate fidelity, $M_\\omega={M}$')
    ax.grid(alpha=.25,which='both'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/f'FIG_BKW_MOMENT_FIDELITY_M{M}.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# 3) Concise evidence table / manuscript notes.
lines=['# Weighting evidence summary','',
'## Velocity-support localization (use L >= 7 as the main box-scaling evidence)','',
'| L | mode | tail fraction | low-support fraction | outer-box fraction |','|---:|---|---:|---:|---:|']
for L,Nv in spec[1:]:
    for m in modes:
        r=box[(L,Nv)]['results'][m]
        lines.append(f"| {L:.2f} | {m} | {r['tail_correction_fraction']:.4e} | {r['low_support_correction_fraction']:.4e} | {r['outer_box_correction_fraction']:.4e} |")
lines += ['','## Interpretation','',
'- L=5.25 is a truncation-stress case, not part of the converged box trend: its raw exact-operator L2 error is much larger than for L>=7.',
'- Euclidean is the minimum ordinary-L2 correction by construction, but most of that correction lies in tail/low-support/outer-box nodes.',
'- f+ and Maxwellian weighting remain strongly localized as the velocity box is enlarged.',
'- Do not claim solver box-independence.  The supported claim is robustness/localization of the projection under box enlargement.',
'- Do not use the nonmonotone c4/vx6 error ratios from the box sweep as a universal accuracy claim; cancellation with raw spectral error can change sign.',
'- Exact BKW at the standard L=7 box and the time-integrated shock provide the higher-order moment-fidelity evidence.' ]
(out/'WEIGHTING_EVIDENCE_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
