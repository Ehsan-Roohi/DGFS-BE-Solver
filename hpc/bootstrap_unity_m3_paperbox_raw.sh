#!/usr/bin/env bash
# One raw Mach-3 run using the JCP-2019 Table-2 velocity discretization.
# Not a claim of full replication: the existing gas constant and exact RH
# endpoints are retained; the table's rounded velocities are reported separately.
set -Eeuo pipefail
trap 'rc=$?; echo "M3_PAPERBOX_BOOTSTRAP_FAILED rc=$rc line=$LINENO" >&2; exit "$rc"' ERR
ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
PY=${DGFS_ENV:-$ROOT/dgfs_py310}/bin/python
SRC=${DGFS_SOLVER_SRC:-$ROOT/j14novclose_20260826_211439/m16_steady_20260831_125816/src}
INPUT=${DGFS_M3_INPUT:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/dgfs_m3_m60_stage1_20260905_001422/M3_M60_raw}
SCRATCH=${DGFS_SCRATCH:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
JOBNAME=dgfs-m3-pbox
for p in "$PY" "$SRC/frfs/__main__.py" "$INPUT/M3_M60_raw.ini" "$INPUT/mesh.frfsm"; do
    [[ -f "$p" ]] || { echo "MISSING_INPUT=$p" >&2; exit 2; }
done
[[ -d "$SCRATCH" && -w "$SCRATCH" && -d "$OUT" && -w "$OUT" ]] || { echo 'DIRECTORY_NOT_WRITABLE'; exit 2; }
OUT=$(cd "$OUT" && pwd -P)
# Prevent accidental repeated submissions from the same output directory.
exec 9>"$OUT/.M3_RAW_PAPERBOX_SUBMIT.lock"
flock -n 9 || { echo 'ANOTHER_SUBMISSION_IN_PROGRESS'; exit 3; }
if [[ ${DGFS_PREPARE_ONLY:-0} != 1 ]]; then
    ACTIVE=$(squeue --me --noheader --format='%j')
    if grep -Fxq "$JOBNAME" <<< "$ACTIVE"; then echo 'M3_RAW_PAPERBOX_ALREADY_ACTIVE'; exit 3; fi
fi
PYTHONNOUSERSITE=1 "$PY" -c 'import numpy,h5py; print("NUMPY_H5PY=PASS")'
WORK=$(mktemp -d "$SCRATCH/dgfs_m3_paperbox_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
cp "$INPUT/M3_M60_raw.ini" "$WORK/input_original.ini"
cp "$INPUT/mesh.frfsm" "$WORK/mesh.frfsm"
cp -- "${BASH_SOURCE[0]}" "$WORK/bootstrap_used.sh"
cat > "$WORK/paperbox_tools.py" <<'PY'
from __future__ import annotations
import argparse, configparser, csv, hashlib, json, math, shutil, sys, zipfile
from pathlib import Path
import h5py
import numpy as np

NAMES = ('mass_flux', 'momentum_flux', 'energy_flux')

def config(p):
    c=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=(';','#'))
    c.optionxform=str
    with Path(p).open() as f: c.read_file(f)
    return c

def scales(c):
    rho0=c.getfloat('non-dim','rho0'); T0=c.getfloat('non-dim','T0')
    R=8.3144598/c.getfloat('non-dim','molarMass0')
    return rho0,T0,R,math.sqrt(2*R*T0)

def vgrid(c):
    _,T0,_,u0=scales(c); N=c.getint('constants','Nv')
    L=c.getfloat('velocity-mesh','cmax')/u0+c.getfloat('velocity-mesh','dev')*math.sqrt(c.getfloat('velocity-mesh','Tmax')/T0)
    a=-L+(np.arange(N)+0.5)*(2*L/N)
    v=np.stack(np.meshgrid(a,a,a,indexing='ij')).reshape(3,-1)
    return v,(2*L/N)**3,L

def eq_state(c,sec):
    rho0,T0,_,u0=scales(c)
    return c.getfloat(sec,'rho')/rho0,np.array([c.getfloat(sec,k) for k in ('ux','uy','uz')])/u0,c.getfloat(sec,'T')/T0

def eq_flux(r,u,T):
    return np.array([r*u[0],r*(u[0]**2+T/2),r*u[0]*(0.5*np.dot(u,u)+1.25*T)])

def maxwellian(v,r,u,T):
    return r/(math.pi*T)**1.5*np.exp(-np.sum((v-u[:,None])**2,axis=0)/T)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(w):
    c=config(w/'input_original.ini'); rho0,T0,R,u0=scales(c)
    if not math.isclose(c.getfloat('non-dim','H0'),.03,rel_tol=1e-10):
        raise ValueError('Expected H0=0.03 m')
    if c.getint('solver','order')!=2 or c.get('solver-time-integrator','scheme')!='dgfs-tvd-rk2':
        raise ValueError('Expected quadratic GLL DG and dgfs-tvd-rk2; refusing to change the transport/time scheme')
    if c.get('solver-elements-line','soln-pts')!='gauss-legendre-lobatto':
        raise ValueError('Expected GLL solution nodes')
    if c.get('scattering-model','type')!='vhs-gll' or not math.isclose(c.getfloat('scattering-model','omega'),.5):
        raise ValueError('Expected vhs-gll hard-sphere collision model')
    if not math.isclose(c.getfloat('scattering-model','dRef'),2.17e-10,rel_tol=1e-10):
        raise ValueError('Unexpected molecular diameter')
    for s in ('soln-bcs-left','soln-bcs-right'):
        if c.get(s,'type')!='dgfs-inlet-normalshock':raise ValueError('Unexpected boundary type')
    r1,u1,t1=eq_state(c,'soln-bcs-left'); r2,u2,t2=eq_state(c,'soln-bcs-right')
    mach=u1[0]/math.sqrt((5/3)*t1/2)
    J1=eq_flux(r1,u1,t1); J2=eq_flux(r2,u2,t2)
    rh=float(np.max(np.abs(J2-J1)/np.abs(J1)))
    if abs(mach-3)>1e-8 or rh>1e-8:raise ValueError(f'Inconsistent RH states: Mach={mach}, flux mismatch={rh}')
    with h5py.File(w/'mesh.frfsm','r') as h:
        sp=np.asarray(h['spt_line_p0'])
    if sp.ndim!=3 or sp.shape[1]!=8 or sp.shape[2]!=1 or not np.allclose([sp.min(),sp.max()],[-.5,.5],atol=1e-12):
        raise ValueError(f'Unexpected mesh shape or extent: {sp.shape}')
    if shutil.disk_usage(w).free<1024**3:raise OSError('Need at least 1 GiB free scratch space')
    for sec,k,val in [('constants','Nv','48'),('constants','Nrho','48'),('constants','NvBatchSize','64'),
       ('spherical-design-rule','M','6'),('velocity-mesh','dev','11'),('velocity-mesh','cmax','0'),
       ('velocity-mesh','Tmax',str(T0)),('scattering-model','projection','none'),
       ('solver-time-integrator','tstart','0'),('solver-time-integrator','tend','10.25'),
       ('solver-time-integrator','dt','0.0005')]: c.set(sec,k,val)
    # Keep exactly the existing initial-condition expressions and boundary states.
    for sec in c.sections():
        if sec.startswith('soln-plugin-') and sec not in ('soln-plugin-nancheck','soln-plugin-dgfsresidualstd','soln-plugin-dgfsdistwriterstd','soln-plugin-dgfsmomwriterstd'):
            raise ValueError(f'Unreviewed plugin in input: {sec}')
    c.set('soln-plugin-dgfsresidualstd','nsteps','100')
    c.set('soln-plugin-dgfsresidualstd','file','kinetic_residual.csv')
    c.remove_option('soln-plugin-dgfsresidualstd','normalisation-resid')
    c.set('soln-plugin-dgfsdistwriterstd','dt-out','1.025')
    c.set('soln-plugin-dgfsdistwriterstd','basedir','.')
    c.set('soln-plugin-dgfsdistwriterstd','basename','dist_M3_RAW_PAPERBOX-{t:.3f}')
    c.set('soln-plugin-dgfsmomwriterstd','dt-out','0.25')
    c.set('soln-plugin-dgfsmomwriterstd','basedir','.')
    c.set('soln-plugin-dgfsmomwriterstd','basename','bulk_M3_RAW_PAPERBOX-{t:.3f}')
    with (w/'M3_RAW_PAPERBOX.ini').open('w') as f:c.write(f)
    v,cw,L=vgrid(c); v2=np.sum(v*v,axis=0); phi=np.vstack([v[0],v[0]**2,.5*v[0]*v2])
    dq={}
    for label, r,u,T,J in [('left',r1,u1,t1,J1),('right',r2,u2,t2,J2)]:
        fm=maxwellian(v,r,u,T); discrete=cw*(phi@fm)
        err=float(np.max(np.abs(discrete-J)/np.abs(J)))
        dq[label]={'flux_relative_quadrature_error':err,'discrete_flux':discrete.tolist()}
        if err>1e-7:raise ValueError(f'Endpoint velocity quadrature failed: {label}, {err}')
    manifest={'scope':'raw Mach-3 with published velocity/space resolution; not yet a validated reproduction',
       'literature':'Jaiswal, Alexeenko, Hu; JCP 378 (2019); Table 2 / Fig 15; arXiv:1809.10186v2',
       'velocity_grid':{'L':L,'Nv':48,'Nrho':48,'M_omega':6},'nelements':8,'DG_polynomial_degree':2,
       'R_specific':R,'Mach_computed':mach,'RH_analytic_flux_relative_mismatch':rh,'endpoint_quadrature':dq,
       'rho1':rho0*r1,'rho2':rho0*r2,'T1':T0*t1,'T2':T0*t2,'u1':u0*u1[0],'u2':u0*u2[0],
       'paper_printed_values':{'u1':2639.19,'u2':879.73,'T1':223,'T2':817.67},
       'differences_from_paper':'Retain existing gas constant and RH-consistent endpoints; paper rounded velocities are not silently substituted. dt=0.0005 and t_end=10.25 are our choices. Initial expressions retained from prior campaign.',
       'initialization':'unchanged linear macroscopic interpolation, local Maxwellian; not a two-Maxwellian blend',
       'input_ini_sha256':sha(w/'input_original.ini'),'mesh_sha256':sha(w/'mesh.frfsm'),
       'numerical_screen':{'flux_span':1e-3,'snapshot_drift_per_unit_time':1e-4,'negative_mass_fraction':1e-5,'outer_mass_fraction':1e-5,'required_consecutive_intervals':2},
       'screen_limits':'No mesh-independence, timestep-independence or published-profile accuracy claim follows from this screen.'}
    (w/'M3_RAW_PAPERBOX_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('M3_PAPERBOX_PREFLIGHT=PASS')
    print(f'CASE=RAW_ONLY L={L} Nv=48 Nrho=48 M_omega=6 steps=20500')
    print(f'RH_UPSTREAM_U={u0*u1[0]:.9f} PAPER_PRINTED_U=2639.19')
    print('ENDPOINT_QUADRATURE_IS_NOT_A_COLLISION_EQUILIBRIUM_TEST')


def report(w,rc):
    c=config(w/'M3_RAW_PAPERBOX.ini'); rho0,T0,R,u0=scales(c)
    v,cw,L=vgrid(c); v2=np.sum(v*v,axis=0); phi=np.vstack([v[0],v[0]**2,.5*v[0]*v2])
    Jref=eq_flux(*eq_state(c,'soln-bcs-left'))
    shell=np.max(np.abs(v),axis=0)>.9*L
    with h5py.File(w/'mesh.frfsm','r') as h:sp=np.asarray(h['spt_line_p0'],float)
    rows=[]; prev=None; prev_t=None; final=None; errors=[]
    files=sorted(w.glob('dist_M3_RAW_PAPERBOX-*.frfss'),key=lambda p:float(p.stem.split('-')[-1]))
    for p in files:
        try:
            t=float(p.stem.split('-')[-1])
            with h5py.File(p,'r') as h:F=np.asarray(h['soln_line_p0'],float)
            if F.shape!=(3,48**3,8):raise ValueError(f'Unexpected state shape {F.shape}')
            if not np.isfinite(F).all():raise ValueError('Non-finite distribution')
            J=cw*np.einsum('kv,uve->kue',phi,F,optimize=True)
            row={'time':t,'min_f':float(F.min()),
                'negative_mass_fraction':float(np.max(np.sum(np.maximum(-F,0),axis=1)/np.maximum(np.sum(np.maximum(F,0),axis=1),1e-300))),
                'outer_mass_fraction':float(np.max(np.sum(abs(F[:,shell,:]),axis=1)/np.maximum(np.sum(abs(F),axis=1),1e-300))),
                'f_rel_drift_per_unit_time':None if prev is None else float(np.linalg.norm((F-prev).ravel())/max(np.linalg.norm(prev.ravel()),1e-300)/(t-prev_t))}
            for k,name in enumerate(NAMES):
                row[name+'_span']=float(np.ptp(J[k])/abs(Jref[k]))
                row[name+'_max_reference_deviation']=float(np.max(abs(J[k]-Jref[k]))/abs(Jref[k]))
            rows.append(row); prev=F; prev_t=t; final=(F,J)
        except Exception as exc:errors.append({'file':p.name,'error':str(exc)})
    criteria=json.loads((w/'M3_RAW_PAPERBOX_MANIFEST.json').read_text())['numerical_screen']
    end_reached=bool(rows and abs(rows[-1]['time']-10.25)<1e-8)
    def passes(r):
        return (r['f_rel_drift_per_unit_time'] is not None and r['f_rel_drift_per_unit_time']<criteria['snapshot_drift_per_unit_time'] and
                max(r[n+'_span'] for n in NAMES)<criteria['flux_span'] and
                r['negative_mass_fraction']<criteria['negative_mass_fraction'] and r['outer_mass_fraction']<criteria['outer_mass_fraction'])
    screen=bool(rc==0 and end_reached and not errors and len(rows)>=3 and all(passes(r) for r in rows[-2:]))
    result={'solver_exit_code':rc,'end_time_reached':end_reached,'rows':rows,'read_errors':errors,
            'stationarity_screen':screen,'validation_status':'NOT_VALIDATED','workdir':str(w),'thresholds':criteria}
    (w/'M3_RAW_PAPERBOX_SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if rows:
        with (w/'M3_RAW_PAPERBOX_FLUX_HISTORY.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if final is not None:
        F,J=final; rho=cw*np.sum(F,axis=1); U=cw*np.einsum('kv,uve->kue',v,F,optimize=True)/rho[None,:,:]
        vv=cw*np.einsum('v,uve->ue',v2,F,optimize=True)/rho
        T=2/3*(vv-np.sum(U*U,axis=0))
        x=sp[:,:,0].min(axis=0)[None,:]+(.5*(np.array([-1.,0.,1.])+1))[:,None]*np.ptp(sp[:,:,0],axis=0)[None,:]
        with (w/'M3_RAW_PAPERBOX_PROFILE.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['element','point','x_over_H0','rho_kg_m3','ux_m_s','T_K','qx_W_m2','Pdev_xx_Pa','Jm','Jp','JE'])
            for e in range(8):
                for uidx in range(3):
                    cc=v-U[:,uidx,e,None];cc2=np.sum(cc*cc,axis=0); fv=F[uidx,:,e]
                    qx=.5*cw*np.dot(cc[0]*cc2,fv)*rho0*u0**3
                    pdev=cw*np.dot(cc[0]**2-cc2/3,fv)*rho0*u0**2
                    writer.writerow([e,uidx,x[uidx,e],rho[uidx,e]*rho0,U[0,uidx,e]*u0,T[uidx,e]*T0,qx,pdev,*J[:,uidx,e]])
    lines=['# Mach-3 raw paper-box baseline','',
        'Same RH physical states and initialization as the prior campaign. Published L=11, Nv=48, M_omega=6; Nrho=48. No projection.',
        'This is a bounded evolution to t=10.25, not an automatic validation or steady-state claim.','',
        '| t | mass-flux span | momentum-flux span | energy-flux span | f drift / time | negative mass | outer mass |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        drift='n/a' if r['f_rel_drift_per_unit_time'] is None else f"{r['f_rel_drift_per_unit_time']:.6e}"
        lines.append(f"| {r['time']:.3f} | {r['mass_flux_span']:.6e} | {r['momentum_flux_span']:.6e} | {r['energy_flux_span']:.6e} | {drift} | {r['negative_mass_fraction']:.6e} | {r['outer_mass_fraction']:.6e} |")
    lines+=['',f'SOLVER_EXIT_CODE={rc}',f'END_TIME_REACHED={end_reached}',f'STATIONARITY_SCREEN={"PASS" if screen else "NOT_MET"}',
            'SHOCK_VALIDATION=NOT_ESTABLISHED','',
            'The fluxes are direct velocity moments at DG nodes, not face numerical fluxes. Pointwise flux span is a diagnostic, not an exact discrete conservation theorem.',
            'The provisional numerical screen requires two adjacent snapshot intervals with f drift/time <1e-4, each flux span/upstream <1e-3, and negative/outer mass fractions <1e-5.',
            'Published-profile accuracy and spatial/velocity/time-step independence remain separate tests.',f'WORKDIR={w}']
    if errors:lines+=['','Read errors: '+json.dumps(errors)]
    (w/'M3_RAW_PAPERBOX_SUMMARY.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
    with zipfile.ZipFile(w/'M3_RAW_PAPERBOX_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for pat in ('M3_RAW_PAPERBOX*','paperbox_tools.py','bootstrap_used.sh','input_original.ini','mesh.frfsm','kinetic_residual.csv','source_hashes.txt','submission.json'):
            for p in w.glob(pat):
                if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
        if files and rows and not errors:z.write(files[-1],files[-1].name)
        if (w/'solver.log').is_file():
            with (w/'solver.log').open('rb') as f:
                f.seek(max(0,(w/'solver.log').stat().st_size-32000));z.writestr('solver_tail.txt',f.read())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','report']);ap.add_argument('work',type=Path);ap.add_argument('--rc',type=int,default=0);a=ap.parse_args()
    if a.action=='prepare':prepare(a.work)
    else:report(a.work,a.rc)
PY
PYTHONNOUSERSITE=1 "$PY" "$WORK/paperbox_tools.py" prepare "$WORK"
(
    cd "$SRC"
    find frfs -type f \( -name '*.py' -o -name '*.mako' \) -print0 | sort -z | xargs -0 sha256sum
) > "$WORK/source_hashes.txt"
# Freeze the actual solver source used, including local patches; omit Git/cache.
mkdir "$WORK/src"
tar -C "$SRC" --exclude=.git --exclude=__pycache__ --exclude='*.pyc' -cf - frfs | tar -C "$WORK/src" -xf -
{
    echo '#!/usr/bin/env bash'
    echo '#SBATCH --partition=gpu'
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --ntasks=1'
    echo '#SBATCH --cpus-per-task=4'
    echo '#SBATCH --gpus=1'
    echo '#SBATCH --constraint=v100&x86_64'
    echo '#SBATCH --mem=24G'
    echo '#SBATCH --time=12:00:00'
    echo "#SBATCH --job-name=$JOBNAME"
    echo "#SBATCH --output=$WORK/slurm-%j.out"
    printf 'WORK=%q\nOUT=%q\nPY=%q\n' "$WORK" "$OUT" "$PY"
    cat <<'SLURM'
set -Eeuo pipefail
trap 'rc=$?; echo "M3_RAW_PAPERBOX_JOB_FAILED rc=$rc line=$LINENO" >&2; exit "$rc"' ERR
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6 openmpi/5.0.3-cuda12.6 conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(dirname "$(dirname "$PY")")"
unset PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$WORK/src"
export CUDA_CACHE_PATH="$WORK/cuda-cache"
mkdir -p "$CUDA_CACHE_PATH"
cd "$WORK"
echo 'M3_RAW_PAPERBOX_START'
nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" -c 'import frfs; print("SOLVER_MODULE",frfs.__file__)'
# A failed run is retained and reported; scientific acceptance never depends
# on the solver process merely returning zero.
set +e
"$PY" -m frfs run mesh.frfsm M3_RAW_PAPERBOX.ini -b cuda > solver.log 2>&1
RC=$?
set -e
"$PY" paperbox_tools.py report "$WORK" --rc "$RC"
for p in M3_RAW_PAPERBOX_SUMMARY.md M3_RAW_PAPERBOX_SUMMARY.json M3_RAW_PAPERBOX_FLUX_HISTORY.csv M3_RAW_PAPERBOX_PROFILE.csv M3_RAW_PAPERBOX_RESULTS.zip; do
    [[ -f "$WORK/$p" ]] || continue
    if [[ "$WORK" != "$OUT" ]]; then
        cp -f "$WORK/$p" "$OUT/$p" || echo "COPY_FAILED_RESULT_REMAINS=$WORK/$p"
    fi
done
echo "M3_RAW_PAPERBOX_OUTPUT=$OUT/M3_RAW_PAPERBOX_SUMMARY.md"
echo "FULL_OUTPUT_RETAINED=$WORK"
exit "$RC"
SLURM
} > "$WORK/run.slurm"
bash -n "$WORK/run.slurm"
if [[ ${DGFS_PREPARE_ONLY:-0} == 1 ]]; then
    echo "PREPARED_ONLY=$WORK"
    echo 'NO_JOB_SUBMITTED'
    exit 0
fi
JOB=$(sbatch --parsable "$WORK/run.slurm"); JOB=${JOB%%;*}
printf '%s\n' "$WORK" > "$OUT/M3_RAW_PAPERBOX_LAST_RUN.txt"
"$PY" - "$WORK" "$JOB" <<'PY'
import json,sys,pathlib,datetime
w=pathlib.Path(sys.argv[1]); d={'job_id':sys.argv[2],'work':str(w),'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(w/'submission.json').write_text(json.dumps(d,indent=2)+'\n')
PY
echo "M3_RAW_PAPERBOX_JOB=$JOB"
echo "M3_RAW_PAPERBOX_WORK=$WORK"
echo "SUMMARY=$OUT/M3_RAW_PAPERBOX_SUMMARY.md"
echo 'ONE_RAW_RUN_ONLY; NO_AUTOMATIC_CONTINUATION_OR_WEIGHT_SWEEP'
