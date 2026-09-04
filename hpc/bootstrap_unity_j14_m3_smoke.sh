#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_SMOKE_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
BASE=$CLOSE/final_runs/run_M6_raw/p3b_M6_raw.ini
MESH=$CLOSE/final_runs/run_M6_raw/mesh.frfsm
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/m3_smoke_$STAMP
mkdir -p "$WORK" "$OUT"
for p in "$ENV_DIR/bin/python" "$BASE" "$MESH" "$SRC"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

# Construct Mach-3 helium Rankine-Hugoniot states and three identical-start configs.
"$ENV_DIR/bin/python" - "$BASE" "$WORK" <<'PY'
import configparser, math, pathlib, sys
base=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
c=configparser.ConfigParser(inline_comment_prefixes=(';','#')); c.optionxform=str
with base.open() as f: c.read_file(f)
R=8.3144598; mm=c.getfloat('non-dim','molarMass0'); T1=c.getfloat('non-dim','T0'); rho1=c.getfloat('non-dim','rho0')
g=5/3; M1=3.0; Rsp=R/mm; a1=math.sqrt(g*Rsp*T1); u1=M1*a1
rr=((g+1)*M1*M1)/((g-1)*M1*M1+2); pr=1+2*g/(g+1)*(M1*M1-1); Tr=pr/rr
rho2=rho1*rr; T2=T1*Tr; u2=u1/rr
# Keep the original dv=0.4375 but enlarge L from 7 to 8.75: Nv=40, Nrho=40.
c.set('constants','Nv','40'); c.set('constants','Nrho','40'); c.set('constants','NvBatchSize','64')
c.set('velocity-mesh','cmax','0'); c.set('velocity-mesh','Tmax',f'{T1:.17g}'); c.set('velocity-mesh','dev','8.75')
c.set('spherical-design-rule','M','6')
c.set('solver-time-integrator','tstart','0'); c.set('solver-time-integrator','tend','0.25'); c.set('solver-time-integrator','dt','0.001')
# Same linear RH initialization pattern as the validated M1.59 case.
c.set('soln-ics','rho',f'{rho1:.17g} + ({rho2:.17g}-{rho1:.17g})*(x + 0.5)')
c.set('soln-ics','T',f'{T1:.17g} + ({T2:.17g}-{T1:.17g})*(x + 0.5)')
c.set('soln-ics','ux',f'{u1:.17g} + ({u2:.17g}-{u1:.17g})*(x + 0.5)')
for s,r,T,u in [('soln-bcs-left',rho1,T1,u1),('soln-bcs-right',rho2,T2,u2)]:
    c.set(s,'rho',f'{r:.17g}'); c.set(s,'T',f'{T:.17g}'); c.set(s,'ux',f'{u:.17g}'); c.set(s,'uy','0'); c.set(s,'uz','0')
print(f'M3_RH rho1={rho1:.9e} T1={T1:.9f} u1={u1:.9f}')
print(f'M3_RH rho2={rho2:.9e} T2={T2:.9f} u2={u2:.9f}')
print(f'M3_RATIOS rho={rr:.9f} p={pr:.9f} T={Tr:.9f}')
u0=math.sqrt(2*Rsp*T1); L=8.75
# Gaussian probability mass outside [-L,L]^3 for each endpoint Maxwellian.
def Phi(z): return 0.5*(1+math.erf(z/math.sqrt(2)))
def outside(mu,sig):
    px=Phi((L-mu)/sig)-Phi((-L-mu)/sig); py=Phi(L/sig)-Phi(-L/sig)
    return 1-px*py*py
print(f'M3_VELOCITY_BOX L={L} Nv=40 dv={2*L/40:.8f}')
print(f'M3_ENDPOINT_OUTSIDE_MASS upstream={outside(u1/u0,math.sqrt(0.5)):.3e} downstream={outside(u2/u0,math.sqrt(Tr/2)):.3e}')
for name,proj in [('M3_raw','none'),('M3_euclidean','euclidean'),('M3_fplus','fplus')]:
    cc=configparser.ConfigParser(inline_comment_prefixes=(';','#')); cc.optionxform=str
    for sec in c.sections():
        cc.add_section(sec)
        for k,v in c.items(sec): cc.set(sec,k,v)
    cc.set('scattering-model','projection',proj); cc.set('scattering-model','projection-solve','device')
    cc.set('soln-plugin-dgfsdistwriterstd','dt-out','0.25'); cc.set('soln-plugin-dgfsdistwriterstd','basedir','.')
    cc.set('soln-plugin-dgfsdistwriterstd','basename',f'dist_{name}-{{t:.2f}}')
    cc.set('soln-plugin-dgfsmomwriterstd','dt-out','0.05'); cc.set('soln-plugin-dgfsmomwriterstd','basedir','.')
    cc.set('soln-plugin-dgfsmomwriterstd','basename',f'bulk_{name}-{{t:.2f}}')
    cc.set('soln-plugin-dgfsresidualstd','file','kinetic_residual.csv'); cc.remove_option('soln-plugin-dgfsresidualstd','normalisation-resid')
    with (out/f'{name}.ini').open('w') as f: cc.write(f)
PY

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --array=0-2
#SBATCH --job-name=dgfs-m3-smoke
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_SMOKE_RUN_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; MESH=${DGFS_MESH:?}
NAMES=(M3_raw M3_euclidean M3_fplus); NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}; CASE=$WORK/$NAME
mkdir -p "$CASE"; cp "$WORK/$NAME.ini" "$CASE/$NAME.ini"; cp "$MESH" "$CASE/mesh.frfsm"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load cuda/12.6; module load openmpi/5.0.3-cuda12.6; module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" CUDA_CACHE_PATH="$WORK/.cuda-cache-$SLURM_ARRAY_TASK_ID"; mkdir -p "$CUDA_CACHE_PATH"
echo "M3_SMOKE_START name=$NAME node=${SLURMD_NODENAME:-unknown}"; nvidia-smi --query-gpu=name --format=csv,noheader
cd "$CASE"; "$ENV_DIR/bin/python" -m frfs run mesh.frfsm "$NAME.ini" -b cuda 2>&1 | tee solver.log
FINAL="$CASE/dist_${NAME}-0.25.frfss"; test -s "$FINAL"; test -s "$CASE/kinetic_residual.csv"; touch "$CASE/RUN_SUCCESS"
rm -rf "$CUDA_CACHE_PATH" || true; echo "M3_SMOKE_DONE name=$NAME final=$FINAL"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m3-smoke-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-m3-smkagg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; WORK=${DGFS_WORK:?}; ENV_DIR=${DGFS_ENV:?}
"$ENV_DIR/bin/python" - "$WORK" "$OUT" <<'PY'
import configparser,csv,h5py,math,numpy as np,pathlib,sys
work=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); names=['M3_raw','M3_euclidean','M3_fplus']
lines=['# Mach-3 helium shock smoke test','', '| mode | min(f) | max negative mass fraction | max outer-shell abs-mass fraction | final residual raw |','|---|---:|---:|---:|---:|']
PASS=True
for name in names:
    case=work/name; assert (case/'RUN_SUCCESS').exists()
    cp=configparser.ConfigParser(inline_comment_prefixes=(';','#')); cp.optionxform=str; cp.read(case/f'{name}.ini')
    Nv=cp.getint('constants','Nv'); L=cp.getfloat('velocity-mesh','dev'); cw=(2*L/Nv)**3
    c0=np.linspace(-L+L/Nv,L-L/Nv,Nv); X,Y,Z=np.meshgrid(c0,c0,c0,indexing='ij'); outer=(np.maximum.reduce([abs(X),abs(Y),abs(Z)]).ravel()>0.9*L)
    with h5py.File(case/f'dist_{name}-0.25.frfss','r') as h: F=np.asarray(h['soln_line_p0'],float)
    mn=float(np.min(F)); negmax=0.; outermax=0.
    for e in range(F.shape[2]):
      for u in range(F.shape[0]):
        f=F[u,:,e]; pos=np.maximum(f,0); neg=np.maximum(-f,0); negmax=max(negmax,float(np.sum(neg)/max(np.sum(pos),1e-300))); outermax=max(outermax,float(np.sum(abs(f[outer]))/max(np.sum(abs(f)),1e-300)))
    with (case/'kinetic_residual.csv').open(newline='') as fh:
      rows=list(csv.DictReader(fh)); fres=float(rows[-1]['f'])
    finite=np.isfinite(F).all() and math.isfinite(fres); ok=finite and negmax<1e-2 and outermax<1e-5; PASS &= ok
    lines.append(f'| {name} | {mn:.4e} | {negmax:.4e} | {outermax:.4e} | {fres:.4e} |')
lines += ['',f'M3_SMOKE_GATE={"PASS" if PASS else "FAIL"}',f'M3_SMOKE_WORK={work}']
(out/'M3_SMOKE_SUMMARY.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if not PASS: raise SystemExit(4)
PY
echo "M3_SMOKE_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3-smoke-agg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_MESH="$MESH" "$WORK/run.slurm"); ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT" "$WORK/aggregate.slurm"); AGG=${AGG%%;*}
echo "M3_SMOKE_ARRAY_JOB=$ARRAY"
echo "M3_SMOKE_AGG_JOB=$AGG"
echo "M3_SMOKE_WORK=$WORK"
echo "M3_SMOKE_SUMMARY=$OUT/M3_SMOKE_SUMMARY.md"
echo "M3_SMOKE_BOOTSTRAP_COMPLETE"
