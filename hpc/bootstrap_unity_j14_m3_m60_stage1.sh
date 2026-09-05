#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_M60_STAGE1_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
SMOKE=${DGFS_M3_M60_SMOKE_WORK:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/dgfs_m3_m60_smoke_20260904_204530}
SCRATCH_BASE=${DGFS_M3_M60_SCRATCH:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$SCRATCH_BASE/dgfs_m3_m60_stage1_$STAMP
T0=${DGFS_M3_M60_STAGE1_START:-0.25}
T1=${DGFS_M3_M60_STAGE1_END:-1.25}
mkdir -p "$WORK" "$OUT"

for p in "$ENV_DIR/bin/python" "$SRC" "$SMOKE"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

NAMES=(M3_M60_raw M3_M60_fplus)
for NAME in "${NAMES[@]}"; do
    CASE0="$SMOKE/$NAME"
    CFG0="$CASE0/$NAME.ini"
    MESH0="$CASE0/mesh.frfsm"
    SNAP0="$CASE0/dist_${NAME}-$(printf '%.2f' "$T0").frfss"
    for p in "$CFG0" "$MESH0" "$SNAP0"; do [[ -s "$p" ]] || { echo "MISSING_INPUT=$p"; exit 3; }; done

    "$ENV_DIR/bin/python" - "$CFG0" "$WORK/$NAME.ini" "$T0" "$T1" "$NAME" <<'PY'
import configparser,sys
src,dst,t0,t1,name=sys.argv[1:]
c=configparser.ConfigParser(inline_comment_prefixes=(';','#')); c.optionxform=str
with open(src) as f: c.read_file(f)
c.set('solver-time-integrator','tstart',t0)
c.set('solver-time-integrator','tend',t1)
c.set('solver-time-integrator','dt','0.001')
c.set('spherical-design-rule','M','60')
c.set('constants','Nv','40'); c.set('constants','Nrho','40')
c.set('velocity-mesh','cmax','0'); c.set('velocity-mesh','Tmax',c.get('non-dim','T0')); c.set('velocity-mesh','dev','8.75')
c.set('soln-plugin-dgfsdistwriterstd','dt-out',str(float(t1)-float(t0)))
c.set('soln-plugin-dgfsdistwriterstd','basedir','.')
c.set('soln-plugin-dgfsdistwriterstd','basename',f'dist_{name}-{{t:.2f}}')
c.set('soln-plugin-dgfsmomwriterstd','dt-out','0.25')
c.set('soln-plugin-dgfsmomwriterstd','basedir','.')
c.set('soln-plugin-dgfsmomwriterstd','basename',f'bulk_{name}-{{t:.2f}}')
c.set('soln-plugin-dgfsresidualstd','file','kinetic_residual_stage1.csv')
c.remove_option('soln-plugin-dgfsresidualstd','normalisation-resid')
with open(dst,'w') as f: c.write(f)
PY
done

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=dgfs-m3m60-s1
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_M60_STAGE1_RUN_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; SMOKE=${DGFS_M3_M60_SMOKE_WORK:?}; T0=${DGFS_T0:?}; T1=${DGFS_T1:?}
NAMES=(M3_M60_raw M3_M60_fplus); NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CASE=$WORK/$NAME; SRC0=$SMOKE/$NAME
mkdir -p "$CASE"
cp "$WORK/$NAME.ini" "$CASE/$NAME.ini"
cp "$SRC0/mesh.frfsm" "$CASE/mesh.frfsm"
T0TAG=$(python3 -c 'import sys; print(f"{float(sys.argv[1]):.2f}")' "$T0")
ln -sf "$SRC0/dist_${NAME}-$T0TAG.frfss" "$CASE/restart.frfss"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load cuda/12.6; module load openmpi/5.0.3-cuda12.6; module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" CUDA_CACHE_PATH="$WORK/.cuda-cache-$SLURM_ARRAY_TASK_ID"; mkdir -p "$CUDA_CACHE_PATH"
echo "M3_M60_STAGE1_START name=$NAME node=${SLURMD_NODENAME:-unknown} interval=$T0,$T1"
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$CASE"
"$ENV_DIR/bin/python" -m frfs restart mesh.frfsm restart.frfss "$NAME.ini" -b cuda 2>&1 | tee solver.log
T1TAG=$("$ENV_DIR/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):.2f}")' "$T1")
FINAL="$CASE/dist_${NAME}-$T1TAG.frfss"
test -s "$FINAL"; test -s "$CASE/kinetic_residual_stage1.csv"; touch "$CASE/RUN_SUCCESS"
rm -rf "$CUDA_CACHE_PATH" || true
echo "M3_M60_STAGE1_DONE name=$NAME final=$FINAL"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m3m60-s1-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --job-name=dgfs-m3m60-s1agg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; WORK=${DGFS_WORK:?}; ENV_DIR=${DGFS_ENV:?}; T1=${DGFS_T1:?}
"$ENV_DIR/bin/python" - "$WORK" "$OUT" "$T1" <<'PY'
import configparser,csv,h5py,math,numpy as np,pathlib,sys
work=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); t1=float(sys.argv[3]); tag=f'{t1:.2f}'
names=['M3_M60_raw','M3_M60_fplus']
lines=['# Mach-3 M_omega=60 raw/fplus continuation to t='+tag,'',
       'Both arms restart from their t=0.25 smoke-test states. Same L=8.75, Nv=40, Nrho=40, M_omega=60.','',
       '| mode | min(f) | max negative mass fraction | max outer-shell abs-mass fraction | final residual raw | finite |',
       '|---|---:|---:|---:|---:|---|']
rows={}
for name in names:
    case=work/name; assert (case/'RUN_SUCCESS').exists()
    cp=configparser.ConfigParser(inline_comment_prefixes=(';','#')); cp.optionxform=str; cp.read(case/f'{name}.ini')
    Nv=cp.getint('constants','Nv'); L=cp.getfloat('velocity-mesh','dev')
    c0=np.linspace(-L+L/Nv,L-L/Nv,Nv); X,Y,Z=np.meshgrid(c0,c0,c0,indexing='ij')
    outer=(np.maximum.reduce([abs(X),abs(Y),abs(Z)]).ravel()>0.9*L)
    with h5py.File(case/f'dist_{name}-{tag}.frfss','r') as h: F=np.asarray(h['soln_line_p0'],float)
    mn=float(np.min(F)); negmax=0.; outermax=0.
    for e in range(F.shape[2]):
      for u in range(F.shape[0]):
        f=F[u,:,e]; pos=np.maximum(f,0); neg=np.maximum(-f,0)
        negmax=max(negmax,float(np.sum(neg)/max(np.sum(pos),1e-300)))
        outermax=max(outermax,float(np.sum(abs(f[outer]))/max(np.sum(abs(f)),1e-300)))
    with (case/'kinetic_residual_stage1.csv').open(newline='') as fh: rr=list(csv.DictReader(fh)); fres=float(rr[-1]['f'])
    finite=bool(np.isfinite(F).all() and math.isfinite(fres))
    rows[name]=(mn,negmax,outermax,fres,finite)
    lines.append(f'| {name} | {mn:.4e} | {negmax:.4e} | {outermax:.4e} | {fres:.4e} | {finite} |')
r=rows['M3_M60_raw']; f=rows['M3_M60_fplus']
neg_ratio=f[1]/max(r[1],1e-300); outer_ratio=f[2]/max(r[2],1e-300)
raw_ok=r[4] and r[1]<1e-3 and r[2]<1e-4
fplus_ok=f[4] and f[1]<1e-3 and f[2]<1e-4 and neg_ratio<5 and outer_ratio<5
lines += ['',f'M3_M60_STAGE1_RAW_GATE={"PASS" if raw_ok else "FAIL"}',
          f'M3_M60_STAGE1_FPLUS_GATE={"PASS" if fplus_ok else "FAIL"}',
          f'M3_M60_STAGE1_FPLUS_NEGATIVE_MASS_RATIO_VS_RAW={neg_ratio:.6e}',
          f'M3_M60_STAGE1_FPLUS_OUTER_MASS_RATIO_VS_RAW={outer_ratio:.6e}',
          f'M3_M60_STAGE1_WORK={work}']
(out/'M3_M60_STAGE1_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "M3_M60_STAGE1_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3m60-s1agg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_M3_M60_SMOKE_WORK="$SMOKE",DGFS_T0="$T0",DGFS_T1="$T1" "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_T1="$T1" "$WORK/aggregate.slurm")
AGG=${AGG%%;*}
echo "M3_M60_STAGE1_ARRAY_JOB=$ARRAY"
echo "M3_M60_STAGE1_AGG_JOB=$AGG"
echo "M3_M60_STAGE1_WORK=$WORK"
echo "M3_M60_STAGE1_SUMMARY=$OUT/M3_M60_STAGE1_SUMMARY.md"
echo "M3_M60_STAGE1_BOOTSTRAP_COMPLETE"
