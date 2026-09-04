#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_GEOM_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-c13fa1cb10e796cb434f6df605f92090fe4fa50d}
STAGE1=${DGFS_M3_STAGE1_WORK:?set DGFS_M3_STAGE1_WORK}
STAMP=$(date +%Y%m%d_%H%M%S); WORK=$CLOSE/m3_geom_$STAMP; mkdir -p "$WORK" "$OUT"
SCRIPT=$WORK/j14_m3_geometric_weight_sweep.py
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_m3_geometric_weight_sweep.py" -o "$SCRIPT"
"$ENV_DIR/bin/python" -m py_compile "$SCRIPT"
for N in M3_raw M3_fplus; do
  [[ -s "$STAGE1/$N/$N.ini" && -s "$STAGE1/$N/mesh.frfsm" && -s "$STAGE1/$N/dist_$N-10.25.frfss" ]] || { echo "MISSING_STAGE1_$N"; exit 2; }
done
cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --job-name=dgfs-m3-geom
#SBATCH --output=LOG_PLACEHOLDER
set -Eeuo pipefail
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; STAGE1=${DGFS_M3_STAGE1_WORK:?}
NAMES=(M3_raw M3_fplus); N=${NAMES[$SLURM_ARRAY_TASK_ID]}
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load cuda/12.6; module load openmpi/5.0.3-cuda12.6; module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" CUDA_CACHE_PATH="$WORK/.cuda-$SLURM_ARRAY_TASK_ID"; mkdir -p "$CUDA_CACHE_PATH"
"$ENV_DIR/bin/python" "$WORK/j14_m3_geometric_weight_sweep.py" --label "$N" --config "$STAGE1/$N/$N.ini" --mesh "$STAGE1/$N/mesh.frfsm" --snapshot "$STAGE1/$N/dist_$N-10.25.frfss" --output-json "$OUT/M3_GEOM_${N}.json"
rm -rf "$CUDA_CACHE_PATH" || true
SLURM
sed -i "s|LOG_PLACEHOLDER|$OUT/dgfs-m3-geom-%A_%a.out|" "$WORK/run.slurm"
cat > "$WORK/agg.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=dgfs-m3-gagg
#SBATCH --output=AGG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}
"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import json,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1]); names=['M3_raw','M3_fplus']; js={n:json.load(open(out/f'M3_GEOM_{n}.json')) for n in names}
lines=['# Mach-3 geometric weighting sweep','','Weight family: w_theta = M[f]^(1-theta) (f+)^theta.  theta=0 is Maxwellian; theta=1 is fplus.','']
for n in names:
 s=js[n]['summary']; lines += [f'## Input state: {n}','',f"- raw max invariant defect: {s['raw_max_inv_defect']:.6e}",'','| theta | max inv defect | corr L2 | Gram cond | tail frac | low-support frac | neg-node frac | qx | qz | Pdev_xx | Pxz | c4 |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for th in ['0.00','0.25','0.50','0.75','1.00']:
  r=s['theta'][th]; m=r['median_mom']; lines.append(f"| {th} | {r['max_inv_defect']:.3e} | {r['median_corr_L2']:.3e} | {r['median_cond']:.2f} | {r['median_tail']:.3e} | {r['median_low']:.3e} | {r['median_negative']:.3e} | {m['qx']:.3e} | {m['qz']:.3e} | {m['Pdev_xx']:.3e} | {m['Pxz']:.3e} | {m['c4']:.3e} |")
 lines += ['']
(out/'M3_GEOMETRIC_SWEEP_SUMMARY.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(out/'DGFS_M3_GEOMETRIC_SWEEP.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in names: z.write(out/f'M3_GEOM_{n}.json',f'M3_GEOM_{n}.json')
 z.write(out/'M3_GEOMETRIC_SWEEP_SUMMARY.md','M3_GEOMETRIC_SWEEP_SUMMARY.md')
print('\n'.join(lines))
PY
SLURM
sed -i "s|AGG_PLACEHOLDER|$OUT/dgfs-m3-gagg-%j.out|" "$WORK/agg.slurm"
A=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_M3_STAGE1_WORK="$STAGE1" "$WORK/run.slurm"); A=${A%%;*}
G=$(sbatch --parsable --dependency=afterok:$A --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_OUTPUT_DIR="$OUT" "$WORK/agg.slurm"); G=${G%%;*}
echo "M3_GEOM_ARRAY_JOB=$A"; echo "M3_GEOM_AGG_JOB=$G"; echo "M3_GEOM_SUMMARY=$OUT/M3_GEOMETRIC_SWEEP_SUMMARY.md"; echo "M3_GEOM_BOOTSTRAP_COMPLETE"
