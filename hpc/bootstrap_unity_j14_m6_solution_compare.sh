#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M6_SOLUTION_COMPARE_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
REF=${DGFS_REF:-42f6e70e984ddb052c6e427c65513def47aeadba}
LAUNCH_DIR=${DGFS_OUTPUT_DIR:-$PWD}
DURATION=${DGFS_SHORT_DURATION:-20.0}
DIST_DT=${DGFS_SHORT_DIST_DT:-5.0}
MOM_DT=${DGFS_SHORT_MOM_DT:-1.0}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/m6_solution_compare_$STAMP
M6=$CLOSE/final_runs/run_M6_raw

mkdir -p "$WORK" "$LAUNCH_DIR"

test -x "$ENV_DIR/bin/python"
test -d "$SRC"
test -s "$M6/p3b_M6_raw.ini"
test -s "$M6/mesh.frfsm"

SOURCE=$({ find "$M6" -maxdepth 2 -type f -name 'dist_p3b_M6_raw-*.frfss' -printf '%p\n' 2>/dev/null || true; } | sort -V | tail -1)
[[ -s "$SOURCE" ]] || { echo "M6_SOURCE_SNAPSHOT_NOT_FOUND"; exit 2; }

read -r T0 T1 < <("$ENV_DIR/bin/python" - "$SOURCE" "$DURATION" <<'PY'
import re,sys
p=sys.argv[1]; dur=float(sys.argv[2])
m=re.search(r'-([0-9]+(?:\.[0-9]+)?)\.frfss$',p)
if not m: raise SystemExit('cannot parse source time')
t=float(m.group(1)); print(f'{t:.10g}',f'{t+dur:.10g}')
PY
)

echo "M6_SOLUTION_COMPARE_SOURCE=$SOURCE"
echo "M6_SOLUTION_COMPARE_INTERVAL=$T0,$T1"
echo "M6_SOLUTION_COMPARE_WORK=$WORK"
echo "M6_SOLUTION_COMPARE_OUTPUT=$LAUNCH_DIR"

FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
NEED_KB=$((20*1024*1024))
(( FREE_KB >= NEED_KB )) || { echo "LOW_SPACE free_kb=$FREE_KB need_kb=$NEED_KB"; exit 3; }

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_solution_state_diagnostics.py" -o "$WORK/j14_solution_state_diagnostics.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_solution_state_diagnostics.py"

"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/make_restart_configs.py" \
  --base "$M6/p3b_M6_raw.ini" \
  --residual-csv "$M6/kinetic_residual_p3b.csv" \
  --tstart "$T0" --tend "$T1" \
  --dist-dt-out "$DIST_DT" --mom-dt-out "$MOM_DT" \
  --residual-file kinetic_residual_short.csv \
  --runs 'M6_short_raw:32:6:none,M6_short_euclidean:32:6:euclidean,M6_short_fplus:32:6:fplus' \
  --out-dir "$WORK"

cat > "$WORK/run_array.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --job-name=dgfs-m6-sol
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M6_SOLUTION_RUN_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:?}
ENV_DIR=${DGFS_ENV:?}
SRC=${DGFS_SOLVER_SRC:?}
WORK=${DGFS_WORK:?}
SOURCE=${DGFS_SOURCE:?}
T1=${DGFS_T1:?}

NAMES=(M6_short_raw M6_short_euclidean M6_short_fplus)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CASE=$WORK/$NAME
mkdir -p "$CASE"
cp "$WORK/p3b_$NAME.ini" "$CASE/p3b_$NAME.ini"
cp "$DGFS_MESH" "$CASE/mesh.frfsm"
ln -sf "$SOURCE" "$CASE/dist-source.frfss"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_CACHE_PATH="$WORK/.cuda-cache-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CUDA_CACHE_PATH"

echo "M6_SOLUTION_RUN_START name=$NAME node=${SLURMD_NODENAME:-unknown} source=$SOURCE"
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$CASE"
"$ENV_DIR/bin/python" -m frfs restart mesh.frfsm dist-source.frfss "p3b_$NAME.ini" -b cuda 2>&1 | tee solver.log

T1_2=$("$ENV_DIR/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):.2f}")' "$T1")
FINAL="$CASE/dist_p3b_$NAME-$T1_2.frfss"
test -s "$FINAL"
touch "$CASE/RUN_SUCCESS"
echo "M6_SOLUTION_RUN_DONE name=$NAME final=$FINAL"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m6-sol-%A_%a.out|" "$WORK/run_array.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dgfs-m6-agg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M6_SOLUTION_AGG_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ENV_DIR=${DGFS_ENV:?}; WORK=${DGFS_WORK:?}; SOURCE=${DGFS_SOURCE:?}; T0=${DGFS_T0:?}; OUT=${DGFS_OUTPUT_DIR:?}; MESH=${DGFS_MESH:?}
cd "$WORK"
NAMES=(M6_short_raw M6_short_euclidean M6_short_fplus)
for NAME in "${NAMES[@]}"; do
  CASE="$WORK/$NAME"
  test -e "$CASE/RUN_SUCCESS"
  mapfile -t SNAPS < <(find "$CASE" -maxdepth 1 -type f -name "dist_p3b_$NAME-*.frfss" -printf '%p\n' | sort -V)
  "$ENV_DIR/bin/python" "$WORK/j14_solution_state_diagnostics.py" \
    --config "$CASE/p3b_$NAME.ini" --mesh "$MESH" --label "$NAME" --t0 "$T0" \
    --snapshots "$SOURCE" "${SNAPS[@]}" --output-prefix "$OUT/$NAME"
done

"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import csv,json,math,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1])
names=['M6_short_raw','M6_short_euclidean','M6_short_fplus']

def readcsv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

global_data={n:readcsv(out/f'{n}.global.csv') for n in names}
point_data={n:readcsv(out/f'{n}.points.csv') for n in names}
lines=['# M6 time-integrated projection comparison','',
       'All three runs restart from the identical existing M6 raw distribution snapshot.','',
       '| mode | final min(f) | max neg-mass frac | rel global mass drift | rel global energy drift | max |uy| | max |uz| |',
       '|---|---:|---:|---:|---:|---:|---:|']
summary={}
for n in names:
    g=global_data[n]; a=g[0]; b=g[-1]
    def rel(k): return (float(b[k])-float(a[k]))/max(abs(float(a[k])),1e-300)
    s={'final_min_f':float(b['min_f']),'final_max_negative_mass_fraction':float(b['max_negative_mass_fraction']),
       'relative_global_mass_drift':rel('global_mass'),'relative_global_energy_drift':rel('global_energy'),
       'final_max_abs_uy':float(b['max_abs_uy']),'final_max_abs_uz':float(b['max_abs_uz'])}
    summary[n]=s
    lines.append(f"| {n} | {s['final_min_f']:.4e} | {s['final_max_negative_mass_fraction']:.4e} | {s['relative_global_mass_drift']:.4e} | {s['relative_global_energy_drift']:.4e} | {s['final_max_abs_uy']:.4e} | {s['final_max_abs_uz']:.4e} |")

# Final-profile differences against the simultaneously evolved raw arm.
def final_rows(rows):
    t=max(float(r['elapsed']) for r in rows); return [r for r in rows if abs(float(r['elapsed'])-t)<1e-8]
raw=final_rows(point_data['M6_short_raw'])
for n in names[1:]:
    rr=final_rows(point_data[n]); lines += ['',f'## {n} vs simultaneous raw at final time']
    for field in ['qx','Pdev_xx','c4','qz','Pxz']:
        x=[float(r[field]) for r in raw]; y=[float(r[field]) for r in rr]
        num=math.sqrt(sum((a-b)**2 for a,b in zip(x,y))); den=math.sqrt(sum(a*a for a in x)) or 1e-300
        lines.append(f'- relative L2 difference in {field}: {num/den:.6e}')

(out/'M6_TIMEINTEGRATED_COMPARE_SUMMARY.md').write_text('\n'.join(lines)+'\n')
(out/'M6_TIMEINTEGRATED_COMPARE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
zip_path=out/'DGFS_M6_TIMEINTEGRATED_COMPARE.zip'; zip_path.unlink(missing_ok=True)
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file() and (p.name.startswith('M6_short_') or p.name.startswith('M6_TIMEINTEGRATED_')):
            z.write(p,p.name)
print('\n'.join(lines)); print('BUNDLE='+str(zip_path))
PY

echo "M6_SOLUTION_COMPARE_COMPLETE"
ls -lh "$OUT/DGFS_M6_TIMEINTEGRATED_COMPARE.zip" "$OUT/M6_TIMEINTEGRATED_COMPARE_SUMMARY.md"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$LAUNCH_DIR/dgfs-m6-agg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable \
  --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_SOURCE="$SOURCE",DGFS_T1="$T1",DGFS_MESH="$M6/mesh.frfsm" \
  "$WORK/run_array.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_SOURCE="$SOURCE",DGFS_T0="$T0",DGFS_OUTPUT_DIR="$LAUNCH_DIR",DGFS_MESH="$M6/mesh.frfsm" \
  "$WORK/aggregate.slurm")
AGG=${AGG%%;*}

echo "M6_SOLUTION_COMPARE_ARRAY_JOB=$ARRAY"
echo "M6_SOLUTION_COMPARE_AGG_JOB=$AGG"
echo "M6_SOLUTION_COMPARE_FINAL_ZIP=$LAUNCH_DIR/DGFS_M6_TIMEINTEGRATED_COMPARE.zip"
echo "M6_SOLUTION_COMPARE_BOOTSTRAP_COMPLETE"
