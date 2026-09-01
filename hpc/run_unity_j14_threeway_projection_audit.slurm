#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=dgfs-3way
#SBATCH --output=dgfs-3way-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "DGFS_THREEWAY_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSEOUT=${DGFS_CLOSEOUT:?DGFS_CLOSEOUT is required}
STEADY=${DGFS_STEADY_DIR:?DGFS_STEADY_DIR is required}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_THREEWAY_OUT:-$PWD}
AUDIT="$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_threeway_projection_audit.py"

mkdir -p "$OUT"
cd "$OUT"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$OUT/.cuda-cache-threeway-${SLURM_JOB_ID:-manual}"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

test -x "$ENV_DIR/bin/python"
test -s "$AUDIT"

echo "===== DGFS THREE-WAY OPERATOR AUDIT ====="
echo "node=${SLURMD_NODENAME:-unknown}"
echo "out=$OUT"
echo "src=$SRC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

latest_dist() {
    local d=$1
    find "$d" -maxdepth 2 -type f -name 'dist_p3b_*.frfss' -printf '%p\n' | sort -V | tail -1
}

run_case() {
    local label=$1 config=$2 mesh=$3 snapshot=$4
    test -s "$config"
    test -s "$mesh"
    test -s "$snapshot"
    echo "===== CASE $label ====="
    echo "config=$config"
    echo "mesh=$mesh"
    echo "snapshot=$snapshot"
    "$ENV_DIR/bin/python" "$AUDIT" \
      --label "$label" \
      --config "$config" \
      --mesh "$mesh" \
      --snapshot "$snapshot" \
      --output-json "THREEWAY_${label}.json" \
      --output-csv "THREEWAY_${label}.csv" \
      --repeats 1
}

M6R="$CLOSEOUT/final_runs/run_M6_raw"
M6P="$CLOSEOUT/final_runs/run_M6_fplus"
M16R="$STEADY/stage_1/M16_raw"
M16P="$STEADY/stage_1/M16_fplus"

run_case M6_raw \
  "$M6R/p3b_M6_raw.ini" "$M6R/mesh.frfsm" "$(latest_dist "$M6R")"
run_case M6_fplus_state \
  "$M6P/p3b_M6_fplus.ini" "$M6P/mesh.frfsm" "$(latest_dist "$M6P")"
run_case M16_raw \
  "$M16R/p3b_M16_raw.ini" "$M16R/mesh.frfsm" "$(latest_dist "$M16R")"
run_case M16_fplus_state \
  "$M16P/p3b_M16_fplus.ini" "$M16P/mesh.frfsm" "$(latest_dist "$M16P")"

"$ENV_DIR/bin/python" - <<'PY'
import json, pathlib, statistics, zipfile
root = pathlib.Path('.')
paths = sorted(root.glob('THREEWAY_*.json'))
reports = [json.loads(p.read_text()) for p in paths]

lines = [
    '# DGFS raw vs unweighted vs fplus-weighted operator audit',
    '',
    'This is post-processing of existing distribution snapshots only; no DGFS time integration was performed.',
    '',
    '| State | M_omega | raw defect | unweighted defect | weighted defect | med tail frac unw | med tail frac wgt | med low-support frac unw | med low-support frac wgt | med cond unw | med cond wgt |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in reports:
    s = r['summary']
    lines.append(
        f"| {s['label']} | {s['M_omega']} | {s['raw_max_defect']:.3e} | "
        f"{s['unweighted_max_defect']:.3e} | {s['weighted_max_defect']:.3e} | "
        f"{s['median_tail_correction_l2_fraction']['unweighted']:.3e} | "
        f"{s['median_tail_correction_l2_fraction']['weighted']:.3e} | "
        f"{s['median_low_support_correction_l2_fraction']['unweighted']:.3e} | "
        f"{s['median_low_support_correction_l2_fraction']['weighted']:.3e} | "
        f"{s['median_scaled_gram_condition']['unweighted']:.3e} | "
        f"{s['median_scaled_gram_condition']['weighted']:.3e} |"
    )

lines += ['', '## High-order collision-moment disturbance', '']
for r in reports:
    s = r['summary']
    lines.append(f"### {s['label']} (M_omega={s['M_omega']})")
    for mom, v in s['median_high_order_relative_disturbance'].items():
        lines.append(f"- {mom}: unweighted={v['unweighted']:.6e}; weighted={v['weighted']:.6e}")
    lines.append(
        '- weighted-smaller fractions: ' +
        ', '.join(f"{k}={v:.3f}" for k,v in s['weighted_smaller_fraction'].items())
    )
    lines.append('')

lines += [
    '## Interpretation rule',
    '',
    '- Both projections must reduce invariant defects to roundoff.',
    '- Plain Euclidean correction L2 is expected to favor the unweighted projection by construction.',
    '- Evidence for the weighted method must instead appear as reduced tail/low-support contamination, reduced high-order-moment disturbance, or another physically defensible support-aware diagnostic without pathological conditioning.',
]
(root/'THREEWAY_OPERATOR_AUDIT_SUMMARY.md').write_text('\n'.join(lines)+'\n')

bundle = root/'DGFS_THREEWAY_OPERATOR_AUDIT.zip'
bundle.unlink(missing_ok=True)
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.glob('THREEWAY_*')):
        if p.is_file(): z.write(p, p.name)
    z.write(root/'THREEWAY_OPERATOR_AUDIT_SUMMARY.md', 'THREEWAY_OPERATOR_AUDIT_SUMMARY.md')
print('\n'.join(lines))
print(f'BUNDLE={bundle.resolve()}')
PY

rm -rf "$CUDA_CACHE_PATH" || true

echo "===== COMPLETE ====="
ls -lh THREEWAY_*.json THREEWAY_*.csv THREEWAY_OPERATOR_AUDIT_SUMMARY.md DGFS_THREEWAY_OPERATOR_AUDIT.zip
echo DGFS_THREEWAY_JOB_COMPLETE
