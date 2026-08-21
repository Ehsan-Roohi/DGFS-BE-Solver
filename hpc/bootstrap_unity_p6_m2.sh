#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "P6_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV=${DGFS_ENV:-$ROOT/dgfs_py310}
DVM=${DVM_M2_REFERENCE:-/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep/standing_M2_hmom_x40_nx1600_v97_19_19_vmax12_fullstate.npz}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/phase6-dvm-m2-crosscheck}
PY=${DVM_PYTHON:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}
RUN="$ROOT/p6_m2_$(date +%Y%m%d_%H%M%S)"
test -s "$DVM"; test -x "$ENV/bin/python"; test -x "$PY"
mkdir -p "$RUN"
git clone --depth 1 --branch "$REF" "$REPO" "$RUN/src"
"$PY" "$RUN/src/cases/dvm_m2_crosscheck/build_m2_case.py" --dvm "$DVM" --out-dir "$RUN" --elements 32
cp "$RUN/src/hpc/p6_m2_smoke.slurm" "$RUN/run.slurm"
JOB=$(cd "$RUN" && sbatch --parsable --export=ALL,DGFS_ENV="$ENV",DGFS_SOLVER_SRC="$RUN/src" ./run.slurm)
echo "P6_JOB_ID=${JOB%%;*}"
echo "P6_RUN_DIR=$RUN"
echo "P6_DVM_REFERENCE=$DVM"
echo P6_BOOTSTRAP_COMPLETE
