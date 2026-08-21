#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "P4E_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/phase3-angular-conservative-audit}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DGFS_ROOT/p4e_$STAMP"
SOLVER_SRC="$RUN_DIR/src"
test -x "$DGFS_ENV/bin/python"

if [[ -n "${DGFS_P4A_ROOT:-}" ]]; then
  BASELINE=$DGFS_P4A_ROOT
else
  BASELINE=""
  while IFS= read -r candidate; do
    if [[ -s "$candidate/dist_dgfs_fig14b-0.0.frfss" \
       && -s "$candidate/run_M16_raw/dist_p3b_M16_raw-1.00.frfss" \
       && -s "$candidate/run_M16_fplus/dist_p3b_M16_fplus-1.00.frfss" \
       && -s "$candidate/run_M24_raw/dist_p3b_M24_raw-1.00.frfss" ]]; then
      BASELINE=$candidate; break
    fi
  done < <(find "$DGFS_ROOT" -maxdepth 1 -type d -name 'p4a_*' -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
fi
[[ -d "$BASELINE" ]] || { echo P4E_VALID_P4A_BASELINE_NOT_FOUND; exit 2; }
test -s "$BASELINE/mesh.frfsm"; test -s "$BASELINE/dgfs_fig14b.ini"

mkdir -p "$RUN_DIR"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"
PKG="$SOLVER_SRC/cases/jcp2019_fig14b_normal_shock/phase3"
cp "$BASELINE/dist_dgfs_fig14b-0.0.frfss" "$RUN_DIR/"
cp "$BASELINE/mesh.frfsm" "$RUN_DIR/"
cp "$BASELINE/dgfs_fig14b.ini" "$RUN_DIR/"
if [[ -s "$BASELINE/kinetic_residual.csv" ]]; then
  cp "$BASELINE/kinetic_residual.csv" "$RUN_DIR/"
else
  printf 't,f,f_normalized\n' > "$RUN_DIR/kinetic_residual.csv"
fi
cp -r "$PKG" "$RUN_DIR/p3"
cp -r "$SOLVER_SRC/cases/jcp2019_fig14b_normal_shock/solver_hook" "$RUN_DIR/solver_hook"
cp "$SOLVER_SRC/hpc/p4e_transverse_time_history.slurm" "$RUN_DIR/run.slurm"
SOURCE_COMMIT=$(git -C "$SOLVER_SRC" rev-parse HEAD)
SOURCE_SHA=$(sha256sum "$BASELINE/dist_dgfs_fig14b-0.0.frfss" | awk '{print $1}')
printf 'baseline_root=%s\nsource_ref=%s\nsource_commit=%s\nt0_sha256=%s\npurpose=transverse_projection_time_history\n' \
  "$BASELINE" "$DGFS_REF" "$SOURCE_COMMIT" "$SOURCE_SHA" > "$RUN_DIR/INPUT.txt"
JOB_ID=$(cd "$RUN_DIR" && sbatch --parsable \
  --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC",DGFS_P4A_ROOT="$BASELINE" \
  ./run.slurm)
echo "P4E_JOB_ID=${JOB_ID%%;*}"
echo "P4E_RUN_DIR=$RUN_DIR"
echo "P4E_BASELINE_ROOT=$BASELINE"
echo P4E_BOOTSTRAP_COMPLETE
