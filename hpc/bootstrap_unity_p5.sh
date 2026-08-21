#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "P5_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
REPO=${DGFS_REPO_URL:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_GIT_REF:-agent/phase5-independent-validation}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT/p5_$STAMP"

PROFILES=${DGFS_P5_PROFILES:-}
if [[ -z "$PROFILES" && -n "${DGFS_P5_P4E_DIR:-}" ]]; then
    PROFILES="$DGFS_P5_P4E_DIR/p4e_results/p4e_physical_profiles.csv"
fi
if [[ -z "$PROFILES" ]]; then
    PROFILES=$(find "$ROOT" -maxdepth 6 -type f -name 'p4e_physical_profiles.csv' \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [[ ! -s "$PROFILES" ]]; then
    echo "P5_PHYSICAL_PROFILES_NOT_FOUND_UNDER=$ROOT"
    echo "Set DGFS_P5_PROFILES=/absolute/path/p4e_physical_profiles.csv"
    find "$ROOT" -maxdepth 4 -type f \( -name 'p4e_*.zip' -o -name '*physical_profiles*.csv' \) \
        -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20 || true
    exit 2
fi
echo "P5_DISCOVERED_PROFILES=$PROFILES"

EXTERNAL=${DGFS_P5_REFERENCE_CSV:-}
PROVENANCE=${DGFS_P5_REFERENCE_PROVENANCE:-}
if [[ -z "$EXTERNAL" ]]; then
    for candidate in "$ROOT/reference/jcp2019_dsmc.csv" "$ROOT/reference/jcp2019_dvm.csv"; do
        [[ -s "$candidate" ]] && { EXTERNAL=$candidate; break; }
    done
fi
if [[ -n "$EXTERNAL" && -z "$PROVENANCE" ]]; then
    PROVENANCE="${EXTERNAL%.*}.provenance.json"
fi
if [[ -n "$EXTERNAL" ]]; then
    [[ -s "$PROVENANCE" ]] || { echo "P5_REFERENCE_PROVENANCE_NOT_FOUND=$PROVENANCE"; exit 3; }
fi

mkdir -p "$RUN_DIR"
git clone --depth 1 --branch "$REF" "$REPO" "$RUN_DIR/src"
JOB=$(sbatch --parsable --chdir="$RUN_DIR" \
    --export=ALL,P5_RUN_DIR="$RUN_DIR",P5_SOURCE_DIR="$RUN_DIR/src",P5_PROFILES="$PROFILES",P5_EXTERNAL_REFERENCE="$EXTERNAL",P5_EXTERNAL_PROVENANCE="$PROVENANCE",P5_TIME="${DGFS_P5_TIME:-1.0}" \
    "$RUN_DIR/src/hpc/p5_external_validation.slurm")
JOB=${JOB%%;*}
echo "P5_JOB_ID=$JOB"
echo "P5_RUN_DIR=$RUN_DIR"
echo "P5_SOURCE_PROFILES=$PROFILES"
echo "P5_EXTERNAL_REFERENCE=${EXTERNAL:-NONE}"
echo "P5_EXTERNAL_PROVENANCE=${PROVENANCE:-NONE}"
echo "P5_BOOTSTRAP_COMPLETE"
