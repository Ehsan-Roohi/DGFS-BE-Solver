#!/bin/bash
#SBATCH --job-name=dgfs-p5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
trap 'rc=$?; echo "P5_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

: "${P5_RUN_DIR:?}"
: "${P5_SOURCE_DIR:?}"
: "${P5_PROFILES:?}"
cd "$P5_RUN_DIR"

module purge || true
module load conda/latest 2>/dev/null || module load conda 2>/dev/null || true

PYTHON=${P5_PYTHON:-python3}
ARGS=(--profiles "$P5_PROFILES" --time "${P5_TIME:-1.0}" --out-dir results)
if [[ -n "${P5_EXTERNAL_REFERENCE:-}" ]]; then
    [[ -s "$P5_EXTERNAL_REFERENCE" ]]
    [[ -s "${P5_EXTERNAL_PROVENANCE:-}" ]]
    ARGS+=(--external-reference "$P5_EXTERNAL_REFERENCE" --external-provenance "$P5_EXTERNAL_PROVENANCE")
fi

"$PYTHON" "$P5_SOURCE_DIR/cases/jcp2019_fig14b_normal_shock/validation/p5_validate.py" "${ARGS[@]}" | tee p5_audit.log

cp -p "$P5_PROFILES" results/input_physical_profiles.csv
[[ -z "${P5_EXTERNAL_REFERENCE:-}" ]] || cp -p "$P5_EXTERNAL_REFERENCE" results/external_reference.csv
[[ -z "${P5_EXTERNAL_PROVENANCE:-}" ]] || cp -p "$P5_EXTERNAL_PROVENANCE" results/external_provenance.json
printf 'source_profiles=%s\nexternal_reference=%s\nexternal_provenance=%s\ntime=%s\n' \
    "$P5_PROFILES" "${P5_EXTERNAL_REFERENCE:-NONE}" "${P5_EXTERNAL_PROVENANCE:-NONE}" "${P5_TIME:-1.0}" > results/INPUT.txt

touch P5_SUCCESS
ZIP="p5_${SLURM_JOB_ID}.zip"
zip -q -9 -r "$ZIP" results p5_audit.log P5_SUCCESS
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "UPLOAD_ZIP=$P5_RUN_DIR/$ZIP"
echo "UPLOAD_SHA=$P5_RUN_DIR/$ZIP.sha256.txt"
echo "P5_COMPLETE"
