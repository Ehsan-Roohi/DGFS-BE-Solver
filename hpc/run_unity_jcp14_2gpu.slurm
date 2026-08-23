#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --job-name=jcp14-mpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --constraint=vram16
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%A_%a.out

set -Eeuo pipefail
trap 'rc=$?; rm -f CASE_SUCCESS; echo "JCP14_MPI_CASE_FAILED elements=${ELEMENTS:-unknown} rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CAMPAIGN=${DGFS_MPI_CAMPAIGN:?DGFS_MPI_CAMPAIGN is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
ELEMENTS=${SLURM_ARRAY_TASK_ID:?run as array 4,8}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-24}
BENCH_SPAN=${DGFS_MPI_BENCH_SPAN:-0.5}
SEGMENT_SPAN=${DGFS_SEGMENT_SPAN:-10}
NOMINAL_DT=0.001
THRESHOLD=2e-5
RUN_DIR="$CAMPAIGN/e$ELEMENTS"
TOOLS="$SRC/hpc"
cd "$RUN_DIR"

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
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
mkdir -p "$CUDA_CACHE_PATH" serial_bench mpi_bench mpi_partition mpi_combined production final

python "$SRC/cases/jcp2019_fig14b_validation/verify_case.py"
srun --exclusive --exact --nodes=1 --ntasks=2 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=closest nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

read -r CHECKPOINT_TIME NORMALISATION < <(python - SOURCE.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{m['checkpoint_time']:.12g} {m['normalisation_residual']:.17g}")
PY
)
BENCH_END=$(python - "$CHECKPOINT_TIME" "$BENCH_SPAN" <<'PY'
import sys
print(f"{float(sys.argv[1]) + float(sys.argv[2]):.12g}")
PY
)
BENCH_TAG=$(python - "$BENCH_END" <<'PY'
import sys
print(f"{float(sys.argv[1]):.1f}")
PY
)

python "$TOOLS/partition_line_checkpoint.py" --parts 2 \
    --mesh source_mesh.frfsm --solution source_dist.frfss --output-dir mpi_partition
python "$TOOLS/make_jcp14_continuation_config.py" --source dgfs.ini \
    --output serial_bench/dgfs.ini --start "$CHECKPOINT_TIME" --end "$BENCH_END" \
    --output-span "$BENCH_SPAN" --normalisation "$NORMALISATION" --batch 256
cp serial_bench/dgfs.ini mpi_bench/dgfs.ini

echo "JCP14_MPI_BENCH_SERIAL_START elements=$ELEMENTS interval=$CHECKPOINT_TIME,$BENCH_END"
(
    cd serial_bench
    /usr/bin/time -f '%e' -o WALLTIME_SECONDS.txt \
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gpus-per-task=1 --gpu-bind=closest python -m frfs restart \
        ../source_mesh.frfsm ../source_dist.frfss dgfs.ini -b cuda > solver.log 2>&1
)
test -s "serial_bench/dist-$BENCH_TAG.frfss"
test -s "serial_bench/bulk-$BENCH_TAG.frfss"

echo "JCP14_MPI_BENCH_2GPU_START elements=$ELEMENTS interval=$CHECKPOINT_TIME,$BENCH_END"
(
    cd mpi_bench
    /usr/bin/time -f '%e' -o WALLTIME_SECONDS.txt \
        srun --exclusive --exact --nodes=1 --ntasks=2 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 python -m frfs restart \
        ../mpi_partition/source_mesh.frfsm ../mpi_partition/source_dist.frfss dgfs.ini -b cuda \
        > solver.log 2>&1
)
test -s "mpi_bench/dist-$BENCH_TAG.frfss"
test -s "mpi_bench/bulk-$BENCH_TAG.frfss"

python "$TOOLS/partition_line_checkpoint.py" --parts 1 \
    --mesh mpi_partition/source_mesh.frfsm --solution "mpi_bench/dist-$BENCH_TAG.frfss" \
    --output-dir mpi_combined
python "$TOOLS/combine_native_moments.py" \
    --source-mesh mpi_partition/source_mesh.frfsm \
    --source-bulk "mpi_bench/bulk-$BENCH_TAG.frfss" \
    --target-mesh mpi_combined/source_mesh.frfsm \
    --output "mpi_combined/bulk-$BENCH_TAG.frfss"
python "$TOOLS/verify_jcp14_mpi_equivalence.py" \
    --serial-dist "serial_bench/dist-$BENCH_TAG.frfss" \
    --serial-bulk "serial_bench/bulk-$BENCH_TAG.frfss" \
    --mpi-dist "mpi_combined/dist-$BENCH_TAG.frfss" \
    --mpi-bulk "mpi_combined/bulk-$BENCH_TAG.frfss" \
    --serial-wall serial_bench/WALLTIME_SECONDS.txt \
    --mpi-wall mpi_bench/WALLTIME_SECONDS.txt --output mpi_benchmark.json

BASE_MESH=mpi_partition/source_mesh.frfsm
RESTART_DIST="mpi_bench/dist-$BENCH_TAG.frfss"
FINAL_CFG=mpi_bench/dgfs.ini
SEG_START=$BENCH_END
FINAL_END=""
FINAL_RESIDUAL=""

if python "$SRC/cases/jcp2019_fig14b_validation/check_convergence.py" \
        --csv mpi_bench/kinetic_residual.csv --segment-start "$CHECKPOINT_TIME" \
        --segment-end "$BENCH_END" --dt "$NOMINAL_DT" --threshold "$THRESHOLD" \
        --output production/convergence_benchmark.json; then
    FINAL_END=$BENCH_TAG
    FINAL_RESIDUAL=$(python -c "import json; print(json.load(open('production/convergence_benchmark.json'))['paper_normalized_residual'])")
else
    RC=$?
    (( RC == 2 )) || exit "$RC"
fi

for ((SEGMENT=0; SEGMENT<MAX_SEGMENTS && ${#FINAL_END}==0; SEGMENT++)); do
    SEG_END=$(python - "$SEG_START" "$SEGMENT_SPAN" <<'PY'
import sys
print(f"{float(sys.argv[1]) + float(sys.argv[2]):.12g}")
PY
    )
    START_TAG=$(python - "$SEG_START" <<'PY'
import sys
print(f"{float(sys.argv[1]):.1f}")
PY
    )
    END_TAG=$(python - "$SEG_END" <<'PY'
import sys
print(f"{float(sys.argv[1]):.1f}")
PY
    )
    TAG=$(printf '%02d' "$SEGMENT")
    CFG="production/segment_$TAG.ini"
    LOG="production/solver_$TAG.log"
    CONV="production/convergence_$TAG.json"
    python "$TOOLS/make_jcp14_continuation_config.py" --source dgfs.ini \
        --output "$CFG" --start "$SEG_START" --end "$SEG_END" \
        --output-span "$SEGMENT_SPAN" --normalisation "$NORMALISATION" --batch 256
    FINAL_CFG=$CFG
    echo "JCP14_MPI_SEGMENT_START elements=$ELEMENTS segment=$SEGMENT interval=$SEG_START,$SEG_END"
    (
        cd production
        /usr/bin/time -f '%e' -o "wall_$TAG.txt" \
            srun --exclusive --exact --nodes=1 --ntasks=2 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 python -m frfs restart \
            "../$BASE_MESH" "../$RESTART_DIST" "segment_$TAG.ini" -b cuda \
            > "solver_$TAG.log" 2>&1
    )
    test -s "production/dist-$END_TAG.frfss"
    test -s "production/bulk-$END_TAG.frfss"
    if python "$SRC/cases/jcp2019_fig14b_validation/check_convergence.py" \
            --csv production/kinetic_residual.csv --segment-start "$SEG_START" \
            --segment-end "$SEG_END" --dt "$NOMINAL_DT" --threshold "$THRESHOLD" \
            --output "$CONV"; then
        FINAL_END=$END_TAG
        FINAL_RESIDUAL=$(python -c "import json; print(json.load(open('$CONV'))['paper_normalized_residual'])")
        RESTART_DIST="production/dist-$END_TAG.frfss"
        break
    else
        RC=$?
        (( RC == 2 )) || exit "$RC"
    fi
    RESTART_DIST="production/dist-$END_TAG.frfss"
    SEG_START=$SEG_END
done

if [[ -z "$FINAL_END" ]]; then
    echo "JCP14_MPI_MAX_SEGMENTS_REACHED elements=$ELEMENTS max=$MAX_SEGMENTS"
    exit 5
fi

if [[ "$RESTART_DIST" == mpi_bench/* ]]; then
    FINAL_MPI_DIST="$RESTART_DIST"
    FINAL_MPI_BULK="mpi_bench/bulk-$FINAL_END.frfss"
else
    FINAL_MPI_DIST="$RESTART_DIST"
    FINAL_MPI_BULK="production/bulk-$FINAL_END.frfss"
fi
python "$TOOLS/partition_line_checkpoint.py" --parts 1 --mesh "$BASE_MESH" \
    --solution "$FINAL_MPI_DIST" --output-dir final
FINAL_DIST_NAME=${FINAL_MPI_DIST##*/}
mv "final/$FINAL_DIST_NAME" final/dist-final.frfss
mv final/source_mesh.frfsm final/mesh.frfsm
python "$TOOLS/combine_native_moments.py" --source-mesh "$BASE_MESH" \
    --source-bulk "$FINAL_MPI_BULK" --target-mesh final/mesh.frfsm \
    --output final/bulk-final.frfss
cp "$FINAL_CFG" final/dgfs.ini
python -m frfs export final/mesh.frfsm final/bulk-final.frfss final/physical-final.vtu -d 24
test -s final/physical-final.vtu
cp final/mesh.frfsm mesh.frfsm
cp final/dist-final.frfss dist-final.frfss
cp final/bulk-final.frfss bulk-final.frfss
cp final/physical-final.vtu physical-final.vtu
cp production/kinetic_residual.csv kinetic_residual.csv 2>/dev/null || cp mpi_bench/kinetic_residual.csv kinetic_residual.csv
{
    echo "benchmark=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14"
    echo "mach=1.59"
    echo "spatial_elements=$ELEMENTS"
    echo "velocity_batch=256"
    echo "mpi_ranks=2"
    echo "source_checkpoint_time=$CHECKPOINT_TIME"
    echo "benchmark_end_time=$BENCH_END"
    echo "final_time=$FINAL_END"
    echo "paper_normalized_residual=$FINAL_RESIDUAL"
    echo "paper_threshold=$THRESHOLD"
    echo "solver_commit=$(git -C "$SRC" rev-parse HEAD)"
} > METADATA.txt
touch CASE_SUCCESS
sha256sum dgfs.ini mesh.frfsm dist-final.frfss bulk-final.frfss physical-final.vtu \
    kinetic_residual.csv mpi_benchmark.json METADATA.txt CASE_SUCCESS > SHA256SUMS.txt
echo "JCP14_MPI_CASE_COMPLETE elements=$ELEMENTS final_time=$FINAL_END residual=$FINAL_RESIDUAL"
