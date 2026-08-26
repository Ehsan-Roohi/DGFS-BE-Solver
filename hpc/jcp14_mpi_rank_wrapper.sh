#!/usr/bin/env bash
set -Eeuo pipefail

WORLD_RANK=${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is required}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?OMPI_COMM_WORLD_LOCAL_RANK is required}

case "$LOCAL_RANK" in
    0|1) ;;
    *)
        echo "JCP14_MPI_INVALID_LOCAL_RANK world_rank=$WORLD_RANK local_rank=$LOCAL_RANK" >&2
        exit 64
        ;;
esac

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
echo "JCP14_MPI_RANK_BIND world_rank=$WORLD_RANK local_rank=$LOCAL_RANK visible=$CUDA_VISIBLE_DEVICES" >&2
exec "$@"
