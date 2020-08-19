#!/bin/bash
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J cudaMemcpyAsync_PinnedToGPU

module reset
module load gcc
module load cuda

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/cudaMemcpyAsync_PinnedToGPU.csv \
--benchmark_filter="Comm_cudaMemcpyAsync_PinnedToGPU.*/0/(0|3)"
date
