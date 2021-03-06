#!/bin/bash
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -P csc362

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

module load gcc
module load cuda

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_list_tests \
--benchmark_out=$SCRATCH/cudaMemcpyAsync_GPUToPinned.csv \
--benchmark_filter="Comm_cudaMemcpyAsync_GPUToPinned.*/0/(0|3)"
date
