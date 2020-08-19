#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J ZeroCopy_GPUToGPU

module reset
module load gcc/5.4.0
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
--benchmark_out=$SCRATCH/ZeroCopy_GPUToGPU.csv \
--benchmark_filter="ZeroCopy_GPUToGPU.*/0/(1|3)/" \
--numa 0 --cuda 0 --cuda 1 --cuda 3

date
