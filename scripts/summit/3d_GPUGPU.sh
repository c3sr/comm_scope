#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J 3d_GPUGPU

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
--benchmark_out=$SCRATCH/3d_GPUGPU.csv \
--benchmark_filter="3d.*(pack|GPUToGPU|pull|push).*/0/(1|3)/" \
--cuda 0 --cuda 1 --cuda 3

date
