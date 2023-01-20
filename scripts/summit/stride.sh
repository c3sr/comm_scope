#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -P csc465
#BSUB -J stride

module reset
module load gcc
module load cuda/11.0.3

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc465

date

# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/stride.csv \
--benchmark_filter="stride.*/0/(0|3)/" \
--cuda 0 --cuda 3

date
