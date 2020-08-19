#!/bin/bash
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J list

module reset
module load gcc
module load cuda

set -eou pipefail
set -x

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
--benchmark_filter="UM" \
--benchmark_list_tests \
--numa 0 \
--cuda 0 --cuda 1 --cuda 3

date
