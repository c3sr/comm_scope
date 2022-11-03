#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J cudart_kernel

module reset
module load gcc
module load cuda/11.0.3
#module load nsight-systems/2020.3.1.71

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

date

# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs \
../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/cudart_kernel.csv \
--benchmark_filter="cudart_kernel/0/(0|3)" \
--numa 0 \
--cuda 0 --cuda 3

date

exit 0

nsys profile -t cuda -o $SCRATCH/cudart_kernel -f true \

