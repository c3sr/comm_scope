#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -P csc465
#BSUB -J cudaMemcpyAsync_Host 
#BSUB -o cudaMemcpyAsync_Host.o%J
#BSUB -e cudaMemcpyAsync_Host.e%J

set -eou pipefail

export ROOT=$HOME/repos/comm_scope
export SCRATCH=/gpfs/alpine/scratch/cpearson/csc465
export SCRIPTS=$ROOT/scripts/summit
export CSV=cudaMemcpyAsync_Host.csv

. $ROOT/load-env.sh

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info $ROOT/build-summit/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/$CSV \
--benchmark_filter="cudaMemcpyAsync_(HostToGPU|GPUToHost).*/0/(0|3)"

mv -v $SCRATCH/$CSV $SCRIPTS/$CSV

date
