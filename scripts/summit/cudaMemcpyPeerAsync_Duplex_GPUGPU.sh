#!/bin/bash
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -P csc362
#BSUB -J cudaMemcpyPeerAsync_Duplex_GPUGPU

module reset
module load gcc/5.4.0
module load cuda
module load nsight-systems/2020.3.1.71

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs ../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/cudaMemcpyPeerAsync_Duplex_GPUGPU.csv \
--benchmark_filter="cudaMemcpyPeerAsync_Duplex_GPUGPU/0/(1|3)"

date
