#!/bin/bash
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -P csc465
#BSUB -J cudaMemcpyPeerAsync_Duplex_GPUGPUPeer

module reset
module load gcc/5.4.0
module load cuda

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc465

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/cudaMemcpyPeerAsync_Duplex_GPUGPUPeer.csv \
--benchmark_filter="cudaMemcpyPeerAsync_Duplex_GPUGPUPeer/0/(1|3)"
date
