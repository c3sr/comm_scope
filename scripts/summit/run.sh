#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --time=1:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --sockets-per-node=2 
#SBATCH --cores-per-socket=20 
#SBATCH --threads-per-core=4 
#SBATCH --mem-per-cpu=1200
#SBATCH --gres=gpu:v100:4

set -eou pipefail

export SCRATCH=/gpfs/alpine/scratch/cpearson/csc362

date
# -n (total rs)
# -g (gpus per rs)
# -a (MPI tasks per rs)
# -c (cores per rs)
jsrun -n1 -a1 -g6 -c42 -b rs js_task_info ../../build/comm_scope \
#--benchmark_list_tests \
--benchmark_out_format=csv \
--benchmark_out=$SCRATCH/cudaMemcpyAsync_GPUToGPU.csv \
--benchmark_filter="cudaMemcpyAsync_GPUToGPU.*0/0/(1|3)"
date
