#!/bin/bash

#SBATCH -A CSC465_crusher
#SBATCH -J prefetch_managed_GPUToGPU
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 8

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

echo export HSA_XNACK=1
export HSA_XNACK=1

srun -n 1 -G 8 -c 56 \
$ROOT/build-crusher/comm_scope \
--benchmark_filter=prefetch_managed_GPUToGPU/0/ \
--benchmark_out="$ROOT"/scripts/crusher/prefetch_managed_GPUToGPU.csv \
--benchmark_out_format=csv
