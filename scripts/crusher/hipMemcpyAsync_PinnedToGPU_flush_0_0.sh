#!/bin/bash

#SBATCH -A CSC465_crusher
#SBATCH -J hipMemcpyAsync_PinnedToGPU_flush_0_0
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 8

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

srun -n 1 -G 8 -c 56 \
$ROOT/build-crusher/comm_scope \
--benchmark_filter=hipMemcpyAsync_PinnedToGPU_flush/0/0 \
--benchmark_out="$ROOT"/scripts/crusher/hipMemcpyAsync_PinnedToGPU_flush_0_0.csv \
--benchmark_out_format=csv
