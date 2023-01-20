#!/bin/bash

#SBATCH -A CSC465_crusher
#SBATCH -J hipMemcpy_GPUToNUMA_flush_0_3
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
--benchmark_filter=hipMemcpy_GPUToNUMA_flush/0/3 \
--benchmark_out="$ROOT"/scripts/crusher/hipMemcpy_GPUToNUMA_flush_0_3.csv \
--benchmark_out_format=csv
