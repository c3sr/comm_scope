#!/bin/bash

#SBATCH -A CSC465
#SBATCH -J hipMemcpy_GPUToGPU
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -G 4

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

srun -n 1 -c 64 -G 4 $ROOT/build-crusher/comm_scope \
--benchmark_filter=hipMemcpy_GPUToGPU \
--benchmark_out="$ROOT"/scripts/crusher/hipMemcpy_GPUToGPU.csv \
--benchmark_out_format=csv
