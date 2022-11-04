#!/bin/bash

#SBATCH -A CSC465
#SBATCH -J hipMemcpy_GPUToGPU
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

srun -G 4 $ROOT/build-crusher/comm_scope \
--benchmark_filter=hipMemcpy_GPUToGPU \
--benchmark_out="$ROOT"/scripts/crusher/hipMemcpy_GPUToGPU.csv \
--benchmark_out_format=csv
