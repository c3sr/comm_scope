#!/bin/bash

#SBATCH -A CSC465
#SBATCH -J implicit_mapped_GPUWrGPU
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

srun -n 1 -G 4 -c 64 \
$ROOT/build-crusher/comm_scope \
--benchmark_filter=implicit_mapped_GPUWrGPU/ \
--benchmark_out="$ROOT"/scripts/crusher/implicit_mapped_GPUWrGPU.csv \
--benchmark_out_format=csv
