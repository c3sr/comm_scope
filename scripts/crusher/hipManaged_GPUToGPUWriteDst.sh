#!/bin/bash

#SBATCH -A CSC465_crusher
#SBATCH -J hipManaged_GPUToGPUWriteDst
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -G 8

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

srun -n 1 -G 8 -c 64 \
$ROOT/build-crusher/comm_scope \
--benchmark_filter=hipManaged_GPUToGPUWriteDst/0/ \
--benchmark_out="$ROOT"/scripts/crusher/hipManaged_GPUToGPUWriteDst.csv \
--benchmark_out_format=csv
