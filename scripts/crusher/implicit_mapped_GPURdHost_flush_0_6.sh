#!/bin/bash

#SBATCH -A CSC465_crusher
#SBATCH -J implicit_mapped_GPURdHost_flush_0_6
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 8

set -eou pipefail

ROOT="/ccs/home/cpearson/repos/comm_scope"

. $ROOT/load-env.sh

echo export HAS_XNACK=1
export HAS_XNACK=1

srun -n 1 -G 8 -c 56 \
$ROOT/build-crusher/comm_scope \
--benchmark_filter=implicit_mapped_GPURdHost_flush/0/6 \
--benchmark_out="$ROOT"/scripts/crusher/implicit_mapped_GPURdHost_flush_0_6.csv \
--benchmark_out_format=csv
