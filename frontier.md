## OLCF Frontier

```
salloc -A CSC465 -J interactive -t 02:00:00 -p batch -N 1
srun -n 1 -G 8 ./comm_scope --benchmark_filter="hipMemcpyAsync.*Pageable"
srun -n 1 -G 8 ./comm_scope --benchmark_filter="implicit_mapped_GPURdHost"
```

```
mkdir -p build-frontier && cd build-frontier
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=hipcc \
-DSCOPE_ARCH_MI250X=ON \
-DSCOPE_USE_NUMA=ON

nice -n20 make -j16
```

```
squeue -u cpearson -o %all | cut -d'|' -f9,10
```