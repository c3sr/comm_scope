## OLCF Crusher

```
salloc -A CSC465_crusher -J interactive -t 03:00:00 -p batch -N 1
srun -n 1 -G 8 ./comm_scope --benchmark_filter="hipMemcpyAsync.*Pageable"
```

```
mkdir build-crusher && cd build-crusher
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=hipcc \
-DSCOPE_ARCH_MI250X=ON \
-DSCOPE_USE_NUMA=ON
```
