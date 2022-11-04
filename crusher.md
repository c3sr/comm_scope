## OLCF Crusher

```
salloc -A csc465 -J interactive -t 03:00:00 -p batch -N 1 -n 1 -c 64
```

```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=hipcc \
-DSCOPE_USE_HIP=ON \
-DSCOPE_USE_NUMA=ON
```