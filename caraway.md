## Sandia Caraway

```
salloc -p MI250
```

```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=hipcc \
-DSCOPE_USE_HIP=ON \
-DSCOPE_USE_NUMA=ON
```