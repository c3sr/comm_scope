# Vortex



```
mkdir build-vortex && cd build-vortex
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSCOPE_USE_CUDA=ON \
  -DSCOPE_USE_NUMA=ON
```


```
jsrun -n 1 -g 4 -c 16 ./comm_scope --benchmark_list_tests
```