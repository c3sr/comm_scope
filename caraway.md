## Sandia Caraway

```
salloc -p MI250
export MODULEPATH="$MODULEPATH:/projects/x86-64-naples-nvidia/modulefiles/"
```

```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=hipcc \
-DSCOPE_USE_HIP=ON \
-DSCOPE_USE_NUMA=ON
```

## A100

```
salloc -p A100

export MODULEPATH="$MODULEPATH:/projects/x86-64-naples-nvidia/modulefiles/"
module load cuda/11.4 cmake numa/2.0.11

export COMM_SCOPE_SRC=$HOME/repos/comm_scope
export COMM_SCOPE_BUILD=$COMM_SCOPE_SRC/build-caraway-a100

mkdir -p $COMM_SCOPE_BUILD
cmake \
-S $COMM_SCOPE_SRC \
-B $COMM_SCOPE_BUILD \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_CUDA_COMPILER=nvcc \
-DSCOPE_USE_CUDA=ON \
-DSCOPE_USE_NUMA=ON \
2>&1 | tee $COMM_SCOPE_BUILD/configure.log

m -C $COMM_SCOPE_BUILD \
| tee $COMM_SCOPE_BUILD/build.log

```

### CUDA Memcpy Async latency

```
$COMM_SCOPE_BUILD/comm_scope --benchmark_list_tests --benchmark_filter="Comm_cudaMemcpyAsync_(PinnedToGPU|GPUToPinned)/0/0"
```
