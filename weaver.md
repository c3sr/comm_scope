# Weaver

```
# -x for exclusive node
bsub -x -gpu num=1 -Is $SHELL

module load gcc/11.3.0 cuda/11.8 cmake

export COMM_SCOPE_SRC=$HOME/repos/comm_scope
export COMM_SCOPE_BUILD=$COMM_SCOPE_SRC/build-weaver-v100

mkdir -p $COMM_SCOPE_BUILD
cmake \
-S $COMM_SCOPE_SRC \
-B $COMM_SCOPE_BUILD \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_CUDA_COMPILER=nvcc \
-DSCOPE_USE_CUDA=ON \
-DSCOPE_USE_NUMA=OFF \
2>&1 | tee $COMM_SCOPE_BUILD/configure.log

m -C $COMM_SCOPE_BUILD \
| tee $COMM_SCOPE_BUILD/build.log

$COMM_SCOPE_BUILD/comm_scope --benchmark_filter="Comm_cudaMemcpyAsync_(PinnedToGPU|GPUToPinned)/0/0" --benchmark_format=csv --benchmark_list_tests
```