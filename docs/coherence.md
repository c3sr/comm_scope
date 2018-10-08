# NUMA-Pinned Unified Memory Coherence Bandwidth

Comm|Scope defines 3 microbenchmarks to measure unified memory coherence bandwidth.
These benchmarks may be listed with the argument
    
The NUMA and/or CUDA devices may be selected with the `-n` and `-c` command line flags.

## Implementations

|`--benchmark_filter`|Description|Argument Format|
|-|-|-|
| `Comm_UM_Coherence_GPUToGPU` | GPU To GPU | `log2 size` |
| `Comm_UM_Coherence_GPUToHost` | GPU To Host | `log2 size` |
| `Comm_UM_Coherence_HostToGPU` | Host To GPU | `log2 size` |

## Technique

A single unified-memory allocation is established.
`cudaMemPrefetchAsync` is used to ensure that the backing pages are on the source device at the beginning of each benchmark iteration.
The destination device writes a single value to each page in the allocation, forcing a coherence page migration.

### `gpu_write` kernel

```cpp
__global__ void gpu_write(char *ptr, const size_t count, const size_t stride) {
  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx             = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}
```

### `cpu_write` kernel

```cpp
static void cpu_write(char *ptr, const size_t n, const size_t stride) {
  for (size_t i = 0; i < n; i += stride) {
    benchmark::DoNotOptimize(ptr[i] = 0);
  }
}
```


### CPU to GPU

```cpp
// host-to-device setup
numa_bind(src)
cudaSetDevice(dst)
cudaMallocManaged(&ptr)

// host-to-device benchmark loop
loop (state)
    // move pages to src
    cudaMemPrefetchAsync(ptr, src)
    cudaEventRecord(start)
    // execute on dst to force page access
    gpu_write(ptr) 
    cudaEventRecord(stop)
    cudaEventSynchronize(stop)
    state.time = cudaEventGetElapsedTime(start, stop)
end loop
```

### GPU to CPU

```cpp
// device-to-host setup
cudaSetDevice(src)
numa_bind(dst)
cudaMallocManaged(&ptr)

// device-to-host benchmark loop
loop (state)
    // move pages to src
    cudaMemPrefetchAsync(ptr, src)
    cudaDeviceSynchronize(src)
    // execute workload on cpu
    state.resumeTiming()
    cpu_write(ptr)
    state.stopTiming()
end loop
```


### GPU to GPU

```cpp
//setup
cudaMallocManaged(&ptr)

// benchmark loop
loop (state)
    // move pages to src
    cudaMemPrefetchAsync(ptr, src)
    // ensure pages are on src
    cudaSetDevice(src)
    cudaDeviceSynchronize()
    // execute workload on dst device
    cudaSetDevice(dst)
    cudaEventRecord(start)
    gpu_write(ptr) 
    cudaEventRecord(stop)
    // record time
    cudaEventSynchronize(stop)
    state.time = cudaEventGetElapsedTime(start, stop)
end loop
```
