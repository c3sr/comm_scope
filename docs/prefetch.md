# NUMA-Pinned Unified Memory Prefetch Bandwidth

Comm|Scope defines 3 microbenchmarks to measure unified memory prefetch bandwidth.
These benchmarks may be listed with the argument
    
The NUMA and/or CUDA devices may be selected with the `-n` and `-c` command line flags.

## Implementations

|`--benchmark_filter=`|Description|Argument Format|
|-|-|-|
| `Comm_UM_Prefetch_GPUToGPU` | GPU To GPU | `log2 size` |
| `Comm_UM_Prefetch_GPUToHost` | GPU To Host | `log2 size` |
| `Comm_UM_Prefetch_HostToGPU` | Host To GPU | `log2 size` |

## Technique

A single unified-memory allocation is established.
`cudaMemPrefetchAsync` is used to ensure that the backing pages are on the source device at the beginning of each benchmark iteration.
The performancing of using `cudaMemPrefetchAsync` to move backing pages from the source device to the destination device is then timed by wrapping it in `cudaEvent_t`.

### Pseudocode

```cpp
// setup
numa_bind(src)
cudaSetDevice(dst)
cudaMallocManaged(&ptr)

// benchmark loop
loop (state)
    // move pages to src
    cudaMemPrefetchAsync(ptr, src)
    cudaEventRecord(start)
    // execute on dst to force page access
    cudaMemPrefetchAsync(ptr, dst)
    cudaEventRecord(stop)
    cudaEventSynchronize(stop)
    state.time = cudaEventGetElapsedTime(start, stop)
end loop
```

