# Unified Memory Coherence Bandwidth

Comm|Scope defines 3 microbenchmarks to measure unified memory coherence bandwidth.
These benchmarks may be listed with the argument
    
    --benchmark_filter="Comm_UM_Coherence"
    
The NUMA and/or CUDA devices may be selected with the `--numa_ids` and `--cuda_ids` command line flags.

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `Comm_UM_Coherence_GPUToGPU` | GPUToGPU | `log2 size / src GPU / dst GPU` |
| `Comm_UM_Coherence_GPUToHost` | GPUToHost | `log2 size / Host NUMA node / GPU` |
| `Comm_UM_Coherence_HostToGPU` | Host To GPU | `log2 size / Host NUMA node / GPU` |

## Technique

A single unified-memory allocation is established.
`cudaMemPrefetchAsync` is used to ensure that the backing pages are on the source device at the beginning of each benchmark iteration.
The destination device writes a single value to each page in the allocation, forcing a coherence page migration.

### CPU/GPU setup phase

For a host -> device transfer, the setup and benchmark loop looks like this

```
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
    write_gpu(ptr) 
    cudaEventRecord(stop)
    cudaEventSynchronize(stop)
    state.time = cudaEventGetElapsedTime(start, stop)
end loop
```

For a device -> host transfer, the setup and benchmark loop looks like this

```
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
    write_cpu(ptr)
    state.stopTiming()
end loop
```


## GPU/GPU Technique

```
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
    write_gpu(ptr) 
    cudaEventRecord(stop)
    // record time
    cudaEventSynchronize(stop)
    state.time = cudaEventGetElapsedTime(start, stop)
end loop
```
