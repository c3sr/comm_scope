# NUMA-Pinned Explicit cudaMemcpy Bandwidth

Comm|Scope defines 10 microbenchmarks to measure explicit cudaMemcpyAsync bandwidth, including simultaneous bidirectional transfers.

The NUMA nodes and GPUs may be selected with the `-n` and `-c` flags.

For `HostToPinned`, the first two `-n` flags control the source and destination NUMA nodes.

## Implementations

|`--benchmark_filter=`|Description|Argument Format|
|-|-|-|
| `Comm_NUMAMemcpy_GPUToGPU` | GPU to GPU | `log2 size` |
| `Comm_NUMAMemcpy_GPUToHost` | GPU to pageable host | `log2 size` |
| `Comm_NUMAMemcpy_GPUToPinned` | GPU to pinned host | `log2 size` |
| `Comm_NUMAMemcpy_GPUToWC` | GPU to write-combining host | `log2 size` |
| `Comm_NUMAMemcpy_HostToGPU` | Pageable host to GPU | `log2 size` |
| `Comm_NUMAMemcpy_HostToPinned` | Pageable host to Pinned Host| `log2 size` |
| `Comm_NUMAMemcpy_PinnedToGPU` | Pinned host to GPU| `log2 size` |
| `Comm_NUMAMemcpy_WCToGPU` | Write-combining host to GPU | `log2 size` |
| `Comm_Duplex_Memcpy_Host` | GPU / pageable host bidirectional | `log2 size` |
| `Comm_Duplex_Memcpy_Pinned` | GPU / pinned host bidirectional | `log2 size` |

## Techniques

### Host allocation setup

When host allocations are required, they are created in the following way

```cpp
numa_bind(numa_id);

// For pinned
ptr = aligned_alloc(page_size, bytes);
cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);

// For pageable
ptr = aligned_alloc(page_size, bytes)

// For Write-Combined
cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined);
```

### Device allocation setup

When a device allocations are required, they are created in the following way

```cpp
cudaSetDevice(cuda_id)
cudaMallocManaged(&ptr, bytes, size);
```

### Bidirectional Transfers

For bidirectional transfers, one stream is created for each direction.
The elapsed time is measured from the start of whichever copy begins first, to the end of whichever copy finished last.

### Benchmark Loop

For all benchmarks the benchmark loop looks like this

```
loop (state)

    loop(streams)
      //move pages
      cudaEventRecord(start, stream)
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream)
      cudaEventRecord(stop, stream)
    end loop

    loop(stops){
      cudaEventSynchronize(stops)
    }
    end loop

    //record time
    loop(starts)
      loop(stops)
        cudaEventElapsedTime(&millis, start, stop)
      end loop
    end loop

    //record spread between starts/stops
    loop(streams)
      cudaEventElapsedTime(&startTime1, starts[0], starts[1])
      cudaEventElapsedTime(&startTime2, starts[1], starts[0])
      cudaEventElapsedTime(&stopTime1, stops[0], stops[1])
      cudaEventElapsedTime(&stopTime1, stops[1], stops[0])
    end loop
end loop

```


