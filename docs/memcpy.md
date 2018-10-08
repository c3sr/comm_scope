# Explicit Memcpy Bandwidth

These benchmarks examine memcpy bandwidth achieved through explicit `cudaMemcpyAsync` calls.
The GPUs are selected with the `-c` command-line flag.
Dual-GPu benchmarks use the first two devices.
Single-GPU benchmarks use the first device.
To use GPUs 0 and 1, for example:

    -c 0 -c 1

## Implementations

| `--benchmark_filter=`|Description|Argument Format|
|-|-|-|
| `Comm_Memcpy_GPUToGPUPeer` | GPU to GPU with peer access enabled | `log2 size` |
| `Comm_Memcpy_GPUToHost`    | GPU to pageable host                | `log2 size` |
| `Comm_Memcpy_GPUToWC`      | GPU to write-combining host         | `log2 size` |
| `Comm_Memcpy_HostToGPU`    | Pageable host to GPU                | `log2 size` |
| `Comm_Memcpy_WCToGPU`      | Write-combining host to GPU         | `log2 size` |
| `Comm_Duplex_Memcpy_GPUGPU` | GPU / GPU bidirectional            | `log2 size` |

## Technique

Whether the benchmark has one transfer or two, a source and destination allocation are established for each transfer.
Pageable host allocations are `memset` to ensure that backing pages are allocated.
`cudaSetDevice` is used to select the desired CUDA device.
During each iteration, a call to `cudaMemcpyAsync` is surrounded by two `cudaEventRecord`s, and then the time between those events is computed to determine the transfer time.

### Duplex Transfers

One stream for each direction is established, and asynchronous memory transfers with `cudaMemcpyAsync` are started on both streams.
The total time is measured as the difference between the earliest start of the first transfer and the final end of the last transfer.

```cpp
//setup: create one  stream per copy
vector<cudaStream_t> streams

//start and stop event for each copy
vector<cudaEvent_t> starts
vector<cudaEvent_t> stops

//allocate src and dst for copies
cudaSetDevice(gpu0)
cudaMalloc(&ptr, bytes)
cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0)
cudaSetDevice(gpu1)
cudaMalloc(&ptr, bytes)
err = cudaDeviceEnablePeerAccess(gpu0,0)

//allocate src and dst for second copy
cudaSetDevice(gpu1)
cudaMalloc(&ptr, bytes)
err = cudaDeviceEnablePeerAccess(gpu0, 0)
cudaSetDevice(gpu0)
cudaMalloc(&ptr, bytes)
err = cudaDeviceEnablePeerAccess(gpu1,0)


// benchmark loop
loop (state)

    loop(streams)
      cudaEventRecord(start, stream)
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)
      cudaEventRecord(stop, stream)
    end loop

    loop(stops){
      cudaEventSynchronize(stops)
    }
    end loop

    // find longest time between any pair of start and stop events
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
