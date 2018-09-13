# Explicit Memcpy Bandwidth

These benchmarks examine memcpy bandwidth achieved through explicit `cudaMemcpyAsync` calls.
    
    --benchmark_filter="DUPLEX_Memcpy_GPUGPU"

The GPUs are selected with the `--cuda_device_ids` command-line flag.
To use GPUs 0 and 1, for example:

    --cuda_device_ids=0,1

## Implementations

| `--benchmark_filter=`|Description|Argument Format|
|-|-|-|
| `Comm_Memcpy_GPUToGPUPeer` | GPU to GPU with peer access enabled | `log2 size` |
| `Comm_Memcpy_GPUToHost`    | GPU to pageable host                | `log2 size` |
| `Comm_Memcpy_GPUToWC`      | GPU to write-combining host         | `log2 size` |
| `Comm_Memcpy_HostToGPU`    | Pageable host to GPU                | `log2 size` |
| `Comm_Memcpy_WCToGPU`      | Write-combining host to GPU         | `log2 size` |
| `Comm_MemcpyDuplex_GPUGPU` | GPU to GPU bidirectional            | `log2 size` |

## Technique

### Unidirectional Transfers


### Duplex Transfers

One stream for each direction is established, and asynchronous memory transfers with `cudaMemcpyAsync` are started on both streams.
The total time is measured as the difference between when the earlier transfer starts and the later transfer ends.

```
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
