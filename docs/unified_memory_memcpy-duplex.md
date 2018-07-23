# Unified Memory Memcpy-Duplex Bandwidth

Comm|Scope defines 1 microbenchmark to measure unified memory duplex bandwidth.
This benchmark may be listed with the argument
    
    --benchmark_filter="DUPLEX_Memcpy"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `DUPLEX_Memcpy_GPUGPU` | GPUToGPU | `log2 size / src GPU / dst GPU` |

## GPU/GPU Technique

For a gpu0 --> gpu1, gpu1 -> gpu0 transfer, the benchmark setup phase looks like this:

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
      //move pages
      cudaEventRecord(start, stream)
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)
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
