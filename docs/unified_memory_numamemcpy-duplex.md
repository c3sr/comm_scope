# Unified Memory Numamemcpy-Duplex Bandwidth

Comm|Scope defines 4 microbenchmarks to measure unified memory duplex bandwidth.
These benchmarks may be listed with the argument
    
    --benchmark_filter="DUPLEX_Memcpy"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `DUPLEX_Memcpy_HostToGPU` | HostToGPU | `log2 size / Host NUMA / GPU` |
| `DUPLEX_Memcpy_GPUToHost` | GPUToHost | `log2 size / GPU / Host NUMA` |
| `DUPLEX_Memcpy_PinnedToGPU` | PinnedToGPU | `log2 size / Host NUMA / GPU` |
| `DUPLEX_Memcpy_GPUToPinned` | GPUToPinned | `log2 size / GPU / Host NUMA` |

## CPU/GPU Technique 

For a host -> device, device -> host transfer, the benchmark setup phase looks like this

```
// host-to-device device-to-host setup
// create one stream per copy
vector<cudaStream_t> streams

// start and stop events for each copy
vector<cudaEvent_t> starts
vctor<cudaEvent_t> stops


//src and dst allocation for copies
cudaSetDevice(numa)
malloc(bytes)
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)

//src and dst allocation for second copy
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)
cudaSetDevice(numa)
malloc(bytes)

```
## CPU/GPU Pinned Technique

For a pinned host -> device, device -> pinned host transfer, the benchmark setup phase looks like this

```
// host-to-device, device-to-host setup: create one stream per copy
vector<cudaStream_t> streams

// start and stop events for each copy
vector<cudaEvent_t> starts
vctor<cudaEvent_t> stops

//src and dst allocation for copies
cudaSetDevice(numa)
cudaMallocHost(&ptr, bytes)
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)

//src and dst allocation for second copy
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)
cudaSetDevice(numa)
cudaMallocHost(&ptr, bytes)

```
## GPU/CPU Technique

For a device -> host, host -> device transfer, the benchmark setup phase looks like this

```
// device-to-host, host-to-device setup: create one stream per copy
vector<cudaStream_t> streams

// start and stop events for each copy
vector<cudaEvent_t> starts
vctor<cudaEvent_t> stops

//src and dst allocation for copies
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)
cudaSetDevice(numa)
malloc(bytes)

//src and dst allocation for second copy
cudaSetDevice(numa)
malloc(bytes)
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)

```
## GPU/CPU Pinnned Technique

For a device -> host, host -> device transfer, the benchmark setup phase looks like this

```
// device-to-host, host-to-device setup: create one stream per copy
vector<cudaStream_t> streams

// start and stop events for each copy
vector<cudaEvent_t> starts
vctor<cudaEvent_t> stops

//src and dst allocation for copies
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)
cudaSetDevice(numa)
cudaMallocHost(&ptr, bytes)

//src and dst allocation for second copy
cudaSetDevice(numa)
cudaMallocHost(&ptr, bytes)
cudaSetDevice(gpu)
cudaMalloc(&ptr, bytes)

```

For all four benchmarks benchmark loop looks like this

```
loop (state)

    loop(streams)
      //
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


