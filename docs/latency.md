# Unified Memory Latency

Comm | Scope defines 3 microbenchmarks to measure unified memory coherence latency.
These benchmarks may be listed with the argument
    
    --benchmark_filter="Comm_UM_Latency"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `Comm_UM_Latency_GPUToGPU` | GPU To GPU | `log2 size / src GPU / dst GPU` |
| `Comm_UM_Latency_GPUToHost` | GPU To Host | `log2 size / Host NUMA node / GPU` |
| `Comm_UM_Latency_HostToGPU` | Host To GPU | `log2 size / Host NUMA node / GPU` |

## Technique

A single unified memory allocation is created with `cudaMallocManaged`.
A linked-list is set up in that allocation, with each element being the offset of the next element in the list.
The stride between elements is set to 128K, to be at least two 64K pages apart (on POWER).

`cudaMemPrefetchAsync` is used to ensure that pages begin each iteration on the source device.
A single-threaded traversal workload is executed on the destination device, which traverses a fixed number of linked list strides.
Each access should incur moving a page from src to dst.
The marginal performance increase as the number of strides is increased should correspond to the total cost of serving a load against a page that is not present on the destination device.

### GPU Traversal Function

```cuda
__global__ void gpu_traverse(size_t *ptr, const size_t steps) {
  size_t next = 0;
  for (int i = 0; i < steps; ++i) {
    next = ptr[next];
  }
  ptr[next] = 1;
}
```

invoked with

```cuda
gpu_traverse<<<1,1>>>(...)
```

### CPU Traversal Function

```c++
void cpu_traverse(size_t *ptr, const size_t steps) {

  size_t next = 0;
  for (size_t i = 0; i < steps; ++i) {
    next = ptr[next];
  }
  ptr[next] = 1;
}
```
