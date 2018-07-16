# Unified Memory Latency

Comm | Scope defines 3 microbenchmarks to measure unified memory coherence bandwidth.
These benchmarks may be listed with the argument
    
    --benchmark_filter="Comm_UM_Coherence"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `Comm_UM_Latency_GPUToGPU` | GPUToGPU | `log2 size / src GPU / dst GPU` |
| `Comm_UM_Latency_GPUToHost` | GPUToHost | `log2 size / Host NUMA node / GPU` |
| `Comm_UM_Latency_HostToGPU` | Host To GPU | `log2 size / Host NUMA node / GPU` |

## Technique
