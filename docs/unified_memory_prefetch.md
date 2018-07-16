# Unified Memory Prefetch Bandwidth

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `Comm_UM_Prefetch_GPUToGPU` | GPUToGPU | `log2 size / src GPU / dst GPU` |
| `Comm_UM_Prefetch_GPUToHost` | GPUToHost | `log2 size / Host NUMA node / GPU` |
| `Comm_UM_Prefetch_HostToGPU` | Host To GPU | `log2 size / Host NUMA node / GPU` |

## Technique

