/* Use half the GPU to write to the host and the other half to read from the
 * host
 */

#include "scope/scope.hpp"

#include "args.hpp"
#include "kernels.hu"

constexpr int rdDimBlock = 256;
constexpr int wrDimBlock = 256;
auto GpuWrFunc = gpu_write<rdDimBlock, int32_t>;
auto GpuRdFunc = gpu_read<rdDimBlock, int32_t>;

#define NAME Comm_ZeroCopy_Duplex_GPUGPU
#define NAME2 "Comm_ZeroCopy_Duplex_GPUGPU"

namespace NAME {
__global__ void busy_wait(clock_t *d, clock_t clock_count) {
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  if (d) {
    *d = clock_offset;
  }
}
} // namespace NAME

auto Comm_ZeroCopy_HostGPU = [](benchmark::State &state, const int numa,
                                const int cuda) {
  numa::ScopedBind binder(numa);

  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  cudaStream_t stream[2];
  void *cpu[2] = {};
  cudaEvent_t start[2] = {};
  cudaEvent_t stop[2] = {};

  OR_SKIP_AND_RETURN(cuda_reset_device(cuda), "");

  cpu[0] = aligned_alloc(pageSize, bytes);
  cpu[1] = aligned_alloc(pageSize, bytes);
  defer(free(cpu[0]));
  defer(free(cpu[1]));
  if (bytes && (!cpu[0] || !cpu[1])) {
    state.SkipWithError(NAME2 " failed to allocate host memory");
    return;
  }
  std::memset(cpu[0], 0xDEADBEEF, bytes);
  std::memset(cpu[1], 0xDEADBEEF, bytes);

  OR_SKIP_AND_RETURN(cudaHostRegister(cpu[0], bytes, cudaHostRegisterMapped),
                     "");
  OR_SKIP_AND_RETURN(cudaHostRegister(cpu[1], bytes, cudaHostRegisterMapped),
                     "");
  defer(cudaHostUnregister(cpu[0]));
  defer(cudaHostUnregister(cpu[1]));

  // get a valid device pointers
  void *dptr[2] = {};
  cudaDeviceProp prop;
  OR_SKIP_AND_RETURN(cudaGetDeviceProperties(&prop, cuda), "");
#if __CUDACC_VER_MAJOR__ >= 9
  if (prop.canUseHostPointerForRegisteredMem) {
#else
  if (false) {
#endif
    dptr[0] = cpu[0];
    dptr[1] = cpu[1];
  } else {
    OR_SKIP_AND_RETURN(cudaHostGetDevicePointer(&dptr[0], cpu[0], 0), "");
    OR_SKIP_AND_RETURN(cudaHostGetDevicePointer(&dptr[1], cpu[1], 0), "");
  }

  // create streams
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream[0]), "");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream[1]), "");
  defer(cudaStreamDestroy(stream[0]));
  defer(cudaStreamDestroy(stream[1]));

  OR_SKIP_AND_RETURN(cudaEventCreate(&start[0]), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&start[1]), "");
  defer(cudaEventDestroy(start[0]));
  defer(cudaEventDestroy(start[1]));
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop[0]), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop[1]), "");
  defer(cudaEventDestroy(stop[0]));
  defer(cudaEventDestroy(stop[1]));

  // compute kernel params that are half of GPU
  int rdDimGrid;
  int wrDimGrid;
  {
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
    GpuWrFunc, wrDimBlock, 0);
  wrDimGrid = maxActiveBlocks * prop.multiProcessorCount;
  }
  {
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
      GpuRdFunc, rdDimBlock, 0);
    rdDimGrid = maxActiveBlocks * prop.multiProcessorCount;
    }

  clock_t cycles = 4096;
  for (auto _ : state) {
  restart_iteration:

    // launch the busy-wait kernel
    NAME::busy_wait<<<1, 1, 0, stream[0]>>>(nullptr, cycles);

    // set up the copies
    OR_SKIP_AND_BREAK(cudaEventRecord(start[0], stream[0]),
                      ""); // stream 0 start
    gpu_read<rdDimBlock><<<rdDimGrid, rdDimBlock, 0, stream[0]>>>(
        (int32_t *)dptr[0], (int32_t *)nullptr, bytes); // stream 0 copy
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream[1], start[0], 0),
                      ""); // stream 1 wait for stream 0 to start
    OR_SKIP_AND_BREAK(cudaEventRecord(start[1], stream[1]),
                      ""); // stream 1 start
    gpu_write<wrDimBlock><<<wrDimGrid, wrDimBlock, 0, stream[1]>>>(
        (int32_t *)dptr[1], bytes);                             // stream 1 copy
    OR_SKIP_AND_BREAK(cudaEventRecord(stop[1], stream[1]), ""); // stream 1 stop
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream[0], stop[1], 0),
                      ""); // stream 0 wait for stream 1 to stop
    OR_SKIP_AND_BREAK(cudaEventRecord(stop[0], stream[0]), ""); // stream 0 stop

    // wait for streams to finish work, and restart iteration if needed
    cudaError_t err = cudaEventQuery(start[0]);
    if (cudaSuccess == err) {
      // busy-wait is done, so it was too slow.
      OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream[0]), "");
      OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream[1]), "");
      cycles *= 2;
      goto restart_iteration;
    } else if (cudaErrorNotReady == err) {
      // kernel was long enough
    } else {
      OR_SKIP_AND_BREAK(err, "errored while waiting for kernel");
    }

    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop[0]), "");
    float millis = 0;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start[0], stop[0]), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes * 2;
  state.counters["numa"] = numa;
  state.counters["cuda"] = cuda;
};

static void registerer() {

  for (auto cuda : unique_cuda_device_ids()) {
    for (auto numa : numa::ids()) {

      std::string name(NAME2);
      name += "/" + std::to_string(numa) + "/" + std::to_string(cuda);
      benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_HostGPU, numa,
                                   cuda)
          ->ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME2);
