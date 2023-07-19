#if __CUDACC_VER_MAJOR__ >= 8

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_Demand_Duplex_GPUGPU"

template <bool NOOP = false>
__global__ void gpu_write(char *ptr, const size_t count, const size_t stride) {
  if (NOOP) {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_Demand_Duplex_GPUGPU = [](benchmark::State &state, const int gpu0,
                                    const int gpu1) {
  if (gpu0 == gpu1) {
    state.SkipWithError(NAME " requuires two different GPUs");
    return;
  }

  const size_t pageSize = page_size();
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  cudaStream_t streams[2] = {nullptr};
  char *ptrs[2] = {nullptr};

  // start and end events in gpu0's stream. end0 will not be recorded until
  // after end1
  cudaEvent_t start = nullptr;
  cudaEvent_t end0 = nullptr;

  // end event
  cudaEvent_t end1 = nullptr;

  // initialize data structures for device `dev`
#define INIT(dev)                                                              \
  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu##dev), "failed to reset");          \
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu##dev), "failed to set");                \
  OR_SKIP_AND_RETURN(cudaStreamCreate(&streams[dev]),                          \
                     "failed to create stream");                               \
  OR_SKIP_AND_RETURN(cudaMallocManaged(&ptrs[dev], bytes),                     \
                     "failed to cudaMallocManaged");                           \
  OR_SKIP_AND_RETURN(cudaMemset(ptrs[dev], 0, bytes), "failed to cudaMemset")

  INIT(0);
  INIT(1);

  // record the "pimary" events in the stream associated with gpu0
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "failed to set gpu0");
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "failed to create start event")
  OR_SKIP_AND_RETURN(cudaEventCreate(&end0), "failed to create end0")

  // record the end of the transfer task running on gpu1
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "failed to set gpu1");
  OR_SKIP_AND_RETURN(cudaEventCreate(&end1), "failed to create end1")

  for (auto _ : state) {
    // prefetch data to the source device before the transfers
    OR_SKIP_AND_BREAK(cudaMemPrefetchAsync(ptrs[0], bytes, gpu1, streams[0]), "");
    OR_SKIP_AND_BREAK(cudaMemPrefetchAsync(ptrs[1], bytes, gpu0, streams[1]), "");
    OR_SKIP_AND_BREAK(cudaStreamSynchronize(streams[0]), "");
    OR_SKIP_AND_BREAK(cudaStreamSynchronize(streams[1]), "");

    OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), "");
    OR_SKIP_AND_BREAK(cudaEventRecord(start, streams[0]), "");
    gpu_write<<<256, 256, 0, streams[0]>>>(ptrs[0], bytes, pageSize);
    OR_SKIP_AND_BREAK(cudaGetLastError(), "");
    OR_SKIP_AND_BREAK(cudaSetDevice(gpu1), "");
    gpu_write<<<256, 256, 0, streams[1]>>>(ptrs[1], bytes, pageSize);
    OR_SKIP_AND_BREAK(cudaGetLastError(), "");
    OR_SKIP_AND_BREAK(cudaEventRecord(end1, streams[1]), "");
    OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), "");
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(streams[0], end1, 0 /*must be 0*/), "");
    OR_SKIP_AND_BREAK(cudaEventRecord(end0, streams[0]), "");

    // once stream 0 is finished, we can compute the elapsed time
    OR_SKIP_AND_BREAK(cudaStreamSynchronize(streams[0]), "");
    float millis = 0;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, end0), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "");
  OR_SKIP_AND_RETURN(cudaEventDestroy(end0), "");
  OR_SKIP_AND_RETURN(cudaEventDestroy(end1), "");

  for (auto s : streams) {
    OR_SKIP_AND_RETURN(cudaStreamDestroy(s), "");
  }

  for (auto p : ptrs) {
    OR_SKIP_AND_RETURN(cudaFree(p), "");
  }
};

static void registerer() {
  for (size_t i : scope::system::cuda_devices()) {
    for (size_t j : scope::system::cuda_devices()) {
      if (i < j) {
        std::string name = std::string(NAME) + "/" + std::to_string(i) + "/" +
                           std::to_string(j);
        benchmark::RegisterBenchmark(name.c_str(), Comm_Demand_Duplex_GPUGPU, i,
                                     j)
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
