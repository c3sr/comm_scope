#if __CUDACC_VER_MAJOR__ >= 8

#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm_Prefetch_Duplex_GPUGPU"

#define OR_SKIP(stmt)                                                                                                  \
  if (PRINT_IF_ERROR(stmt)) {                                                                                          \
    state.SkipWithError(NAME);                                                                                         \
    return;                                                                                                            \
  }

auto Comm_Prefetch_Duplex_GPUGPU = [](benchmark::State &state, const int gpu0, const int gpu1) {
  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (gpu0 == gpu1) {
    state.SkipWithError(NAME " requires two different GPUs");
    return;
  }

  const size_t pageSize = page_size();
  const auto bytes      = 1ULL << static_cast<size_t>(state.range(0));
  cudaStream_t streams[2];
  char *ptrs[2] = {nullptr};
  // start and end events. stop0 will not be recorded until after stop1
  cudaEvent_t start = nullptr;
  cudaEvent_t stop0 = nullptr;
  cudaEvent_t stop1 = nullptr;

#define INIT(dev)                                                                                                      \
  OR_SKIP(utils::cuda_reset_device(gpu##dev));                                                                         \
  OR_SKIP(cudaSetDevice(gpu##dev));                                                                                    \
  OR_SKIP(cudaStreamCreate(&streams[dev]));                                                                            \
  OR_SKIP(cudaMallocManaged(&ptrs[dev], bytes));                                                                       \
  OR_SKIP(cudaMemset(ptrs[dev], 0, bytes))

  INIT(0);
  INIT(1);

  // record the "pimary" events in the stream associated with gpu0
  OR_SKIP(cudaSetDevice(gpu0));
  OR_SKIP(cudaEventCreate(&start))
  OR_SKIP(cudaEventCreate(&stop0))

  // record the end of the transfer task running on gpu1
  OR_SKIP(cudaSetDevice(gpu1));
  OR_SKIP(cudaEventCreate(&stop1))

  for (auto _ : state) {
    // prefetch data to src and sync
    OR_SKIP(cudaMemPrefetchAsync(ptrs[0], bytes, gpu1, streams[0]));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[1], bytes, gpu0, streams[1]));
    OR_SKIP(cudaStreamSynchronize(streams[0]));
    OR_SKIP(cudaStreamSynchronize(streams[1]));

    OR_SKIP(cudaSetDevice(gpu0));
    OR_SKIP(cudaEventRecord(start, streams[0]));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[0], bytes, gpu0, streams[0]));

    OR_SKIP(cudaSetDevice(gpu1));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[1], bytes, gpu1, streams[1]));
    OR_SKIP(cudaEventRecord(stop1, streams[1]));

    OR_SKIP(cudaSetDevice(gpu0));
    OR_SKIP(cudaStreamWaitEvent(streams[0], stop1, 0 /*must be 0*/));
    OR_SKIP(cudaEventRecord(stop0, streams[0]));

    OR_SKIP(cudaStreamSynchronize(streams[0]));
    float millis = 0;
    OR_SKIP(cudaEventElapsedTime(&millis, start, stop0));
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"]  = gpu0;
  state.counters["gpu1"]  = gpu1;

  OR_SKIP(cudaEventDestroy(start));
  OR_SKIP(cudaEventDestroy(stop0));
  OR_SKIP(cudaEventDestroy(stop1));

  for (auto s : streams) {
    OR_SKIP(cudaStreamDestroy(s));
  }

  for (auto p : ptrs) {
    OR_SKIP(cudaFree(p));
  }
};

static void registerer() {
  for (size_t i : unique_cuda_device_ids()) {
    for (size_t j : unique_cuda_device_ids()) {
      if (i < j) {
        std::string name = std::string(NAME) + "/" + std::to_string(i) + "/" + std::to_string(j);
        benchmark::RegisterBenchmark(name.c_str(), Comm_Prefetch_Duplex_GPUGPU, i, j)->SMALL_ARGS()->UseManualTime();
      }
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
