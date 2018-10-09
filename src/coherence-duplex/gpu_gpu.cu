#if CUDA_VERSION_MAJOR >= 8

#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm_Coherence_Duplex_GPUGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

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
  size_t wx             = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_Coherence_Duplex_GPUGPU = [](benchmark::State &state, const int gpu0, const int gpu1) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (gpu0 == gpu1) {
    state.SkipWithError(NAME " requuires two different GPUs");
    return;
  }

  const size_t pageSize = page_size();
  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));
  cudaStream_t streams[2];
  char *ptrs[2] = {nullptr};



#define INIT(dev) \
  OR_SKIP(utils::cuda_reset_device(gpu##dev)); \
  OR_SKIP(cudaSetDevice(gpu##dev)); \
  OR_SKIP(cudaStreamCreate(&streams[dev])); \
  OR_SKIP(cudaMallocManaged(&ptrs[dev], bytes)); \
  OR_SKIP(cudaMemset(ptrs[dev], 0, bytes))

  INIT(0);
  INIT(1);

  for (auto _ : state) {
    state.PauseTiming();

    OR_SKIP(cudaMemPrefetchAsync(ptrs[0], bytes, gpu1, streams[0]));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[1], bytes, gpu0, streams[1]));
    OR_SKIP(cudaSetDevice(gpu0));
    OR_SKIP(cudaDeviceSynchronize());
    OR_SKIP(cudaSetDevice(gpu1));
    OR_SKIP(cudaDeviceSynchronize());

    OR_SKIP(cudaSetDevice(gpu0));
    state.ResumeTiming();
    gpu_write<<<256, 256, 0, streams[0]>>>(ptrs[0], bytes, pageSize);
    OR_SKIP(cudaSetDevice(gpu1));
    gpu_write<<<256, 256, 0, streams[1]>>>(ptrs[1], bytes, pageSize);
    OR_SKIP(cudaGetLastError());
    

    OR_SKIP(cudaStreamSynchronize(streams[0]));
    OR_SKIP(cudaStreamSynchronize(streams[1]));
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  for (auto s : streams) {
    OR_SKIP(cudaStreamDestroy(s));
  }

  for(auto p : ptrs) {
    OR_SKIP(cudaFree(p));
  }

};

static void registerer() {
  for (size_t i : unique_cuda_device_ids()) {
    for (size_t j : unique_cuda_device_ids()) {
      if (i < j) {
        std::string name = std::string(NAME) + "/" + std::to_string(i) + "/" + std::to_string(j);
        benchmark::RegisterBenchmark(name.c_str(), Comm_Coherence_Duplex_GPUGPU, i, j)->SMALL_ARGS()->UseRealTime();
      }
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // CUDA_VERSION_MAJOR >= 8
