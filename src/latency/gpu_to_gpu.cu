#if CUDA_VERSION_MAJOR >= 8

#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"

#define NAME "Comm_UM_Latency_GPUToGPU"

template <bool NOOP = false>
__global__ void gpu_traverse(size_t *ptr, const size_t steps) {

  if (NOOP) {
    return;
  }
  size_t next = 0;
  for (int i = 0; i < steps; ++i) {
    next = ptr[next];
  }
  ptr[next] = 1;
}

auto Comm_UM_Latency_GPUToGPU = [](benchmark::State &state, const int src_gpu, const int dst_gpu) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (src_gpu == dst_gpu) {
    state.SkipWithError(NAME "src and dst GPU should be different");
    return;
  }

  const size_t steps = state.range(0);
  const size_t stride = 65536 * 2;
  const size_t bytes  = sizeof(size_t) * (steps + 1) * stride;

  if (PRINT_IF_ERROR(utils::cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset src device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset dst device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
    return;
  }

  size_t *ptr = nullptr;
  if (PRINT_IF_ERROR(cudaMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMallocManaged");
    return;
  }
  defer(cudaFree(ptr));

  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }
  // set up stride pattern
  for (size_t i = 0; i < steps; ++i) {
    ptr[i * stride] = (i + 1) * stride;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(src_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA src device");
    return;
  }
  if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
    state.SkipWithError(NAME " failed to synchronize");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
    return;
  }
  cudaEvent_t start, stop;
  if (PRINT_IF_ERROR(cudaEventCreate(&start))) {
    state.SkipWithError(NAME " failed to create start event");
    return;
  }
  defer(cudaEventDestroy(start));

  if (PRINT_IF_ERROR(cudaEventCreate(&stop))) {
    state.SkipWithError(NAME " failed to create end event");
    return;
  }
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, src_gpu))) {
      state.SkipWithError(NAME " failed to prefetch to src");
      return;
    }
    if (PRINT_IF_ERROR(cudaSetDevice(src_gpu))) {
      state.SkipWithError(NAME " failed to set CUDA src device");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }
    if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
      state.SkipWithError(NAME " failed to set dst device");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    cudaEventRecord(start);
    gpu_traverse<<<1, 1>>>(ptr, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
    state.counters["strides"] = steps;
    state.counters["src_gpu"] = src_gpu;
    state.counters["dst_gpu"] = dst_gpu;
  }
};

static void registerer() {
  for (size_t i = 0; i <  unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i + 1; j < unique_cuda_device_ids().size(); ++j) {
      auto src_gpu = unique_cuda_device_ids()[i];
      auto dst_gpu = unique_cuda_device_ids()[j];
      std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) + "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Latency_GPUToGPU, src_gpu, dst_gpu)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);

#endif // CUDA_VERSION_MAJOR >= 8