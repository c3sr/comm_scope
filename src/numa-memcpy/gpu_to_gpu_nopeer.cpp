#if USE_NUMA == 1

#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "init/numa.hpp"
#include "utils/numa.hpp"

#define NAME "Comm_NUMAMemcpy_GPUToGPU"

auto Comm_NUMAMemcpy_GPUToGPU = [](benchmark::State &state,
    const int numa_id,
    const int src_gpu,
    const int dst_gpu) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " no NUMA control available");
    return;
  }

  if (num_gpus() < 2) {
    state.SkipWithError(NAME " requires at least two GPUs");
    return;
  }
  
  if (src_gpu == dst_gpu) {
    state.SkipWithError(NAME " requires two different GPUs");
    return;
  }

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

  numa_bind_node(numa_id);

  char *src = nullptr;
  char *dst = nullptr;

  if (PRINT_IF_ERROR(utils::cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(dst_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(src_gpu))) {
    state.SkipWithError(NAME " failed to set src device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(src));
  if (PRINT_IF_ERROR(cudaMemset(src, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }
  cudaError_t err = cudaDeviceDisablePeerAccess(dst_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set dst device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));
  if (PRINT_IF_ERROR(cudaMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform dst cudaMemset");
    return;
  }
  err = cudaDeviceDisablePeerAccess(src_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {

    cudaEventRecord(start, NULL);
    const auto cuda_err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
  state.counters["numa_id"] = numa_id;

  // re-enable NUMA scheduling
  numa_bind_node(-1);

  // re-enable peer access
  err = cudaSetDevice(src_gpu);
  err = cudaDeviceEnablePeerAccess(dst_gpu, 0);
  // docs say should return cudaErrorInvalidDevice, actually returns cudaErrorPeerAccessUnsupported?
  if (cudaSuccess != err && cudaErrorInvalidDevice != err && cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }
  err = cudaSetDevice(dst_gpu);
  err = cudaDeviceEnablePeerAccess(src_gpu, 0);
  if (cudaSuccess != err && cudaErrorInvalidDevice != err && cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }

  numa_bind_node(-1);
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i <  unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i + 1; j < unique_cuda_device_ids().size(); ++j) {
      auto src_gpu = unique_cuda_device_ids()[i];
      auto dst_gpu = unique_cuda_device_ids()[j];
      for (auto numa_id : unique_numa_ids()) {
        name = std::string(NAME) 
             + "/" + std::to_string(numa_id) 
             + "/" + std::to_string(src_gpu) 
             + "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_NUMAMemcpy_GPUToGPU, numa_id, src_gpu, dst_gpu)->SMALL_ARGS()->UseManualTime();
      }
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);
#endif // USE_NUMA == 1
