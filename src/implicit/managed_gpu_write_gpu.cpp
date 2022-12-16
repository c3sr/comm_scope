#include "args.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_managed_GPUWrGPU"

auto Comm_implicit_managed_GPUWrGPU = [](benchmark::State &state, const int src_gpu,
                                  const int dst_gpu) {

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  void *ptr;
  hipEvent_t start;
  hipEvent_t stop;

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_gpu))) {
      state.SkipWithError(NAME " failed to reset hip src device");
      return;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(dst_gpu))) {
      state.SkipWithError(NAME " failed to reset hip dst device");
      return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(src_gpu))) {
      state.SkipWithError(NAME " failed to set hip src device");
      return;
  }

  if (PRINT_IF_ERROR(hipMallocManaged(&ptr, bytes))) {
      state.SkipWithError(NAME " failed to perform hipMallocManaged");
      return;
  }

  if (PRINT_IF_ERROR(hipMemset(ptr, 1, bytes))) {
      state.SkipWithError(NAME " failed to perform hipMemset");
      return;
  }

  if (PRINT_IF_ERROR(hipEventCreate(&start))) {
      state.SkipWithError(NAME " failed to create start event");
      return;
  }
  defer(hipEventDestroy(start));

  if (PRINT_IF_ERROR(hipEventCreate(&stop))) {
      state.SkipWithError(NAME " failed to create stop event");
      return;
  }
  defer(hipEventDestroy(stop));      

  for (auto _ : state) {
    if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, src_gpu))) {
      state.SkipWithError(NAME "failed to prefetch");
      return; 
    }
    if (PRINT_IF_ERROR(hipDeviceSynchronize())) {
      state.SkipWithError(NAME "failed to sync");
      return; 
    }

    hipEventRecord(start);
    gpu_write<uint64_t><<<2048, 256>>>(ptr, bytes);
    hipEventRecord(stop);
    if (PRINT_IF_ERROR(hipEventSynchronize(stop))) {
      state.SkipWithError(NAME " failed to do kernels");
      return;
    }

    float millis = 0;
    if (PRINT_IF_ERROR(hipEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
};

static void registerer() {

  const std::vector<Device> hips = scope::system::hip_devices();

  for (size_t i = 0; i < hips.size(); ++i) {
    for (size_t j = i; j < hips.size(); ++j) {
      int src_gpu = hips[i].device_id();
      int dst_gpu = hips[j].device_id();
      std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) +
                         "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_implicit_managed_GPUWrGPU,
                                   src_gpu, dst_gpu)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);


