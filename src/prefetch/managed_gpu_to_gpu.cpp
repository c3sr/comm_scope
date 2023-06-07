#include "args.hpp"
#include "scope/scope.hpp"

#define NAME "Comm_prefetch_managed_GPUToGPU"

auto Comm_prefetch_managed_GPUToGPU = [](benchmark::State &state,
                                         const int src_gpu, const int dst_gpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset hip src device");
    return;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(dst_gpu))) {
    state.SkipWithError(NAME " failed to reset hip src device");
    return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set hip dst device");
    return;
  }

  char *ptr = nullptr;
  if (PRINT_IF_ERROR(hipMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMallocManaged");
    return;
  }
  defer(hipFree(ptr));

  if (PRINT_IF_ERROR(hipMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  hipEvent_t start, stop;
  if (PRINT_IF_ERROR(hipEventCreate(&start))) {
    state.SkipWithError(NAME " failed to create event");
    return;
  }
  defer(hipEventDestroy(start));

  if (PRINT_IF_ERROR(hipEventCreate(&stop))) {
    state.SkipWithError(NAME " failed to create event");
    return;
  }
  defer(hipEventDestroy(stop));

  for (auto _ : state) {
    hipMemPrefetchAsync(ptr, bytes, src_gpu);
    hipSetDevice(src_gpu);
    hipDeviceSynchronize();
    hipSetDevice(dst_gpu);
    hipDeviceSynchronize();
    if (PRINT_IF_ERROR(hipGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }

    hipEventRecord(start);
    if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, dst_gpu))) {
      state.SkipWithError(NAME " failed prefetch");
      break;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

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

  std::vector<Device> hips = scope::system::hip_devices();

  for (size_t i = 0; i < hips.size(); ++i) {
    for (size_t j = i; j < hips.size(); ++j) {
      auto src_gpu = hips[i].device_id();
      auto dst_gpu = hips[j].device_id();
      std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) +
                         "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_prefetch_managed_GPUToGPU,
                                   src_gpu, dst_gpu)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
