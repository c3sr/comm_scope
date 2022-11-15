#include "args.hpp"
#include "implicit.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_mapped_GPUWrGPU"

constexpr int CACHE_LINE_SIZE = 64;

auto Comm_implicit_mapped_GPUWrGPU = [](benchmark::State &state,
                                        const int wr_gpu, const int own_gpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  char *ptr = nullptr;
  hipEvent_t start;
  hipEvent_t stop;

  if (PRINT_IF_ERROR(scope::hip_reset_device(own_gpu))) {
    state.SkipWithError(NAME " failed to reset hip owner");
    return;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(wr_gpu))) {
    state.SkipWithError(NAME " failed to reset hip src device");
    return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(own_gpu))) {
    state.SkipWithError(NAME " failed to set hip dst device");
    return;
  }

  if (PRINT_IF_ERROR(hipMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMalloc");
    return;
  }
  defer(hipFree(ptr));

  if (PRINT_IF_ERROR(hipMemset(ptr, 0, bytes))) {
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

    if (PRINT_IF_ERROR(hipSetDevice(own_gpu))) {
      state.SkipWithError(NAME " failed to set hip dst device");
      return;
    }

    if (PRINT_IF_ERROR(hipMemset(ptr, 0, bytes))) {
      state.SkipWithError(NAME " failed to perform hipMemset");
      return;
    }

    if (PRINT_IF_ERROR(hipSetDevice(wr_gpu))) {
      state.SkipWithError(NAME " failed to set hip dst device");
      return;
    }

    hipEventRecord(start);
    gpu_write<<<2048, 256>>>(ptr, bytes, CACHE_LINE_SIZE);
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
  state.counters["own_gpu"] = own_gpu;
  state.counters["wr_gpu"] = wr_gpu;
};

static void registerer() {

  std::vector<MemorySpace> hipSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::hip_device);

  for (size_t i = 0; i < hipSpaces.size(); ++i) {
    for (size_t j = i + 1; j < hipSpaces.size(); ++j) {
      int wr_gpu = hipSpaces[i].device_id();
      int own_gpu = hipSpaces[j].device_id();
      std::string name = std::string(NAME) + "/" + std::to_string(wr_gpu) +
                         "/" + std::to_string(own_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_implicit_mapped_GPUWrGPU,
                                   wr_gpu, own_gpu)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
