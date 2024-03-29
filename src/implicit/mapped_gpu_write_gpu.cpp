#include "args.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_mapped_GPUWrGPU"

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

  if (PRINT_IF_ERROR(hipSetDevice(wr_gpu))) {
    state.SkipWithError(NAME " failed to set hip wr_gpu device");
    return;
  }

  // need to create events on device that will execute kernel?
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

  // TODO: cleanup
  if (wr_gpu != own_gpu) {
    hipError_t err = hipDeviceEnablePeerAccess(own_gpu, 0);
    if (hipErrorPeerAccessAlreadyEnabled != err) {
      if (PRINT_IF_ERROR(err)) {
        state.SkipWithError(NAME " wr_gpu enable peer access to own_gpu");
        return;
      }
    }
  }

  if (PRINT_IF_ERROR(hipSetDevice(own_gpu))) {
    state.SkipWithError(NAME " failed to set hip own_gpu device");
    return;
  }

  // TODO: cleanup
  if (wr_gpu != own_gpu) {
    hipError_t err = hipDeviceEnablePeerAccess(wr_gpu, 0);
    if (hipErrorPeerAccessAlreadyEnabled != err) {
      if (PRINT_IF_ERROR(err)) {
        state.SkipWithError(NAME " own_gpu enable peer access to wr_gpu");
        return;
      }
    }
  }

  if (PRINT_IF_ERROR(hipMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMalloc");
    return;
  }
  defer(hipFree(ptr));

  if (PRINT_IF_ERROR(hipMemset(ptr, 1, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  for (auto _ : state) {
    if (PRINT_IF_ERROR(hipSetDevice(own_gpu))) {
      state.SkipWithError(NAME " failed to set hip dst device");
      return;
    }

    if (PRINT_IF_ERROR(hipMemset(ptr, 1, bytes))) {
      state.SkipWithError(NAME " failed to perform hipMemset");
      return;
    }

    if (PRINT_IF_ERROR(hipSetDevice(wr_gpu))) {
      state.SkipWithError(NAME " failed to set hip dst device");
      return;
    }

    hipError_t e1 = hipEventRecord(start);
    gpu_write<uint64_t><<<2048, 256>>>(ptr, bytes);
    hipError_t e2 = hipEventRecord(stop);
    if (PRINT_IF_ERROR(e1)) {
      state.SkipWithError(NAME " failed to record start event");
      return;
    }
    if (PRINT_IF_ERROR(e2)) {
      state.SkipWithError(NAME " failed to record stop event");
      return;
    }
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
    for (size_t j = i; j < hipSpaces.size(); ++j) {
      int wr_gpu = hipSpaces[i].device_id();
      int own_gpu = hipSpaces[j].device_id();

      int can;
      if (i == j) {
        can = true; // can access self
      } else {
        HIP_RUNTIME(hipDeviceCanAccessPeer(&can, wr_gpu, own_gpu));
      }
      if (can) {
        std::string name = std::string(NAME) + "/" + std::to_string(wr_gpu) +
                           "/" + std::to_string(own_gpu);
        benchmark::RegisterBenchmark(
            name.c_str(), Comm_implicit_mapped_GPUWrGPU, wr_gpu, own_gpu)
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
