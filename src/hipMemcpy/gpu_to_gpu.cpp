#include <cassert>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_hipMemcpy_GPUToGPU"

auto Comm_hipMemcpy_GPUToGPU = [](benchmark::State &state, const int hipId0, const int hipId1) {

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

  void *src = nullptr;
  void *dst = nullptr;

  OR_SKIP_AND_RETURN(scope::hip_reset_device(hipId0), "failed to reset HIP device");
  OR_SKIP_AND_RETURN(scope::hip_reset_device(hipId1), "failed to reset HIP device");

  // Allocate src
  if (PRINT_IF_ERROR(hipSetDevice(hipId0))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(hipMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(hipFree(src));

  // Allocate device
  if (PRINT_IF_ERROR(hipSetDevice(hipId0))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(hipMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(hipFree(dst));

  // zero src
  if (PRINT_IF_ERROR(hipMemset(src, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  // zero dst
  if (PRINT_IF_ERROR(hipMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  for (auto _ : state) {
    std::memset(dst, 0, bytes);
    auto start = scope::clock::now();
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
    double elapsed = scope::duration(scope::clock::now() - start).count();

    if (PRINT_IF_ERROR(err)) {
      state.SkipWithError(NAME " failed to perform hipMemcpy");
      break;
    }
    state.SetIterationTime(elapsed);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["hip_0"] = hipId0;
  state.counters["hip_1"] = hipId1;
};

static void registerer() {
  std::string name;

  std::vector<MemorySpace> spaces = scope::system::hip_memory_spaces();

  for (const MemorySpace &ms0 : spaces) {
    for (const MemorySpace &ms1 : spaces) {
      const int d0 = ms0.device_id();
      const int d1 = ms1.device_id();
      name = std::string(NAME) + "/" + std::to_string(d0) + "/" + std::to_string(d1);
      benchmark::RegisterBenchmark(name.c_str(), Comm_hipMemcpy_GPUToGPU, d0, d1)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);

