#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_hipMemcpyAsync_GPUToGPU"

auto Comm_hipMemcpyAsync_GPUToGPU = [](benchmark::State &state, const int srcId,
                                    const int dstId) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  if (PRINT_IF_ERROR(scope::hip_reset_device(srcId))) {
    state.SkipWithError(NAME " failed to reset HIP device");
    return;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(dstId))) {
    state.SkipWithError(NAME " failed to reset HIP device");
    return;
  }



  void *src = nullptr;
  void *dst = nullptr;

  if (PRINT_IF_ERROR(hipSetDevice(srcId))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(hipMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to hipMalloc");
    return;
  }
  defer(hipFree(src));
  if (PRINT_IF_ERROR(hipMemset(src, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(dstId))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(hipMalloc(&dst, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMalloc");
    return;
  }
  defer(hipFree(dst));
  if (PRINT_IF_ERROR(hipMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  hipEvent_t start, stop;
  PRINT_IF_ERROR(hipEventCreate(&start));
  PRINT_IF_ERROR(hipEventCreate(&stop));
  defer(hipEventDestroy(start));
  defer(hipEventDestroy(stop));

  for (auto _ : state) {
    hipSetDevice(srcId);
    hipMemset(src, 0, bytes);
    hipSetDevice(dstId);
    hipMemset(dst, 0, bytes);
    hipDeviceSynchronize();
    if (PRINT_IF_ERROR(hipGetLastError())) {
      state.SkipWithError(NAME " failed to reset benchmark state");
      break;
    }

    hipEventRecord(start, NULL);
    hipError_t err =
        hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    state.PauseTiming();

    if (PRINT_IF_ERROR(err)) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(hipEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_id"] = srcId;
  state.counters["dst_id"] = dstId;
};

static void registerer() {
  LOG(trace, NAME " registerer...");

  std::vector<MemorySpace> hipSpaces = scope::system::memory_spaces(MemorySpace::Kind::hip_device);

  std::string name;
  for (const auto &srcSpace : hipSpaces) {
    for (const auto &dstSpace : hipSpaces) {

      const int srcId = srcSpace.device_id();
      const int dstId = dstSpace.device_id();

      name = std::string(NAME) + "/" + std::to_string(srcId) + "/" +
            std::to_string(dstId);
      benchmark::RegisterBenchmark(name.c_str(), Comm_hipMemcpyAsync_GPUToGPU,
                                  srcId, dstId)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);


