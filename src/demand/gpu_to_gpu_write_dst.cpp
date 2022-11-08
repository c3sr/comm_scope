#include "args.hpp"
#include "demand.hpp"

#define NAME "Comm_hipManaged_GPUToGPUWriteDst"

constexpr int CACHE_LINE_SIZE = 64;

auto Comm_hipManaged_GPUToGPUWriteDst = [](benchmark::State &state, const int src_gpu,
                                  const int dst_gpu) {


  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  UnaryData data = setup<Kind::GPUToGPU>(state, NAME, bytes, src_gpu, dst_gpu);
  defer(hipFree(data.ptr));
  defer(hipEventDestroy(data.start));
  defer(hipEventDestroy(data.stop));
  if (data.error) {
    return;
  }


  for (auto _ : state) {
    prep_iteration<Kind::GPUToGPU>(data.ptr, bytes, src_gpu, dst_gpu);
    if (PRINT_IF_ERROR(hipGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }

    hipEventRecord(data.start);
    gpu_write<<<2048, 256>>>(data.ptr, bytes, CACHE_LINE_SIZE);
    hipEventRecord(data.stop);
    if (PRINT_IF_ERROR(hipEventSynchronize(data.stop))) {
      state.SkipWithError(NAME " failed to do kernels");
      return;
    }

    float millis = 0;
    if (PRINT_IF_ERROR(hipEventElapsedTime(&millis, data.start, data.stop))) {
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
    for (size_t j = i + 1; j < hips.size(); ++j) {
      int src_gpu = hips[i].device_id();
      int dst_gpu = hips[j].device_id();
      std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) +
                         "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_hipManaged_GPUToGPUWriteDst,
                                   src_gpu, dst_gpu)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);


