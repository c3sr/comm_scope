#include "args.hpp"
#include "gpu_to_gpu.hpp"

#define NAME "Comm_hipManaged_GPUToGPUWriteDst"

constexpr int CACHE_LINE_SIZE = 64;

/* use one thread from each warp to write a 0 to each stride
*/
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
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_hipManaged_GPUToGPUWriteDst = [](benchmark::State &state, const int src_gpu,
                                  const int dst_gpu) {


  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  Data data = setup(state, NAME, bytes, src_gpu, dst_gpu);
  defer(hipFree(data.ptr));
  defer(hipEventDestroy(data.start));
  defer(hipEventDestroy(data.stop));
  if (data.error) {
    return;
  }


  for (auto _ : state) {
    prep_iteration(data.ptr, bytes, src_gpu, dst_gpu);
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

  for (size_t i = 0; i < scope::system::hip_device_ids().size(); ++i) {
    for (size_t j = i + 1; j < scope::system::hip_device_ids().size(); ++j) {
      auto src_gpu = scope::system::hip_device_ids()[i];
      auto dst_gpu = scope::system::hip_device_ids()[j];
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


