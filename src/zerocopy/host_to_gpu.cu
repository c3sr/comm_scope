#include "scope/scope.hpp"

#include "args.hpp"
#include "kernels.hu"

#define NAME "Comm_ZeroCopy_HostToGPU"

enum class ShouldFlush { No, Yes };

const char * to_string(ShouldFlush flush) {
  switch (flush) {
    case ShouldFlush::No: return "";
    case ShouldFlush::Yes: return "_flush";
  }
  exit(EXIT_FAILURE);
}

auto Comm_ZeroCopy_HostToGPU = [](benchmark::State &state, const int src_numa,
                                  const int dst_cuda, const ShouldFlush flush) {
  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(src_numa);

  OR_SKIP_AND_RETURN(cuda_reset_device(dst_cuda), "");
  OR_SKIP_AND_RETURN(cudaSetDevice(dst_cuda), "");

  void *ptr = aligned_alloc(pageSize, bytes);
  defer(free(ptr));
  if (!ptr && bytes) {
    state.SkipWithError(NAME " failed to allocate host memory");
    return;
  }
  std::memset(ptr, 0xDEADBEEF, bytes);

  OR_SKIP_AND_RETURN(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped), "");
  defer(cudaHostUnregister(ptr));

  // get a valid device pointer
  void *dptr;
  cudaDeviceProp prop;
  OR_SKIP_AND_RETURN(cudaGetDeviceProperties(&prop, dst_cuda), "");

#if __CUDACC_VER_MAJOR__ >= 9
  if (prop.canUseHostPointerForRegisteredMem) {
#else
  if (false) {
#endif
    dptr = ptr;
  } else {
    OR_SKIP_AND_RETURN(cudaHostGetDevicePointer(&dptr, ptr, 0), "");
  }

  cudaEvent_t start, stop;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "");
  defer(cudaEventDestroy(start));
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "");
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {

    if (ShouldFlush::Yes == flush) {
      flush_all(ptr, bytes);
    }

    OR_SKIP_AND_BREAK(cudaEventRecord(start), "");
    constexpr unsigned GD = 256;
    constexpr unsigned BD = 256;
    gpu_read<BD><<<GD, BD>>>((int32_t *)dptr, (int32_t *)nullptr, bytes);

    OR_SKIP_AND_BREAK(cudaEventRecord(stop), "");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "");

    float millis = 0;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_numa"] = src_numa;
  state.counters["dst_cuda"] = dst_cuda;
};

static void registerer() {
  for (auto flush : {ShouldFlush::No, ShouldFlush::Yes}) {
    for (auto cuda_id : unique_cuda_device_ids()) {
      for (auto numa_id : numa::ids()) {
        std::string name = std::string(NAME) + to_string(flush) + "/" + std::to_string(numa_id) +
                           "/" + std::to_string(cuda_id);
        benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_HostToGPU,
                                     numa_id, cuda_id, flush)
            ->ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
