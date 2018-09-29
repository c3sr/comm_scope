#if CUDA_VERSION_MAJOR >= 8

#include <thread>
#include <condition_variable>
#include <memory>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"
#include "init/numa.hpp"

#define NAME "Comm_UM_Coherence_GPUToHostMt"

std::condition_variable cv;
std::mutex m;
volatile bool ready = false;

static void cpu_write(char *ptr, const size_t n, const size_t stride) {
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, []{return ready;});
  }

  for (size_t i = 0; i < n; i += stride) {
    benchmark::DoNotOptimize(ptr[i] = 0);
  }
}

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
  size_t wx             = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_UM_Coherence_GPUToHost_Mt = [] (benchmark::State &state,
#if USE_NUMA
const int numa_id,
#endif // USE_NUMA
const int cuda_id,
const int num_threads) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  if (PRINT_IF_ERROR(utils::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }

  char *ptr = nullptr;
  if (PRINT_IF_ERROR(cudaMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMallocManaged");
    return;
  }
  defer(cudaFree(ptr));

  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point_t;
  std::vector<time_point_t> starts(num_threads);
  std::vector<time_point_t> stops(num_threads);
  std::vector<std::thread> workers(num_threads);


  for (auto _ : state) {
    state.PauseTiming();

    cudaError_t err = cudaMemPrefetchAsync(ptr, bytes, cuda_id);
    if (cudaErrorInvalidDevice == err) {
      gpu_write<<<256, 256>>>(ptr, bytes, pageSize);
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    // Create all threads
    for (int i = 0; i < num_threads; ++i) {
      workers[i] = std::thread(cpu_write, &ptr[i * bytes / num_threads], bytes / num_threads, pageSize);
    }

    // unleash threads
    {
      std::lock_guard<std::mutex> lk(m);
      ready = true;
    }

    //notify threads they can go
    auto start = std::chrono::high_resolution_clock::now();
    cv.notify_all();

    for (auto &w: workers) {
      w.join();
    }
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif // USE_NUMA

#if USE_NUMA
  // reset to run on any node
  numa_bind_node(-1);
#endif
};

static void registerer() {
  for (auto num_threads : {1, 2, 4, 8, 12, 16}) {
    for (auto cuda_id : unique_cuda_device_ids()) {
#if USE_NUMA
      for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
        std::string name = std::string(NAME)
#if USE_NUMA 
                         + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                         + "/" + std::to_string(cuda_id)
                         + "/" + std::to_string(num_threads);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Coherence_GPUToHost_Mt,
#if USE_NUMA
          numa_id,
#endif // USE_NUMA
          cuda_id, num_threads)->SMALL_ARGS()->UseManualTime();
#if USE_NUMA
      }
#endif // USE_NUMA
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // CUDA_VERSION_MAJOR >= 8
