#include "args.hpp"
#include "demand.hpp"

#define NAME "Comm_hipManaged_GPUToHostMt"

constexpr int CACHE_LINE_SIZE = 64;

auto Comm_hipManaged_GPUToHostMt = [](benchmark::State &state, const int src_gpu,
                                  const int dst_numa, const int num_threads) {


  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  UnaryData data = setup<Kind::GPUToHost>(state, NAME, bytes, src_gpu, dst_numa);
  defer(hipFree(data.ptr));
  defer(hipEventDestroy(data.start));
  defer(hipEventDestroy(data.stop));
  if (data.error) {
    return;
  }

  std::vector<std::thread> workers(num_threads);
  std::vector<scope::time_point> starts(num_threads);
  std::vector<scope::time_point> stops(num_threads);

  for (auto _ : state) {
    prep_iteration<Kind::GPUToHost>(data.ptr, bytes, src_gpu, dst_numa);
    if (PRINT_IF_ERROR(hipGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }

    // Create all threads
    for (int i = 0; i < num_threads; ++i) {
      workers[i] =
          std::thread(cpu_write, &data.ptr[i * bytes / num_threads],
                      bytes / num_threads, CACHE_LINE_SIZE, &starts[i], &stops[i]);
    }

    // unleash threads
    {
      std::unique_lock<std::mutex> lk(m);
      ready = true;
      cv.notify_all();
    }

    for (auto &w : workers) {
      w.join();
    }
    ready = false;


    double maxElapsed = 0;
    for (const auto threadStart : starts) {
      for (const auto threadStop : stops) {
        auto threadElapsed =
            scope::duration(threadStop - threadStart).count();
        maxElapsed = std::max(maxElapsed, threadElapsed);
      }
    }
    // std::cerr << maxElapsed << "\n";

    state.SetIterationTime(maxElapsed);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_numa"] = dst_numa;
};

static void registerer() {

  std::vector<MemorySpace> hipSpaces = scope::system::memory_spaces(MemorySpace::Kind::hip_device);
  std::vector<MemorySpace> numaSpaces = scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (auto num_threads : {1, 2, 4, 6, 8, 10}) {
    for (const MemorySpace &ns : numaSpaces) {
      for (const MemorySpace &hs : hipSpaces) {
        auto src_gpu = hs.device_id();
        auto dst_numa = ns.numa_id();
        if (numa::can_execute_in_node(dst_numa)) {
          std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) +
                            "/" + std::to_string(dst_numa) + "/" + std::to_string(num_threads);
          benchmark::RegisterBenchmark(name.c_str(), Comm_hipManaged_GPUToHostMt,
                                      src_gpu, dst_numa, num_threads)
              ->MT_ARGS()
              ->UseManualTime()
              ->MinTime(0.005);
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);


