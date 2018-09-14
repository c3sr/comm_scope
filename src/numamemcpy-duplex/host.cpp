#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1

#include <cassert>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "init/numa.hpp"
#include "utils/numa.hpp"

#define NAME "Comm/Duplex/NUMAMemcpy/Host" 

#define OR_SKIP(stmt, msg) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(msg); \
    return; \
  }

static void Comm_Duplex_NUMAMemcpy_Host(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const int numa   = FLAG(numa_ids)[0];
  const int gpu    = FLAG(cuda_device_ids)[0];

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP(utils::cuda_reset_device(gpu), NAME " failed to reset CUDA device");
  OR_SKIP(cudaSetDevice(gpu), NAME " failed to set device");
  numa_bind_node(numa); 

  // There are two copies, one gpu -> host, one host -> gpu


  // Create One stream per copy
  cudaStream_t stream1, stream2;
  std::vector<cudaStream_t> streams = {stream1, stream2};
  OR_SKIP(cudaStreamCreate(&streams[0]), NAME "failed to create stream");
  OR_SKIP(cudaStreamCreate(&streams[1]), NAME "failed to create stream");

  // Start and stop events for each copy
  cudaEvent_t start1, start2, stop1, stop2;
  std::vector<cudaEvent_t> starts = {start1, start2};
  std::vector<cudaEvent_t> stops = {stop1, stop2};
  OR_SKIP(cudaEventCreate(&starts[0]), NAME " failed to create event");
  OR_SKIP(cudaEventCreate(&starts[1]), NAME " failed to create event");
  OR_SKIP(cudaEventCreate(&stops[0] ), NAME " failed to create event");
  OR_SKIP(cudaEventCreate(&stops[1] ), NAME " failed to create event");

  // Source and destination for each copy
  std::vector<char *> srcs, dsts;

  // bookkeeping for cuda frees and hos frees
  std::vector<void *> cuda_frees, frees;

  // create a source and destination allocation for first copy: gpu ->host
  char *ptr;
  OR_SKIP(cudaMalloc(&ptr, bytes), NAME " failed to perform cudaMalloc");
  OR_SKIP(cudaMemset(ptr, 0, bytes), NAME " failed to perform src cudaMemset");
  srcs.push_back(ptr);
  cuda_frees.push_back(ptr);

  ptr = static_cast<char*>(aligned_alloc(65536, bytes));
  std::memset(ptr, 0, bytes);
  dsts.push_back(ptr);
  frees.push_back(ptr);

  // create a dst and src allocation for the second copy host -> gpu
  OR_SKIP(cudaMalloc(&ptr, bytes), NAME " failed to perform cudaMalloc");
  OR_SKIP(cudaMemset(ptr, 0, bytes), NAME " failed to perform src cudaMemset");
  dsts.push_back(ptr);
  cuda_frees.push_back(ptr);

  ptr = static_cast<char*>(aligned_alloc(65536, bytes));
  std::memset(ptr, 0, bytes);
  srcs.push_back(ptr);
  frees.push_back(ptr);

  assert(starts.size() == stops.size());
  assert(streams.size() == starts.size());
  assert(srcs.size() == dsts.size());
  assert(streams.size() == srcs.size());

  for (auto _ : state) {

    // Start all copies
    for (size_t i = 0; i < streams.size(); ++i) {
      auto start = starts[i];
      auto stop = stops[i];
      auto stream = streams[i];
      auto src = srcs[i];
      auto dst = dsts[i];
      OR_SKIP(cudaEventRecord(start, stream), NAME " failed to record start event");
      OR_SKIP(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream), NAME " failed to start cudaMemcpyAsync");
      OR_SKIP(cudaEventRecord(stop, stream), NAME " failed to record stop event");
    }

    // Wait for all copies to finish
    for (auto s : stops) {
      OR_SKIP(cudaEventSynchronize(s), NAME " failed to synchronize");
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for (const auto stop : stops) {
        float millis;
        OR_SKIP(cudaEventElapsedTime(&millis, start, stop), NAME " failed to compute elapsed tiume");
        maxMillis = std::max(millis, maxMillis);
      }
    }
    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters.insert({{"bytes", bytes}});
  state.counters["cuda_id"] = gpu;
  state.counters["numa_id"] = numa;

  float stopSum = 0;
  float startSum = 0;
  for ( const auto stream : streams ){

        float startTime1, startTime2, stopTime1, stopTime2;
        OR_SKIP(cudaEventElapsedTime(&startTime1, starts[0], starts[1]), NAME " failed to compare start times");
        OR_SKIP(cudaEventElapsedTime(&startTime2, starts[1], starts[0]), NAME " failed to compare start times");
        OR_SKIP(cudaEventElapsedTime(&stopTime1, stops[0],  stops[1]),  NAME " failed to compare stop times");
        OR_SKIP(cudaEventElapsedTime(&stopTime2, stops[1],  stops[0]),  NAME " failed to compare stop times");
        startSum += std::max(startTime1, startTime2);
        stopSum += std::max(stopTime1, stopTime2);
  }

  state.counters["avg_start_spread"] = startSum/state.iterations();
  state.counters["avg_stop_spread"] = stopSum/state.iterations();

  for (auto p : cuda_frees) {
    cudaFree(p);
  }
  for (auto p : frees) {
    free(p);
  }

}

BENCHMARK(Comm_Duplex_NUMAMemcpy_Host)->SMALL_ARGS()->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1
