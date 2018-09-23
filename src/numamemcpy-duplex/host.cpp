#if USE_NUMA == 1

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
#include "transfer.hpp"

#define NAME "Comm_Duplex_NUMAMemcpy_Host" 

#define OR_SKIP(stmt, msg) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(msg); \
    return; \
  }

auto Comm_Duplex_NUMAMemcpy_Host = [](benchmark::State &state, std::vector<CudaMemcpyConfig*> transfers) {
  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  // reset all GPUs that participate in any transfer
  for (auto config : transfers) {
    config->reset_gpu();
  }

  // create allocations, events, and streams
  for (auto config : transfers) {
    OR_SKIP(config->init(bytes), NAME " failed to initialize transfer");
  }

  for (auto _ : state) {
    // Start all copies
    for (const auto config : transfers) {
      auto start = config->start_;
      auto stop = config->stop_;
      auto stream = config->stream_;
      auto src = config->src_;
      auto dst = config->dst_;
      OR_SKIP(cudaEventRecord(start, stream), NAME " failed to record start event");
      OR_SKIP(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream), NAME " failed to start cudaMemcpyAsync");
      OR_SKIP(cudaEventRecord(stop, stream), NAME " failed to record stop event");
    }

    // Wait for all copies to finish
    for (const auto config : transfers) {
      auto stop = config->stop_;
      OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize");
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto config_start : transfers) {
      for (const auto config_stop : transfers) {
        auto start = config_start->start_;
        auto stop = config_stop->stop_;
        float millis;
        OR_SKIP(cudaEventElapsedTime(&millis, start, stop), NAME " failed to compute elapsed tiume");
        maxMillis = std::max(millis, maxMillis);
      }
    }
    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;

  for (auto config : transfers) {
    OR_SKIP(config->fini(), NAME " failed to release transfer");
  }
};

static void registerer() {

  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : unique_numa_ids()) {
      std::vector<CudaMemcpyConfig*> transfers;
      transfers.push_back(new PageableCopyConfig(cudaMemcpyHostToDevice, numa_id, cuda_id));
      transfers.push_back(new PageableCopyConfig(cudaMemcpyDeviceToHost, cuda_id, numa_id));
      std::string name = std::string(NAME) 
                       + "/" + std::to_string(numa_id) 
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_Duplex_NUMAMemcpy_Host, transfers)->SMALL_ARGS()->UseManualTime();
    }
  }
  
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // USE_NUMA == 1
