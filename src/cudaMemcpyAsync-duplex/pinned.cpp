 #include "scope/scope.hpp"
 
#include "args.hpp"
#include "transfer.hpp"

#define NAME "Comm_cudaMemcpyAsync_Duplex_Pinned" 

auto Comm_cudaMemcpyAsync_Duplex_Pinned = [](benchmark::State &state, std::vector<CudaMemcpyConfig*> transfers, const bool flush) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  // reset all GPUs that participate in any transfer
  for (auto config : transfers) {
    config->reset_gpu();
  }

  // create allocations, events, and streams
  for (auto config : transfers) {
    OR_SKIP_AND_RETURN(config->init(bytes), NAME " failed to initialize transfer");
  }

  for (auto _ : state) {

    // flush caches
    if (flush) {
      for (const auto config : transfers) {
        auto kind = config->kind_;
        if (cudaMemcpyDeviceToHost == kind) {
          flush_all(config->dst_, bytes);
        } else if (cudaMemcpyHostToDevice == kind) {
          flush_all(config->src_, bytes);
        }
      }
    }

    // Start all copies
    for (const auto config : transfers) {
      auto start = config->start_;
      auto stop = config->stop_;
      auto stream = config->stream_;
      auto src = config->src_;
      auto dst = config->dst_;
      OR_SKIP_AND_BREAK(cudaEventRecord(start, stream), NAME " failed to record start event");
      OR_SKIP_AND_BREAK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream), NAME " failed to start cudaMemcpyAsync");
      OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream), NAME " failed to record stop event");
    }

    // Wait for all copies to finish
    for (const auto config : transfers) {
      auto stop = config->stop_;
      OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), NAME " failed to synchronize");
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto config_start : transfers) {
      for (const auto config_stop : transfers) {
        auto start = config_start->start_;
        auto stop = config_stop->stop_;
        float millis;
        OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), NAME " failed to compute elapsed tiume");
        maxMillis = std::max(millis, maxMillis);
      }
    }
    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;

  for (auto config : transfers) {
    OR_SKIP_AND_RETURN(config->fini(), NAME " failed to release transfer");
  }
};

static void registerer() {
  std::string name;
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::ids()) {
      std::vector<CudaMemcpyConfig*> transfers;
      transfers.push_back(new PinnedCopyConfig(cudaMemcpyHostToDevice, numa_id, cuda_id));
      transfers.push_back(new PinnedCopyConfig(cudaMemcpyDeviceToHost, cuda_id, numa_id));
      name = std::string(NAME) 
           + "/" + std::to_string(numa_id) 
           + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudaMemcpyAsync_Duplex_Pinned, transfers, false)->SMALL_ARGS()->UseManualTime();
      name = std::string(NAME) + "_flush"
           + "/" + std::to_string(numa_id) 
           + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudaMemcpyAsync_Duplex_Pinned, transfers, true)->SMALL_ARGS()->UseManualTime();
    }
  }
  
}

SCOPE_AFTER_INIT(registerer, NAME);

