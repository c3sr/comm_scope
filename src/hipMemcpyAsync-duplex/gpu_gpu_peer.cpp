
#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_Duplex_GPUGPUPeer"

auto Comm_cudaMemcpyAsync_Duplex_GPUGPUPeer = [](benchmark::State &state,
                                                 const int gpu0,
                                                 const int gpu1) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu0), "failed to reset CUDA device");
  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu1), "failed to reset CUDA device");

  // There are two copies, one gpu0 -> gpu1, one gpu1 -> gpu0

  // Create One stream per copy
  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  std::vector<cudaStream_t> streams = {stream1, stream2};
  OR_SKIP_AND_RETURN(cudaStreamCreate(&streams[0]), "failed to create stream");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&streams[1]), "failed to create stream");

  // Start and stop events for each copy
  cudaEvent_t start1 = nullptr;
  cudaEvent_t start2 = nullptr;
  cudaEvent_t stop1 = nullptr;
  cudaEvent_t stop2 = nullptr;
  std::vector<cudaEvent_t> starts = {start1, start2};
  std::vector<cudaEvent_t> stops = {stop1, stop2};
  OR_SKIP_AND_RETURN(cudaEventCreate(&starts[0]), "failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&starts[1]), "failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stops[0]), "failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stops[1]), "failed to create event");

  // Source and destination for each copy
  std::vector<char *> srcs, dsts;

  // create a source and destination allocation for first copy

  // allocate on gpu0 and enable peer access
  char *ptr;
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "failed to perform cudaMalloc");
  srcs.push_back(ptr);
  OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes),
                     "failed to perform src cudaMemset");
  cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError("failed to ensure peer access");
    return;
  }

  // allocate on gpu1 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes),
                     "failed to perform src cudaMemset");
  dsts.push_back(ptr);
  err = cudaDeviceEnablePeerAccess(gpu0, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError("failed to ensure peer access");
    return;
  }

  // create a source and destination for second copy
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes),
                     "failed to perform src cudaMemset");
  srcs.push_back(ptr);

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes),
                     "failed to perform src cudaMemset");
  dsts.push_back(ptr);

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
      OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                        NAME " failed to record start event");
      OR_SKIP_AND_BREAK(
          cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
          NAME " failed to start cudaMemcpyAsync");
      OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream),
                        NAME " failed to record stop event");
    }

    // Wait for all copies to finish
    for (auto s : stops) {
      OR_SKIP_AND_BREAK(cudaEventSynchronize(s), NAME " failed to synchronize");
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for (const auto stop : stops) {
        float millis;
        OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop),
                          NAME " failed to compute elapsed tiume");
        maxMillis = std::max(millis, maxMillis);
      }
    }
    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  for (auto src : srcs) {
    OR_SKIP_AND_RETURN(cudaFree(src), "failed to free");
  }
  for (auto dst : dsts) {
    OR_SKIP_AND_RETURN(cudaFree(dst), "failed to free");
  }
};

static void registerer() {
  std::string name;

  const std::vector<Device> cudas = scope::system::cuda_devices();

  for (size_t i = 0; i < cudas.size(); ++i) {
    for (size_t j = i + 1; j < cudas.size(); ++j) {
      auto gpu0 = cudas[i];
      auto gpu1 = cudas[j];
      int ok1, ok2;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok1, gpu0, gpu1)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok2, gpu1, gpu0))) {
        if (ok1 && ok2) {
          name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
                 std::to_string(gpu1);
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_cudaMemcpyAsync_Duplex_GPUGPUPeer, gpu0, gpu1)
              ->G2G_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
