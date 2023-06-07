#include <sstream>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_chunk_push"

typedef int write_t;

static __global__ void Comm_chunk_push_kernel(write_t *__restrict__ dst,
                                              const int chunkSize,
                                              const int chunkFill,
                                              const int n // number of chunks
) {
  // use one warp for each chunk
  assert(chunkFill <= 32);

  const int li = threadIdx.x % 32; // lane index
  const int wi = threadIdx.x / 32; // warp index
  const int bd = blockDim.x / 32;  // dimension of block in warps

  // assign one warp to each chunk
  for (int i = bd * blockIdx.x + wi; i < n; i += gridDim.x * bd) {
    if (li < chunkFill) {
      dst[i * chunkSize + li] = i;
    }
  }
}

auto Comm_chunk_push = [](benchmark::State &state, const int gpu0,
                          const int gpu1) {

#if defined(SCOPE_USE_NVTX)
  {
    std::stringstream name;
    name << NAME << "/" << gpu0 << "/" << gpu1 << "/" << state.range(0) << "/"
         << state.range(1);
    nvtxRangePush(name.str().c_str());
  }
#endif

  // in write_t, not bytes
  const int chunkSize = state.range(0);
  const int chunkFill = state.range(1);

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                     NAME " failed to reset CUDA device");

  // create stream, start and stop events on src gpu on src gpu
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to create stream");
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " failed to create event");

  // fixed number of loads regardless of stride
  write_t *ptr = nullptr;
  const size_t bytes = 1024ull * 1024ull * 1024ull * 2;
  const size_t size = bytes / sizeof(write_t); // number of write_t
  const int n = size / chunkSize;              // number of chunks in allocation
  const int dimBlock = 512;
  const int dimGrid = (n + (dimBlock / 32) - 1) / (dimBlock / 32);

  // gpu0 enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");

  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // gpu1 alloc and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");

  OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes),
                     NAME " failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes),
                     NAME " failed to perform dst cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // run push kernel on src device
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME " unable to set pull device");

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                      NAME " failed to record start event");

    Comm_chunk_push_kernel<<<dimGrid, dimBlock, 0, stream>>>(ptr, chunkSize,
                                                             chunkFill, n);
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream),
                      NAME " failed to record stop event");

    // Wait for all copies to finish
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop),
                      NAME " failed to synchronize");

    // Get the transfer time
    float millis;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop),
                      NAME " failed to compute elapsed tiume");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * n * chunkFill *
                          sizeof(write_t));
  state.counters["st-bytes"] = n * chunkFill * sizeof(write_t);
  state.counters["st-count"] = n * chunkFill;
  state.counters["st-size"] = sizeof(write_t);
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
  state.counters["alloc"] = bytes;
  state.counters["chunkSize"] = chunkSize;
  state.counters["chunkFill"] = chunkFill;
  state.counters["chunkCont"] = n;

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(ptr), "cudaFree");

#if defined(SCOPE_USE_NVTX)
  nvtxRangePop();
#endif
};

static void registerer() {
  std::string name;
  const std::vector<MemorySpace> cudaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::cuda_device);

  for (const auto &space0 : cudaSpaces) {
    for (const auto &space1 : cudaSpaces) {

      auto gpu0 = space0.device_id();
      auto gpu1 = space1.device_id();
      int ok1, ok2;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok1, gpu0, gpu1)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok2, gpu1, gpu0))) {
        if ((ok1 && ok2) || gpu0 == gpu1) {
          name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
                 std::to_string(gpu1);
          benchmark::RegisterBenchmark(name.c_str(), Comm_chunk_push, gpu0,
                                       gpu1)
              ->CHUNK_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
