#include <sstream>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_chunk_pull"

typedef int read_t;

/* chunkFill should be lte 32
*/
static __global__ void Comm_chunk_pull_kernel(read_t *__restrict__ src,
                                              const int chunkSize,
                                              const int chunkFill,
                                              const int n, // number of chunks
                                              read_t * __restrict__ flag) {
  const int li = threadIdx.x % 32; // lane index
  const int wi = threadIdx.x / 32; // warp index
  const int bd = blockDim.x / 32;  // dimension of block in warps

  // assign one warp to each chunk
  for (int i = bd *blockIdx.x + wi; i < n; i += gridDim.x * bd) {
    if (li < chunkFill) {
      read_t t;
      do_not_optimize(t = src[i * chunkSize + li]);
      if (flag) {
        *flag = t;
      }
    }
  }
}

auto Comm_chunk_pull = [](benchmark::State &state, const int gpu0,
                          const int gpu1) {

#if SCOPE_USE_NVTX == 1
  {
    std::stringstream name;
    name << NAME << "/" << gpu0 << "/" << gpu1 << "/" << state.range(0) << "/"
         << state.range(1);
    nvtxRangePush(name.str().c_str());
  }
#endif

  // in read_t, not bytes
  const int chunkSize = state.range(0);
  const int chunkFill = state.range(1);

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                     NAME " failed to reset CUDA device");

  // create stream on dst gpu (pull)
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to create stream");
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");

  // Start and stop events on dst gpu (pull)
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " failed to create event");

  // fixed number of loads regardless of stride
  read_t *src = nullptr;
  const size_t bytes = 1024ull * 1024ull * 1024ull * 2;
  const size_t size = bytes / sizeof(read_t); // number of read_t
  const int n = size / chunkSize;             // number of chunks in allocation
  const int dimBlock = 512;
  int dimGrid = (n + (dimBlock / 32) - 1) / (dimBlock / 32);
  dimGrid = min(dimGrid, 1024);


  // allocate on gpu0 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src, bytes),
                     NAME " failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(src, 0, bytes),
                     NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // enable peer access from gpu1
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // run pull kernel on dst device
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME " unable to set pull device");

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                      NAME " failed to record start event");

    Comm_chunk_pull_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        src, chunkSize, chunkFill, n, nullptr);
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
                          sizeof(read_t));
  state.counters["ld-bytes"] = n * chunkFill * sizeof(read_t);
  state.counters["ld-count"] = n * chunkFill;
  state.counters["ld-size"] = sizeof(read_t);
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
  state.counters["chunkSize"] = chunkSize;
  state.counters["chunkFill"] = chunkFill;
  state.counters["chunkCont"] = n;
  state.counters["dimgrid"] = dimGrid;

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(src), "cudaFree");

#if SCOPE_USE_NVTX == 1
  nvtxRangePop();
#endif
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i; j < unique_cuda_device_ids().size(); ++j) {
      auto gpu0 = unique_cuda_device_ids()[i];
      auto gpu1 = unique_cuda_device_ids()[j];
      int ok1, ok2;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok1, gpu0, gpu1)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok2, gpu1, gpu0))) {
        if ((ok1 && ok2) || i == j) {
          name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
                 std::to_string(gpu1);
          benchmark::RegisterBenchmark(name.c_str(), Comm_chunk_pull, gpu0,
                                       gpu1)
              ->CHUNK_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
