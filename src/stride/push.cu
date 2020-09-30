#include <sstream>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_stride_push"

typedef int write_t;

static __global__ void Comm_stride_push_kernel(write_t *dst,
                                               const int n, // number of reads
                                               const int stride) {

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    dst[i * stride] = i;
  }
}

auto Comm_stride_push = [](benchmark::State &state, const int gpu0,
                           const int gpu1) {
  const int stride = state.range(0);

#if SCOPE_USE_NVTX == 1
  {
    std::stringstream name;
    name << NAME << "/" << gpu0 << "/" << gpu1 << "/" << state.range(0) << "/"
         << state.range(1) << "/" << state.range(2);
    nvtxRangePush(name.str().c_str());
  }
#endif

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                     NAME " failed to reset CUDA device");

  // fixed number of loads regardless of stride
  write_t *dst = nullptr;
  const size_t bytes = 1024ull * 1024ull * 1024ull * 2;
  const size_t size = bytes / sizeof(write_t);
  const int dimGrid = 512;
  const int dimBlock = 512;
  const int n = size / stride; // number of reads

  // allocate on gpu0 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, bytes),
                     NAME " failed to perform cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMemset(dst, 0, bytes),
                     NAME " failed to perform dst cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // create stream on src gpu (push)
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");

  // Start and stop events on dst gpu (push)
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " failed to create event");

  // enable peer access from gpu1
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // run push kernel on src device
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME " unable to set push device");

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                      NAME " failed to record start event");

    Comm_stride_push_kernel<<<dimGrid, dimBlock, 0, stream>>>(dst, n, stride);
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

  state.SetBytesProcessed(int64_t(state.iterations()) * n * sizeof(write_t));
  state.counters["st-bytes"] = n * sizeof(write_t);
  state.counters["st-count"] = n;
  state.counters["st-size"] = sizeof(write_t);
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
  state.counters["alloc"] = bytes;
  state.counters["st-stride"] = stride * sizeof(write_t);

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(dst), "cudaFree");

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
          benchmark::RegisterBenchmark(name.c_str(), Comm_stride_push, gpu0,
                                       gpu1)
              ->STRIDE_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
