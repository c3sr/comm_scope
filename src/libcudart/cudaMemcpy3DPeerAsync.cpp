/* Measure the runtime cost of cudaMemcpy3DPeerAsync
 */

#include "scope/scope.hpp"

#define NAME "Comm_cudart_cudaMemcpy3DPeerAsync"

// all threads use the same memcpy params and the same stream
static cudaStream_t gStream;
static cudaMemcpy3DPeerParms gParams;

auto Comm_cudart_cudaMemcpy3DPeerAsync = [](benchmark::State &state,
                                             const int gpu0, const int gpu1,
                                             cudaStream_t &stream,
                                             cudaMemcpy3DPeerParms &params) {
  // have thread 0 set up shared structures
  if (0 == state.thread_index()) {

    OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                       NAME " failed to reset CUDA device");
    OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                       NAME " failed to reset CUDA device");

    // small enough transfer that the runtime cost is larger
    cudaExtent copyExt;
    copyExt.width = 8;
    copyExt.height = 8;
    copyExt.depth = 8;

    // properties of the allocation
    cudaExtent allocExt;
    allocExt.width = copyExt.width;
    allocExt.height = copyExt.height;
    allocExt.depth = copyExt.depth;

    // allocate on gpu0 and enable peer access
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");
    OR_SKIP_AND_RETURN(cudaMalloc3D(&params.srcPtr, allocExt),
                       NAME " failed to perform cudaMalloc3D");
    allocExt.width = params.srcPtr.pitch;
    OR_SKIP_AND_RETURN(cudaMemset3D(params.srcPtr, 0, allocExt),
                       NAME " failed to perform src cudaMemset");
    if (gpu0 != gpu1) {
      cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
      if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
        state.SkipWithError(NAME " failed to ensure peer access");
      }
    }

    // Create a stream shared by all threads
    OR_SKIP_AND_RETURN(cudaStreamCreate(&stream),
                       NAME "failed to create stream");

    // allocate on gpu1 and enable peer access
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");
    OR_SKIP_AND_RETURN(cudaMalloc3D(&params.dstPtr, allocExt),
                       NAME " failed to perform cudaMalloc3D");
    OR_SKIP_AND_RETURN(cudaMemset3D(params.dstPtr, 0, allocExt),
                       NAME " failed to perform src cudaMemset");
    if (gpu0 != gpu1) {
      cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
      if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
        state.SkipWithError(NAME " failed to ensure peer access");
      }
    }

    // set up copy parameters
    params.dstArray = 0; // provided dstPtr
    params.srcArray = 0; // provided srcPtr
    params.dstDevice = gpu1;
    params.srcDevice = gpu0;
    params.dstPos = make_cudaPos(0, 0, 0);
    params.srcPos = make_cudaPos(0, 0, 0);
    params.extent = copyExt;
  }

  cudaError_t err;
  for (auto _ : state) {
    err = cudaMemcpy3DPeerAsync(&params, stream);
  }
  OR_SKIP_AND_RETURN(err, "failed to cudaMemcpy3DPeerAsync");

  // can't stream sync because thread 0 may destroy the stream before thread 1
  // syncs
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "");
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), NAME " failed to synchronize");
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "");
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), NAME " failed to synchronize");
  state.SetItemsProcessed(state.iterations());
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  if (0 == state.thread_index()) {
    OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
    OR_SKIP_AND_RETURN(cudaFree(params.srcPtr.ptr), NAME "failed to cudaFree");
    OR_SKIP_AND_RETURN(cudaFree(params.dstPtr.ptr), NAME "failed to cudaFree");

#if SCOPE_USE_NVTX == 1
    nvtxRangePop();
#endif
  }
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i; j < unique_cuda_device_ids().size(); ++j) {
      for(size_t numThreads = 1; numThreads <= numa::all_cpus().size(); numThreads *= 2) {
      auto gpu0 = unique_cuda_device_ids()[i];
      auto gpu1 = unique_cuda_device_ids()[j];
      int ok1, ok2;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok1, gpu0, gpu1)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok2, gpu1, gpu0))) {
        if ((ok1 && ok2) || i == j) {
          name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
                 std::to_string(gpu1);
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_cudart_cudaMemcpy3DPeerAsync, gpu0, gpu1,
              std::ref(gStream), std::ref(gParams))
              ->Threads(numThreads)
              ->UseRealTime();
        }
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
