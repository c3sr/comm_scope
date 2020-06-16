/* Measure the runtime cost of cudaMemcpy3DPeerAsync
 */

#include "sysbench/sysbench.hpp"

#define NAME "Comm_cudart_cudaMemcpy3DPeerAsync"

auto Comm_cudart_cudaMemcpy3DPeerAsync = [](benchmark::State &state,
                                            const int gpu0, const int gpu1) {
  OR_SKIP(cuda_reset_device(gpu0), NAME " failed to reset CUDA device");
  OR_SKIP(cuda_reset_device(gpu1), NAME " failed to reset CUDA device");

  // Create One stream per copy
  cudaStream_t stream = nullptr;
  OR_SKIP(cudaStreamCreate(&stream), NAME "failed to create stream");

  // fixed-size transfer
  cudaExtent copyExt;
  copyExt.width = 130;
  copyExt.height = 170;
  copyExt.depth = 190;
  const size_t copyBytes = copyExt.width * copyExt.height * copyExt.depth;

  // properties of the allocation
  cudaExtent allocExt;
  allocExt.width = copyExt.width;
  allocExt.height = copyExt.height;
  allocExt.depth = copyExt.depth;

  cudaPitchedPtr src, dst;

  // allocate on gpu0 and enable peer access
  OR_SKIP(cudaSetDevice(gpu0), NAME "failed to set device");
  OR_SKIP(cudaMalloc3D(&src, allocExt), NAME " failed to perform cudaMalloc3D");
  allocExt.width = src.pitch;
  OR_SKIP(cudaMemset3D(src, 0, allocExt),
          NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // allocate on gpu1 and enable peer access
  OR_SKIP(cudaSetDevice(gpu1), NAME "failed to set device");
  OR_SKIP(cudaMalloc3D(&dst, allocExt), NAME " failed to perform cudaMalloc3D");
  OR_SKIP(cudaMemset3D(dst, 0, allocExt),
          NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // set up copy parameters
  cudaMemcpy3DPeerParms params;
  params.dstArray = 0; // providing dstPtr
  params.srcArray = 0; // providing srcPtr
  params.dstDevice = gpu1;
  params.srcDevice = gpu0;
  params.dstPos = make_cudaPos(0, 0, 0);
  params.srcPos = make_cudaPos(0, 0, 0);
  params.dstPtr = dst;
  params.srcPtr = src;
  params.extent = copyExt;

  for (auto _ : state) {
    // Start copy
    cudaError_t err = cudaMemcpy3DPeerAsync(&params, stream);

    // measure one copy at a time
    state.PauseTiming();
    OR_SKIP_AND_BREAK(err,
                      NAME " failed to start cudaMemcpy3DPeerAsync");

    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream),
                      NAME " failed to synchronize");
    state.ResumeTiming();
  }

  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  OR_SKIP(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP(cudaFree(src.ptr), NAME "failed to cudaFree");
  OR_SKIP(cudaFree(dst.ptr), NAME "failed to cudaFree");

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
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_cudart_cudaMemcpy3DPeerAsync, gpu0, gpu1)->UseRealTime();
        }
      }
    }
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);
