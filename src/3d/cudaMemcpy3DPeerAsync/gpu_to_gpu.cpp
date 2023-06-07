#include <sstream>

#include "scope/scope.hpp"

#include "../args.hpp"

#define NAME "Comm_3d_cudaMemcpy3DPeerAsync_GPUToGPU"

auto Comm_3d_cudaMemcpy3DPeerAsync_GPUToGPU = [](benchmark::State &state,
                                                 const int gpu0,
                                                 const int gpu1) {

#if defined(SCOPE_USE_NVTX)
  {
    std::stringstream name;
    name << NAME << "/" << gpu0 << "/" << gpu1 << "/" << state.range(0) << "/"
         << state.range(1) << "/" << state.range(2);
    nvtxRangePush(name.str().c_str());
  }
#endif

  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu0),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu1),
                     NAME " failed to reset CUDA device");

  // Create One stream per copy
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");

  // Start and stop events for each copy
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " failed to create event");

  // target size to transfer
  cudaExtent copyExt;
  copyExt.width = static_cast<size_t>(state.range(0));
  copyExt.height = static_cast<size_t>(state.range(1));
  copyExt.depth = static_cast<size_t>(state.range(2));
  const size_t copyBytes = copyExt.width * copyExt.height * copyExt.depth;

  // properties of the allocation
  cudaExtent allocExt;
  allocExt.width = 768 * 4; // how many bytes in a row
  allocExt.height = 768;    // how many rows in a plane
  allocExt.depth = 768;

  cudaPitchedPtr src, dst;

  // allocate on gpu0 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&src, allocExt),
                     NAME " failed to perform cudaMalloc3D");
  allocExt.width = src.pitch;
  OR_SKIP_AND_RETURN(cudaMemset3D(src, 0, allocExt),
                     NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // allocate on gpu1 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&dst, allocExt),
                     NAME " failed to perform cudaMalloc3D");
  OR_SKIP_AND_RETURN(cudaMemset3D(dst, 0, allocExt),
                     NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

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
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                      NAME " failed to record start event");
    OR_SKIP_AND_BREAK(cudaMemcpy3DPeerAsync(&params, stream),
                      NAME " failed to start cudaMemcpy3DPeerAsync");
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

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(copyBytes));
  state.counters["bytes"] = copyBytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(src.ptr), NAME "failed to cudaFree");
  OR_SKIP_AND_RETURN(cudaFree(dst.ptr), NAME "failed to cudaFree");

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
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_3d_cudaMemcpy3DPeerAsync_GPUToGPU, gpu0, gpu1)
              ->ASTAROTH_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
