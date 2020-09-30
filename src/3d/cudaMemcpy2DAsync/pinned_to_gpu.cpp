#include <sstream>

#include "scope/scope.hpp"

#include "../args.hpp"

#define NAME "Comm_3d_cudaMemcpy2DAsync_PinnedToGPU"

auto Comm_3d_cudaMemcpy2DAsync_PinnedToGPU = [](benchmark::State &state, const int numaId, const int cudaId) {

  #if SCOPE_USE_NVTX == 1
{
  std::stringstream name;
  name << NAME << "/" << numaId << "/" << cudaId  << "/" << state.range(0) << "/" << state.range(1) << "/" << state.range(2);
  nvtxRangePush(name.str().c_str());
}
  #endif // SCOPE_USE_NVTX

  // bind to CPU & reset device
  numa::ScopedBind binder(numaId);
  OR_SKIP_AND_RETURN(cuda_reset_device(cudaId), "failed to reset GPU");

  // stream for async copy
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");

  // Start and stop event for copy
  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " failed to create event");

  // target size to transfer
  cudaExtent copyExt;
  copyExt.width  = static_cast<size_t>(state.range(0));
  copyExt.height = static_cast<size_t>(state.range(1));
  copyExt.depth  = static_cast<size_t>(state.range(2));
  const size_t copyBytes = copyExt.width * copyExt.height * copyExt.depth;

  // properties of the allocation
  cudaExtent allocExt;
  allocExt.width  = 512;  // how many bytes in a row
  allocExt.height = 512; // how many rows in a plane
  allocExt.depth  = 512;

  cudaPitchedPtr src, dst;

  // allocate on cudaId. cudaMalloc3D may adjust the extent to align
  OR_SKIP_AND_RETURN(cudaSetDevice(cudaId), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&dst, allocExt), "failed to perform cudaMalloc3D");
  allocExt.width = dst.pitch;
  const size_t allocBytes = allocExt.width * allocExt.height * allocExt.depth;
  OR_SKIP_AND_RETURN(cudaMemset3D(dst, 0, allocExt), "failed to perform dst cudaMemset");

  // allocate on CPU. 
  src.ptr = aligned_alloc(page_size(), allocBytes);
  src.pitch = dst.pitch;
  src.xsize = dst.xsize;
  src.ysize = dst.ysize;
  OR_SKIP_AND_RETURN(cudaHostRegister(src.ptr, allocBytes, cudaHostRegisterPortable), "cudaHostRegister()");
  std::memset(src.ptr, 0, allocBytes);

  cudaMemcpy3DParms params;
  params.dstArray  = 0; // providing dstPtr
  params.srcArray  = 0; // providing srcPtr
  params.dstPos    = make_cudaPos(0, 0, 0);
  params.srcPos    = make_cudaPos(0, 0, 0);
  params.dstPtr    = dst;
  params.srcPtr    = src;
  params.extent    = copyExt;
  params.kind = cudaMemcpyDefault;

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream), NAME " failed to record start event");
    OR_SKIP_AND_BREAK(cudaMemcpy3DAsync(&params, stream), NAME " failed to start cudaMemcpy3DAsync");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream), NAME " failed to record stop event");

    // Wait for all copies to finish
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), NAME " failed to synchronize");

    // Get the transfer time
    float millis;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), NAME " failed to compute elapsed tiume");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(copyBytes));
  state.counters["bytes"] = copyBytes;
  state.counters["numaId"]  = numaId;
  state.counters["cudaId"]  = cudaId;

  OR_SKIP_AND_RETURN(cudaHostUnregister(src.ptr), "cudaHostUnregister");
  free(src.ptr);
  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(dst.ptr), NAME "failed to cudaFree");

  #if SCOPE_USE_NVTX == 1
  nvtxRangePop();
  #endif
};

static void registerer() {
  std::string name;
for (auto cudaId : unique_cuda_device_ids()) {
    for (auto numaId : numa::ids()) {

          name = std::string(NAME) + "/" + std::to_string(numaId) + "/" + std::to_string(cudaId);
          benchmark::RegisterBenchmark(name.c_str(), Comm_3d_cudaMemcpy2DAsync_PinnedToGPU, numaId, cudaId)
              ->IC_ARGS()
              ->UseManualTime();

    }
  }
}



SCOPE_AFTER_INIT(registerer, NAME);
