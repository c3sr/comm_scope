#include <sstream>

#include "scope/scope.hpp"

#include "../args.hpp"

#define NAME "Comm_3d_cudaMemcpy2DAsync_GPUToPinned"

auto Comm_3d_cudaMemcpy2DAsync_GPUToPinned = [](benchmark::State &state,
                                              const int numaId,
                                              const int cudaId) {

#if SCOPE_USE_NVTX == 1
  {
    std::stringstream name;
    name << NAME << "/" << numaId << "/" << cudaId << "/" << state.range(0)
         << "/" << state.range(1) << "/" << state.range(2);
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
  allocExt.width  = 768*4;  // how many bytes in a row
  allocExt.height = 768; // how many rows in a plane
  allocExt.depth  = 768;

  cudaPitchedPtr src, dst;

  // allocate on cudaId.
  OR_SKIP_AND_RETURN(cudaSetDevice(cudaId), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&src, allocExt), "failed to perform cudaMalloc3D");
  allocExt.width = src.pitch; // cudaMalloc3D may adjust pitch for alignment
  const size_t allocBytes = allocExt.width * allocExt.height * allocExt.depth;
  OR_SKIP_AND_RETURN(cudaMemset3D(src, 0, allocExt), "failed to perform src cudaMemset");

  // allocate on CPU.
  dst.ptr = aligned_alloc(page_size(), allocBytes);
  dst.pitch = src.pitch;
  dst.xsize = src.xsize;
  dst.ysize = src.ysize;
  OR_SKIP_AND_RETURN(cudaHostRegister(dst.ptr, allocBytes, cudaHostRegisterPortable),
          "cudaHostRegister()");
  std::memset(dst.ptr, 0, allocBytes);

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream),
                 " failed to record start event");
    // use a bunch of 2D copies to do the 3D copy
    for (size_t z = 0; z < copyExt.depth; ++z) {
      void *srcP = (char *)src.ptr + allocExt.width * allocExt.height * z;
      void *dstP = (char *)dst.ptr + allocExt.width * allocExt.height * z;

      OR_SKIP_AND_BREAK(cudaMemcpy2DAsync(dstP, allocExt.width, srcP, allocExt.width,
                                     copyExt.width, copyExt.height,
                                     cudaMemcpyDeviceToHost, stream),
                   NAME " failed to start 2D copy");
    }
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream), " failed to record stop event");

    // Wait for all copies to finish
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "failed to synchronize");

    // Get the transfer time
    float millis;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop),
                 "failed to compute elapsed time");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(copyBytes));
  state.counters["bytes"] = copyBytes;
  state.counters["numaId"] = numaId;
  state.counters["cudaId"] = cudaId;

  OR_SKIP_AND_RETURN(cudaHostUnregister(dst.ptr), "cudaHostUnregister");
  free(dst.ptr);
  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(src.ptr), NAME "failed to cudaFree");
  numa::bind_node(-1);

#if SCOPE_USE_NVTX == 1
  nvtxRangePop();
#endif
};

static void registerer() {
  std::string name;
  for (auto cudaId : unique_cuda_device_ids()) {
    for (auto numaId : numa::ids()) {

      name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
             std::to_string(cudaId);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_3d_cudaMemcpy2DAsync_GPUToPinned, numaId, cudaId)
          ->ASTAROTH_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
