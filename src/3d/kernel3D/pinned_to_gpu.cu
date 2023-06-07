#include <sstream>

#include "scope/scope.hpp"

#include "../args.hpp"

#define NAME "Comm_3d_kernel3D_PinnedToGPU"

__global__ void Comm_3d_kernel3D_PinnedToGPU_kernel(void *__restrict__ dst,
                                                    const void *__restrict__ src,
                                                    const cudaExtent allocExtent,
                                                    const cudaExtent copyExtent,
                                                    const size_t elemSize) {

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int zi = tz; zi < copyExtent.depth; zi += blockDim.z * gridDim.z) {
    for (unsigned int yi = ty; yi < copyExtent.height; yi += blockDim.y * gridDim.y) {
      for (unsigned int xi = tx; xi < copyExtent.width; xi += blockDim.x * gridDim.x) {
        unsigned int ii = zi * allocExtent.height * allocExtent.width + yi * allocExtent.width + xi;
        if (4 == elemSize) {
          uint32_t *pDst = reinterpret_cast<uint32_t *>(dst);
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          uint32_t v = pSrc[ii];
          pDst[ii] = v;
        } else if (8 == elemSize) {
          uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
          const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
          pDst[ii] = pSrc[ii];
        } else {
          char *pDst = reinterpret_cast<char *>(dst);
          const char *pSrc = reinterpret_cast<const char *>(src);
          memcpy(&pDst[ii * elemSize], &pSrc[ii * elemSize], elemSize);
        }
      }
    }
  }
}

inline int64_t nextPowerOfTwo(int64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

inline dim3 make_block_dim(const cudaExtent extent, int64_t threads) {
  assert(threads <= 1024);
  dim3 ret;
  ret.x = std::min(threads, nextPowerOfTwo(extent.width));
  threads /= ret.x;
  ret.y = std::min(threads, nextPowerOfTwo(extent.height));
  threads /= ret.y;
  ret.z = std::min(threads, nextPowerOfTwo(extent.depth));

  // if z is too big, push down into y
  if (ret.z > 64) {
    ret.y *= (ret.z / 64);
    ret.z = 64;
  }

  assert(ret.x <= 1024);
  assert(ret.y <= 1024);
  assert(ret.z <= 64); // maximum
  assert(ret.x * ret.y * ret.z <= 1024);
  return ret;
}

auto Comm_3d_kernel3D_PinnedToGPU = [](benchmark::State &state, const int numaId,
                                       const int cudaId) {

#if defined(SCOPE_USE_NVTX)
  {
    std::stringstream name;
    name << NAME << "/" << numaId << "/" << cudaId << "/" << state.range(0) << "/" << state.range(1)
         << "/" << state.range(2);
    nvtxRangePush(name.str().c_str());
  }
#endif // SCOPE_USE_NVTX

  // bind to CPU & reset device
  numa::ScopedBind binder(numaId);
  OR_SKIP_AND_RETURN(scope::cuda_reset_device(cudaId), "failed to reset GPU");

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
  allocExt.width = 768 * 4; // how many bytes in a row
  allocExt.height = 768;    // how many rows in a plane
  allocExt.depth = 768;

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
  OR_SKIP_AND_RETURN(
      cudaHostRegister(src.ptr, allocBytes, cudaHostRegisterPortable | cudaHostRegisterMapped),
      "cudaHostRegister()");
  std::memset(src.ptr, 0, allocBytes);

  // 4 bytes per thread
  size_t elemSize = 4;
  // convert alloc and copy extent to be in terms of elemSize chunks
  assert(allocExt.width % elemSize == 0);
  allocExt.width /= elemSize;
  assert(copyExt.width % elemSize == 0);
  copyExt.width /= elemSize;

  dim3 blockDim = make_block_dim(copyExt, 512);
  dim3 gridDim;
  gridDim.x = (copyExt.width + blockDim.x - 1) / blockDim.x;
  gridDim.y = (copyExt.height + blockDim.y - 1) / blockDim.y;
  gridDim.z = (copyExt.depth + blockDim.z - 1) / blockDim.z;

  for (auto _ : state) {
    // Start copy
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream), NAME " failed to record start event");
    Comm_3d_kernel3D_PinnedToGPU_kernel<<<gridDim, blockDim, 0, stream>>>(
        dst.ptr, src.ptr, allocExt, copyExt, elemSize);
    OR_SKIP_AND_BREAK(cudaGetLastError(), "kernel");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream), NAME " failed to record stop event");

    // Wait for all copies to finish
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), NAME " failed to synchronize");

    // Get the transfer time
    float millis;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop),
                      NAME " failed to compute elapsed tiume");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(copyBytes));
  state.counters["bytes"] = copyBytes;
  state.counters["numaId"] = numaId;
  state.counters["cudaId"] = cudaId;
  state.counters["dbx"] = blockDim.x;
  state.counters["dby"] = blockDim.y;
  state.counters["dbz"] = blockDim.z;
  state.counters["dgx"] = gridDim.x;
  state.counters["dgy"] = gridDim.y;
  state.counters["dgz"] = gridDim.x;

  OR_SKIP_AND_RETURN(cudaHostUnregister(src.ptr), "cudaHostUnregister");
  free(src.ptr);
  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "cudaEventDestroy");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaFree(dst.ptr), NAME "failed to cudaFree");

#if defined(SCOPE_USE_NVTX)
  nvtxRangePop();
#endif
};

static void registerer() {
  std::vector<MemorySpace> cudaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::cuda_device);
  std::vector<MemorySpace> numaSpaces = scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (const auto &cudaSpace : cudaSpaces) {
    for (const auto &numaSpace : numaSpaces) {

      const int cudaId = cudaSpace.device_id();
      const int numaId = numaSpace.numa_id();

      const std::string name =
          std::string(NAME) + "/" + std::to_string(numaId) + "/" + std::to_string(cudaId);
      benchmark::RegisterBenchmark(name.c_str(), Comm_3d_kernel3D_PinnedToGPU, numaId, cudaId)
          ->ASTAROTH_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
