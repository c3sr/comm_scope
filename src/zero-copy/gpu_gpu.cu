#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "zero-copy/args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"
#include "init/numa.hpp"
#include "utils/cache_control.hpp"


#define NAME "Comm_ZeroCopy_GPUGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

typedef enum {
  READ,
  WRITE,
} AccessType;

static std::string to_string(const AccessType &a) {
  if (a == READ) {
    return "_Read";
  } else {
    return "_Write";
  }
}

template <typename write_t>
__global__ void gpu_write(write_t *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(write_t);

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    ptr[i] = 0;
  }
}


template <typename read_t>
__global__ void gpu_read(const read_t *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(read_t);

  __shared__ int32_t s[256];
  int32_t t;

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    t += ptr[i];
  }
  s[threadIdx.x] = t;
  (void) s[threadIdx.x];
}


auto Comm_ZeroCopy_GPUGPU = [](benchmark::State &state, const int gpu0, const int gpu1, const AccessType access_type, const bool duplex) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));

  std::vector<cudaStream_t> streams(duplex ? 2 : 1);
  std::vector<void *>          ptrs(duplex ? 2 : 1, nullptr);

  OR_SKIP(utils::cuda_reset_device(gpu0));
  OR_SKIP(utils::cuda_reset_device(gpu1));

#define RD_INIT(src, dst, op_idx) \
  OR_SKIP(cudaSetDevice(gpu##src)); \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes)); \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes)); \
  OR_SKIP(cudaSetDevice(gpu##dst)); \
  OR_SKIP(cudaStreamCreate(&streams[op_idx])); \
  { \
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu##src, 0); \
    if (cudaErrorPeerAccessAlreadyEnabled != err) { \
      OR_SKIP(err); \
    } \
  }

// code runs on src, and data on dst
#define WR_INIT(src, dst, op_idx) \
  OR_SKIP(cudaSetDevice(gpu##dst)); \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes)); \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes)); \
  OR_SKIP(cudaSetDevice(gpu##src)); \
  OR_SKIP(cudaStreamCreate(&streams[op_idx])); \
  { \
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu##dst, 0); \
    if (cudaErrorPeerAccessAlreadyEnabled != err) { \
      OR_SKIP(err); \
    } \
  }

  if (READ == access_type) {
    RD_INIT(0, 1, 0);
    if (duplex) {
      RD_INIT(1, 0, 1);
    }
  } else {
    WR_INIT(0, 1, 0);
    if (duplex) {
      WR_INIT(1, 0, 1);
    }
  }

  for (auto _ : state) {
    // READ: gpu1 reads from gpu0 (gpu0 is src, gpu1 is dst)
#define READ_ITER(dst, op_idx) \
    { \
    OR_SKIP(cudaSetDevice(gpu##dst)); \
    gpu_read<int32_t><<<256, 256, 0, streams[op_idx]>>>(static_cast<int32_t*>(ptrs[op_idx]), bytes); \
    }
    // WRITE: gpu0 writes to gpu1 (gpu0 is src, gpu1 is dst)
#define WRITE_ITER(src, op_idx) \
    { \
    OR_SKIP(cudaSetDevice(gpu##src)); \
    gpu_write<int32_t><<<256, 256, 0, streams[op_idx]>>>(static_cast<int32_t*>(ptrs[op_idx]), bytes); \
    }

    if (READ == access_type) {
      READ_ITER(1, 0);
      if (duplex) {
        READ_ITER(0, 1);
      }
    } else {
      WRITE_ITER(0, 0);
      if (duplex) {
        WRITE_ITER(1, 1);
      }
    }


    for (auto s : streams) {
      OR_SKIP(cudaStreamSynchronize(s));
    }
  }

  int64_t bytes_processed = int64_t(state.iterations()) * int64_t(bytes);
  if (duplex) {
    bytes_processed *= 2;
  }
  state.SetBytesProcessed(bytes_processed);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

  for (auto s : streams) {
    OR_SKIP(cudaStreamDestroy(s));
  }
  for (auto p : ptrs) {
    OR_SKIP(cudaFree(p));
  }

};

static void registerer() {

  std::string name;
  for (auto workload : {READ, WRITE}) {
    for (auto duplex : {false, true}) {
      for (auto src_gpu : unique_cuda_device_ids()) {
        for (auto dst_gpu : unique_cuda_device_ids()) {
          if (src_gpu < dst_gpu) {

            int s2d, d2s;
            if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&s2d, src_gpu, dst_gpu))
             && !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&d2s, dst_gpu, src_gpu))) {
              if (s2d && d2s) {


                std::string name(NAME);
                if (duplex) name += "_Duplex";
                name += to_string(workload)
                    + "/" + std::to_string(src_gpu)
                    + "/" + std::to_string(dst_gpu);
                benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_GPUGPU,
                src_gpu, dst_gpu, workload, duplex)->ARGS()->UseRealTime();
              }
            }
          }
        }
      }
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);
