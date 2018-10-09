#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "mapping/args.hpp"
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
  std::vector<cudaEvent_t>   starts(duplex ? 2 : 1);
  std::vector<cudaEvent_t>    stops(duplex ? 2 : 1);
  std::vector<void *>          ptrs(duplex ? 2 : 1, nullptr);

  OR_SKIP(utils::cuda_reset_device(gpu0));
  OR_SKIP(utils::cuda_reset_device(gpu1));
  

#define RD_INIT(src, dst, op_idx) \
  OR_SKIP(cudaSetDevice(gpu##src)); \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes)); \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes)); \
  OR_SKIP(cudaSetDevice(gpu##dst)); \
  OR_SKIP(cudaEventCreate(&starts[op_idx])); \
  OR_SKIP(cudaEventCreate(&stops[op_idx])); \
  OR_SKIP(cudaStreamCreate(&streams[op_idx]));

// code runs on src, and data on dst
#define WR_INIT(src, dst, op_idx) \
  OR_SKIP(cudaSetDevice(gpu##dst)); \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes)); \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes)); \
  OR_SKIP(cudaSetDevice(gpu##src)); \
  OR_SKIP(cudaEventCreate(&starts[op_idx])); \
  OR_SKIP(cudaEventCreate(&stops[op_idx])); \
  OR_SKIP(cudaStreamCreate(&streams[op_idx]));

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
#define READ_ITER(src, dst, op_idx) \
    { \
    OR_SKIP(cudaSetDevice(gpu##dst)); \
    OR_SKIP(cudaEventRecord(starts[op_idx], streams[op_idx])); \
    gpu_read<int32_t><<<256, 256, 0, streams[op_idx]>>>(static_cast<int32_t*>(ptrs[op_idx]), bytes); \
    OR_SKIP(cudaEventRecord(stops[op_idx], streams[op_idx])); \
    }
    // WRITE: gpu0 writes to gpu1 (gpu0 is src, gpu1 is dst)
#define WRITE_ITER(src, dst, op_idx) \
    { \
    OR_SKIP(cudaSetDevice(gpu##src)); \
    OR_SKIP(cudaEventRecord(starts[op_idx], streams[op_idx])); \
    gpu_write<int32_t><<<256, 256, 0, streams[op_idx]>>>(static_cast<int32_t*>(ptrs[op_idx]), bytes); \
    OR_SKIP(cudaEventRecord(stops[op_idx], streams[op_idx])); \
    }

    if (READ == access_type) {
      READ_ITER(0, 1, 0);
      if (duplex) {
        READ_ITER(1, 0, 1);
      }
    } else {
      WRITE_ITER(0, 1, 0);
      if (duplex) {
        WRITE_ITER(1, 0, 1);
      }
    }


    cudaEventSynchronize(stops[0]);
    if (duplex) {
      cudaEventSynchronize(stops[1]);
    }


    float millis = 0;
    for (auto start : starts) {
      for (auto stop : stops) {
        float this_millis;
        OR_SKIP(cudaEventElapsedTime(&this_millis, start, stop));
        millis = std::max(millis, this_millis);
      }
    }
    
    state.SetIterationTime(millis / 1000);
  }

  int64_t bytes_processed = int64_t(state.iterations()) * int64_t(bytes);
  if (duplex) {
    bytes_processed *= 2;
  }
  state.SetBytesProcessed(bytes_processed);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;

};

static void registerer() {

  std::string name;
  for (auto workload : {READ, WRITE}) {
    for (auto duplex : {false, true}) {
      for (auto id0 : unique_cuda_device_ids()) {
        for (auto id1 : unique_cuda_device_ids()) {
          if (id0 < id1) {
            std::string name(NAME);
            if (duplex) name += "_Duplex";
            name += to_string(workload)
                + "/" + std::to_string(id0)
                + "/" + std::to_string(id1);
            benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_GPUGPU,
            id0, id1, workload, duplex)->ARGS()->UseManualTime();
          }
        }
      }
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);
