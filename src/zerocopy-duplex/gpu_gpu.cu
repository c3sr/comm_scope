#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_ZeroCopy_Duplex_GPUGPU"

#define OR_SKIP(stmt)                                                          \
  if (PRINT_IF_ERROR(stmt)) {                                                  \
    state.SkipWithError(NAME);                                                 \
    return;                                                                    \
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
  (void)s[threadIdx.x];
}

auto Comm_ZeroCopy_GPUGPU = [](benchmark::State &state, const int gpu0,
                               const int gpu1, const AccessType access_type) {
  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  cudaStream_t streams[2];
  void *ptrs[2] = {nullptr};
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEvent_t other = nullptr;

  OR_SKIP(cuda_reset_device(gpu0));
  OR_SKIP(cuda_reset_device(gpu1));

  OR_SKIP(cudaSetDevice(gpu0));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaErrorPeerAccessAlreadyEnabled != err) {
      OR_SKIP(err);
    }
  }
  OR_SKIP(cudaSetDevice(gpu1));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaErrorPeerAccessAlreadyEnabled != err) {
      OR_SKIP(err);
    }
  }

  // always associate the stream with the executing gpu
#define RD_INIT(src, dst, op_idx)                                              \
  OR_SKIP(cudaSetDevice(gpu##src));                                            \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes));                                   \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes));                                 \
  OR_SKIP(cudaSetDevice(gpu##dst));                                            \
  OR_SKIP(cudaStreamCreate(&streams[dst]));

// code runs on src, and data on dst
#define WR_INIT(src, dst, op_idx)                                              \
  OR_SKIP(cudaSetDevice(gpu##dst));                                            \
  OR_SKIP(cudaMalloc(&ptrs[op_idx], bytes));                                   \
  OR_SKIP(cudaMemset(ptrs[op_idx], 0, bytes));                                 \
  OR_SKIP(cudaSetDevice(gpu##src));                                            \
  OR_SKIP(cudaStreamCreate(&streams[src]));

  if (READ == access_type) {
    RD_INIT(0, 1, 0); // op 0: 1 reads from 0 in stream 1
    RD_INIT(1, 0, 1); // op 1: 0 read from 1 in stream 0
  } else {
    WR_INIT(0, 1, 0); // wr op 0: 0 writes to 1 in stream 0
    WR_INIT(1, 0, 1); // wr op 1: 1 writes to 0 in stream 1
  }

  // always stream 0 is the primary stream and stream 1 is the secondary
  // time is computed from stream 0, which does not complete until stream 1 is
  // done
  OR_SKIP(cudaSetDevice(gpu0));
  OR_SKIP(cudaEventCreate(&start))
  OR_SKIP(cudaEventCreate(&stop))
  OR_SKIP(cudaSetDevice(gpu1));
  OR_SKIP(cudaEventCreate(&other))

  for (auto _ : state) {
#define READ_ITER(dst, op_idx)                                                 \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu##dst));                                          \
    gpu_read<int32_t><<<256, 256, 0, streams[dst]>>>(                          \
        static_cast<int32_t *>(ptrs[op_idx]), bytes);                          \
  }
#define WRITE_ITER(src, op_idx)                                                \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu##src));                                          \
    gpu_write<int32_t><<<256, 256, 0, streams[src]>>>(                         \
        static_cast<int32_t *>(ptrs[op_idx]), bytes);                          \
  }

#define RECORD_START()                                                         \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu0));                                              \
    OR_SKIP(cudaEventRecord(start, streams[0]));                               \
  }

#define RECORD_OTHER()                                                         \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu1));                                              \
    OR_SKIP(cudaEventRecord(other, streams[1]));                               \
  }

#define WAIT_OTHER()                                                           \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu0));                                              \
    OR_SKIP(cudaStreamWaitEvent(streams[0], other, 0 /* must be 0 */));        \
  }

#define RECORD_STOP()                                                          \
  {                                                                            \
    OR_SKIP(cudaSetDevice(gpu0));                                              \
    OR_SKIP(cudaEventRecord(stop, streams[0]));                                \
  }

    if (READ == access_type) {
      RECORD_START();
      READ_ITER(1, 0);
      READ_ITER(0, 1);
      RECORD_OTHER();
      WAIT_OTHER();
      RECORD_STOP();
    } else {
      RECORD_START();
      WRITE_ITER(0, 0);
      WRITE_ITER(1, 1);
      RECORD_OTHER();
      WAIT_OTHER();
      RECORD_STOP();
    }

    OR_SKIP(cudaStreamSynchronize(streams[0]));
    float millis = 0;
    OR_SKIP(cudaEventElapsedTime(&millis, start, stop));
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
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
    for (auto src_gpu : unique_cuda_device_ids()) {
      for (auto dst_gpu : unique_cuda_device_ids()) {
        if (src_gpu < dst_gpu) {

          int s2d = false;
          int d2s = false;
          if (!PRINT_IF_ERROR(
                  cudaDeviceCanAccessPeer(&s2d, src_gpu, dst_gpu)) &&
              !PRINT_IF_ERROR(
                  cudaDeviceCanAccessPeer(&d2s, dst_gpu, src_gpu))) {
            if (s2d && d2s) {

              std::string name(NAME);
              name += to_string(workload) + "/" + std::to_string(src_gpu) +
                      "/" + std::to_string(dst_gpu);
              benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_GPUGPU,
                                           src_gpu, dst_gpu, workload)
                  ->ARGS()
                  ->UseManualTime();
            } else {
              LOG(debug,
                  "{} can't run on devices {} {}: peer access not available",
                  NAME, src_gpu, dst_gpu);
            }
          }
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
