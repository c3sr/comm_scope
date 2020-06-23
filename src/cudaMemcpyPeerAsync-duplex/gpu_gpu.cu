#include <cassert>
#include <cuda_runtime.h>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyPeerAsync_Duplex_GPUGPU"

namespace comm_cudaMemcpyPeerAsync_Duplex_GPUGPU {
__global__ void busy_wait(clock_t *d, clock_t clock_count) {
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  if (d) {
    *d = clock_offset;
  }
}
} // namespace comm_cudaMemcpyPeerAsync_Duplex_GPUGPU

auto Comm_cudaMemcpyPeerAsync_Duplex_GPUGPU = [](benchmark::State &state,
                                                 const int gpu0,
                                                 const int gpu1) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0), NAME " failed to reset src CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1), NAME " failed to reset dst CUDA device");

  void *src0 = nullptr;
  void *src1 = nullptr;
  void *dst0 = nullptr;
  void *dst1 = nullptr;
  cudaStream_t stream0;
  cudaStream_t stream1;
  cudaError_t err;
  cudaEvent_t start, stop1, stop;

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME " failed to set src device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src0, bytes), NAME " failed to perform src0 cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst0, bytes), NAME " failed to perform src1 cudaMalloc");
  defer(cudaFree(src0));
  defer(cudaFree(dst0));
  OR_SKIP_AND_RETURN(cudaMemset(src0, 0, bytes),
          NAME " failed to perform src0 cudaMemset");
  OR_SKIP_AND_RETURN(cudaMemset(dst0, 0, bytes),
          NAME " failed to perform src1 cudaMemset");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream0), NAME " failed to create stream");
  defer(cudaStreamDestroy(stream0));
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), NAME " couldn't create start event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), NAME " couldn't create stop event");
  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));
  if (gpu0 != gpu1) {
    err = cudaDeviceDisablePeerAccess(gpu1);
    cudaGetLastError(); // clear error
    if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
      state.SkipWithError(NAME " failed to disable peer access");
      return;
    }
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME " failed to set dst device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src1, bytes), NAME " failed to perform src1 cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst1, bytes), NAME " failed to perform dst1 cudaMalloc");
  defer(cudaFree(src1));
  defer(cudaFree(dst1));
  OR_SKIP_AND_RETURN(cudaMemset(src1, 0, bytes), NAME " failed to perform dst cudaMemset");
  OR_SKIP_AND_RETURN(cudaMemset(dst1, 0, bytes), NAME " failed to perform dst cudaMemset");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream1), NAME " failed to create stream");
  defer(cudaStreamDestroy(stream1));
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop1), NAME " couldn't create stop1 event");
  defer(cudaEventDestroy(stop1));
  if (gpu0 != gpu1) {
    err = cudaDeviceDisablePeerAccess(gpu0);
    cudaGetLastError(); // clear error
    if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
      state.SkipWithError(NAME " failed to disable peer access");
      return;
    }
  }

  size_t cycles = 4096;
  for (auto _ : state) {
    OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), NAME " failed to set src device");
    comm_cudaMemcpyPeerAsync_Duplex_GPUGPU::busy_wait<<<1, 1, 0, stream0>>>(
        nullptr, cycles);
    OR_SKIP_AND_BREAK(cudaGetLastError(), NAME " failed to busy_wait");
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream0), NAME " failed to record start");
    OR_SKIP_AND_BREAK(cudaMemcpyPeerAsync(dst1, gpu1, src0, gpu0, bytes, stream0),
            NAME " failed to memcpy");
    OR_SKIP_AND_BREAK(cudaSetDevice(gpu1), NAME " failed to set src device");
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream1, start, 0),
            NAME " failed to set src device");
    OR_SKIP_AND_BREAK(cudaMemcpyPeerAsync(dst0, gpu0, src1, gpu1, bytes, stream1),
            NAME " failed to memcpy");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop1, stream1), NAME " failed to stop");

    // if kernel has ended, it wasn't long enough to cover the host code.
    // finish transfers, increase cycles, and try again
    err = cudaEventQuery(start);
    if (cudaSuccess == err) {
      cycles *= 1.5;
      OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream0),
              NAME " failed to wait for stream0");
      OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream1),
              NAME " failed to wait for stream1");
      continue;
    } else if (cudaErrorNotReady == err) {
      // kernel was long enough
    } else {
      OR_SKIP_AND_BREAK(err, NAME " errored while waiting for kernel");
    }

    OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), NAME " failed to set src device");
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream0, stop1, 0),
            NAME " failed to set src device");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream0), NAME " failed to stop");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), NAME " failed to synchronize");

    float ms = 0.0f;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&ms, start, stop),
            NAME "failed to compute elapsed time");
    state.SetIterationTime(ms / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
  state.counters["wait_cycles"] = cycles;
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i; j < unique_cuda_device_ids().size(); ++j) {
      auto gpu0 = unique_cuda_device_ids()[i];
      auto gpu1 = unique_cuda_device_ids()[j];
      name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
             std::to_string(gpu1);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_cudaMemcpyPeerAsync_Duplex_GPUGPU, gpu0, gpu1)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
