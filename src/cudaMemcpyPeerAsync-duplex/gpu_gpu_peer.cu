#include <cassert>
#include <cuda_runtime.h>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyPeerAsync_Duplex_GPUGPUPeer"

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

auto Comm_cudaMemcpyPeerAsync_Duplex_GPUGPUPeer = [](benchmark::State &state,
                                                     const int gpu0,
                                                     const int gpu1) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                     "failed to reset src CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                     "failed to reset dst CUDA device");

  void *src0 = nullptr;
  void *src1 = nullptr;
  void *dst0 = nullptr;
  void *dst1 = nullptr;
  cudaStream_t stream0;
  cudaStream_t stream1;
  cudaError_t err;
  cudaEvent_t start, stop1, stop;

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "failed to set src device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src0, bytes),
                     "failed to perform src0 cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst0, bytes),
                     "failed to perform src1 cudaMalloc");
  defer(cudaFree(src0));
  defer(cudaFree(dst0));
  OR_SKIP_AND_RETURN(cudaMemset(src0, 0, bytes),
                     "failed to perform src0 cudaMemset");
  OR_SKIP_AND_RETURN(cudaMemset(dst0, 0, bytes),
                     "failed to perform src1 cudaMemset");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream0), "failed to create stream");
  defer(cudaStreamDestroy(stream0));
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "couldn't create start event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "couldn't create stop event");
  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));
  if (gpu0 != gpu1) {
    err = cudaDeviceEnablePeerAccess(gpu1, 0);
    cudaGetLastError(); // clear error
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError("failed to enable peer access to gpu1");
      return;
    }
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "failed to set dst device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src1, bytes),
                     "failed to perform src1 cudaMalloc");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst1, bytes),
                     "failed to perform dst1 cudaMalloc");
  defer(cudaFree(src1));
  defer(cudaFree(dst1));
  OR_SKIP_AND_RETURN(cudaMemset(src1, 0, bytes),
                     "failed to perform dst cudaMemset");
  OR_SKIP_AND_RETURN(cudaMemset(dst1, 0, bytes),
                     "failed to perform dst cudaMemset");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream1), "failed to create stream");
  defer(cudaStreamDestroy(stream1));
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop1), "couldn't create stop1 event");
  defer(cudaEventDestroy(stop1));
  if (gpu0 != gpu1) {
    err = cudaDeviceEnablePeerAccess(gpu0, 0);
    cudaGetLastError(); // clear error
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError("failed to enable peer access to gpu0");
      return;
    }
  }

  size_t cycles = 4096;
  for (auto _ : state) {

    // keep making kernel longer and longer until it hides all host code
    restart_iteration:
      OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), "failed to set src device");
      busy_wait<<<1, 1, 0, stream0>>>(nullptr, cycles);
      OR_SKIP_AND_BREAK(cudaGetLastError(), "failed to busy_wait");
      OR_SKIP_AND_BREAK(cudaEventRecord(start, stream0),
                        "failed to record start");
      OR_SKIP_AND_BREAK(
          cudaMemcpyPeerAsync(dst1, gpu1, src0, gpu0, bytes, stream0),
          "failed to memcpy");
      OR_SKIP_AND_BREAK(cudaSetDevice(gpu1), "failed to set src device");
      OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream1, start, 0),
                        "failed to wait");
      OR_SKIP_AND_BREAK(
          cudaMemcpyPeerAsync(dst0, gpu0, src1, gpu1, bytes, stream1),
          "failed to memcpy");
      OR_SKIP_AND_BREAK(cudaEventRecord(stop1, stream1), "failed to stop");
      OR_SKIP_AND_BREAK(cudaSetDevice(gpu0), "failed to set src device");
      OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream0, stop1, 0),
                        "failed to set src device");
      OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream0), "failed to stop");

      // if kernel has ended, it wasn't long enough to cover the host code:
      // finish transfers, increase cycles, and try again
      err = cudaEventQuery(start);
      if (cudaSuccess == err) {
        cycles *= 2;
        OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream0),
                          "failed to wait for stream0");
        OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream1),
                          "failed to wait for stream1");
        goto restart_iteration;
      } else if (cudaErrorNotReady == err) {
        // kernel was long enough
      } else {
        OR_SKIP_AND_BREAK(err, "errored while waiting for kernel");
      }

    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "failed to synchronize");
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
  const std::vector<MemorySpace> cudaSpaces = scope::system::memory_spaces(MemorySpace::Kind::cuda_device);

  for (const auto &space0 : cudaSpaces) {
    for (const auto &space1 : cudaSpaces) {

      auto gpu0 = space0.device_id();
      auto gpu1 = space1.device_id();
      name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
             std::to_string(gpu1);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_cudaMemcpyPeerAsync_Duplex_GPUGPUPeer, gpu0, gpu1)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
