/* Measure the runtime cost of cudaMemcpy3DPeerAsync
 */

#include "sysbench/sysbench.hpp"

#define NAME "Comm_cudart_cudaMemcpyAsync_GPUToPinned"

auto Comm_cudart_cudaMemcpyAsync_GPUToPinned = [](benchmark::State &state,
                                            const int gpu, const int numa_id) {

  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu), "failed to reset CUDA device");


  // Create One stream per copy
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");

  // fixed-size transfer
  const size_t bytes = 1024 * 10;

  void *src, *dst;

  // allocate on host
  dst = aligned_alloc(page_size(), bytes);
  std::memset(dst, 0, bytes);
  OR_SKIP_AND_RETURN(cudaHostRegister(dst, bytes, cudaHostRegisterPortable), "failed to register allocation");

  // allocate on gpu and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src, bytes), "failed to perform cudaMalloc3D");
  OR_SKIP_AND_RETURN(cudaMemset(src, 0, bytes),
          "failed to perform src cudaMemset");

  for (auto _ : state) {
    // Start copy
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);

    // measure one copy at a time
    state.PauseTiming();
    OR_SKIP_AND_BREAK(err,
                      "failed to start cudaMemcpyAsync");

    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream),
                      "failed to synchronize");
    state.ResumeTiming();
  }

  state.counters["gpu"] = gpu;
  state.counters["numa_id"] = numa_id;

  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaHostUnregister(dst), "");
  free(dst);
  OR_SKIP_AND_RETURN(cudaFree(src), "failed to cudaFree");
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < numa::ids().size(); ++i) {
    for (size_t j = i; j < unique_cuda_device_ids().size(); ++j) {
      auto numa_id = numa::ids()[i];
      auto gpu = unique_cuda_device_ids()[j];
name = std::string(NAME) + "/" + std::to_string(numa_id) + "/" +
                 std::to_string(gpu);
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_cudart_cudaMemcpyAsync_GPUToPinned, gpu, numa_id)->UseRealTime();
        }
      }
    }

SYSBENCH_AFTER_INIT(registerer, NAME);
