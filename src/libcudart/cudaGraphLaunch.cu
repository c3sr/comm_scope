

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudart_cudaGraphLaunch"

__global__ void Comm_cudart_cudaGraphLaunch_kernel() {}

static void copy_launcher(void *dst, const void *src, const int size,
                          cudaStream_t stream, const int iters) {
  for (int i = 0; i < iters; ++i) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
  }
}

static void kernel_launcher(cudaStream_t stream, const int iters) {
  for (int i = 0; i < iters; ++i) {
    Comm_cudart_cudaGraphLaunch_kernel<<<1, 1, 0, stream>>>();
  }
}

auto Comm_cudart_cudaGraphLaunch = [](benchmark::State &state,
                                           const int numa_id,
                                           const int cuda_id) {
  const int launches = state.range(0);

  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(cuda_reset_device(cuda_id), "failed to reset device");
  OR_SKIP_AND_RETURN(cudaSetDevice(cuda_id), "failed to set CUDA dst device");

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");

  void *src = nullptr;
  void *dst = nullptr;

  OR_SKIP_AND_RETURN(cudaMalloc(&src, 100), "");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, 100), "");
  defer(cudaFree(src));
  defer(cudaFree(dst));

  // create the graph to launch
  OR_SKIP_AND_RETURN(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "");
  // copy_launcher(dst, src, 100, stream, launches);
  kernel_launcher(stream, launches);
  OR_SKIP_AND_RETURN(cudaStreamEndCapture(stream, &graph), "");
  OR_SKIP_AND_RETURN(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0), "");
  defer(cudaGraphDestroy(graph));
  defer(cudaGraphExecDestroy(instance));

  for (auto _ : state) {
    state.PauseTiming();
    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream), "");
    state.ResumeTiming();
    OR_SKIP_AND_BREAK(cudaGraphLaunch(instance, stream), "");
  }

  state.SetItemsProcessed(state.iterations());
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numa_id) +
                         "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_cudart_cudaGraphLaunch, numa_id, cuda_id)
          ->GRAPH_ARGS();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
