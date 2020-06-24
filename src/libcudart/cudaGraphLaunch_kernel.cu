

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudart_cudaGraphLaunch_kernel"

__global__ void Comm_cudart_cudaGraphLaunch_kernel_kernel(){}

auto Comm_cudart_cudaGraphLaunch_kernel = [](benchmark::State &state, const int numaId, const int cudaId) {
  const int iters = state.range(0);

  numa::ScopedBind binder(numaId);

  OR_SKIP_AND_RETURN(cuda_reset_device(cudaId), "failed to reset device");
  OR_SKIP_AND_RETURN(cudaSetDevice(cudaId), "failed to set CUDA dst device");

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");

  // create the graph to launch
  OR_SKIP_AND_RETURN(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "");
  for (int i = 0; i < iters; ++i) {
    Comm_cudart_cudaGraphLaunch_kernel_kernel<<<1,1,0,stream>>>();
  }
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
  state.counters["cuda_id"] = cudaId;
  state.counters["numa_id"] = numaId;
};

static void registerer() {
  for (auto cudaId : unique_cuda_device_ids()) {
    for (auto numaId : numa::ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numaId) +
                         "/" + std::to_string(cudaId);
      benchmark::RegisterBenchmark(name.c_str(),
                                   Comm_cudart_cudaGraphLaunch_kernel,
                                   numaId, cudaId)
          ->GRAPH_ARGS();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
