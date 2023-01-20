

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudart_cudaGraphInstantiate_kernel"

__global__ void Comm_cudart_cudaGraphInstantiate_kernel_kernel() {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
auto Comm_cudart_cudaGraphInstantiate_kernel = [](benchmark::State &state,
                                                  const int numaId,
                                                  const int cudaId) {
  const int iters = state.range(0);

  numa::ScopedBind binder(numaId);

  OR_SKIP_AND_RETURN(cuda_reset_device(cudaId), "failed to reset device");
  OR_SKIP_AND_RETURN(cudaSetDevice(cudaId), "failed to set CUDA dst device");

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");

  cudaError_t err;
  for (auto _ : state) {
    state.PauseTiming();
    OR_SKIP_AND_BREAK(cudaStreamBeginCapture(stream
#if __CUDACC_VER_MAJOR__ >= 11 ||                                              \
    (__CUDACC_VER_MAJOR__ >= 10 && __CUDACC_VER_MINOR__ > 0)
                                             ,
                                             cudaStreamCaptureModeGlobal
#endif
                                             ),
                      "");
    for (int i = 0; i < iters; ++i) {
      Comm_cudart_cudaGraphInstantiate_kernel_kernel<<<1, 1, 0, stream>>>();
    }
    OR_SKIP_AND_BREAK(cudaStreamEndCapture(stream, &graph), "");
    state.ResumeTiming();
    err = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    state.PauseTiming();
    OR_SKIP_AND_BREAK(cudaGraphDestroy(graph), "");
    OR_SKIP_AND_BREAK(cudaGraphExecDestroy(instance), "");
    state.ResumeTiming();
  }
  OR_SKIP_AND_RETURN(err, "failed to cudaGraphInstantiate");

  state.SetItemsProcessed(state.iterations());
  state.counters["cuda_id"] = cudaId;
  state.counters["numa_id"] = numaId;
};


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
static void registerer() {
  for (auto cudaId : unique_cuda_device_ids()) {
    for (auto numaId : numa::mems()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numaId) +
                         "/" + std::to_string(cudaId);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_cudart_cudaGraphInstantiate_kernel, numaId, cudaId)
          ->GRAPH_ARGS();
    }
  }
}
#pragma GCC diagnostic pop


SCOPE_AFTER_INIT(registerer, NAME);

