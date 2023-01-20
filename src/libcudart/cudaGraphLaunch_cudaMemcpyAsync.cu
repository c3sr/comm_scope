/* compiled with nvcc so __CUDACC_VER_MAJOR__ is defined
 */

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudart_cudaGraphLaunch_cudaMemcpyAsync"

auto Comm_cudart_cudaGraphLaunch_cudaMemcpyAsync = [](benchmark::State &state,
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

  void *src = nullptr;
  void *dst = nullptr;

  OR_SKIP_AND_RETURN(cudaMalloc(&src, 100), "");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, 100), "");
  defer(cudaFree(src));
  defer(cudaFree(dst));

  // create the graph to launch
  OR_SKIP_AND_RETURN(cudaStreamBeginCapture(stream
#if __CUDACC_VER_MAJOR__ >= 11 ||                                              \
    (__CUDACC_VER_MAJOR__ >= 10 && __CUDACC_VER_MINOR__ > 0)
                                            ,
                                            cudaStreamCaptureModeGlobal
#endif
                                            ),
                     "");
  for (int i = 0; i < iters; ++i) {
    cudaMemcpyAsync(dst, src, 100, cudaMemcpyDefault, stream);
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
    for (auto numaId : numa::mems()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numaId) +
                         "/" + std::to_string(cudaId);
      benchmark::RegisterBenchmark(name.c_str(),
                                   Comm_cudart_cudaGraphLaunch_cudaMemcpyAsync,
                                   numaId, cudaId)
          ->GRAPH_ARGS();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
