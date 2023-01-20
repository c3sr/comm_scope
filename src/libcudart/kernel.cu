/* Measure the runtime cost of a kernel launch
 */

#include "scope/scope.hpp"

#define NAME "Comm_cudart_kernel"

// helper for passing some number of bytes by value
template <unsigned N> struct S { char bytes[N]; };
template <> struct S<0> {};

template <unsigned N> __global__ void Comm_cudart_kernel_kernel(S<N> s) {
  (void)s;
}

auto Comm_cudart_kernel = [](benchmark::State &state, const int gpu,
                             const int numaId) {
  numa::ScopedBind binder(numaId);

  if (0 == state.thread_index()) {
    OR_SKIP_AND_RETURN(cuda_reset_device(gpu), "failed to reset CUDA device");
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "");
  OR_SKIP_AND_RETURN(cudaFree(0), "failed to init");

  const size_t nArgs = state.range(0);

#define LAUNCH(n)                                                              \
  case n: {                                                                    \
    Comm_cudart_kernel_kernel<(n)><<<1, 1>>>(S<(n)>());                        \
    break;                                                                     \
  }

  for (auto _ : state) {
    // Start copy
    switch (nArgs) {
      LAUNCH(0)
      LAUNCH(1)
      LAUNCH(4)
      LAUNCH(8)
      LAUNCH(32)
      LAUNCH(64)
      LAUNCH(96)
      LAUNCH(128)
      LAUNCH(256)
      LAUNCH(512)
      LAUNCH(1024)
      LAUNCH(2048)
      LAUNCH(4096)
    default: {
      state.SkipWithError("unexpected number of params");
      break;
    }
    }
  }
#undef LAUNCH
  OR_SKIP_AND_RETURN(cudaGetLastError(), "failed to launch kernel");

  state.SetItemsProcessed(state.iterations());
  state.counters["gpu"] = gpu;
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), "failed to synchronize");
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (int numaId : numa::mems()) {
      for (size_t numThreads = 1;
           numThreads <= numa::cpus_in_node(numaId).size(); numThreads *= 2) {
        auto gpu = unique_cuda_device_ids()[i];
        name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
               std::to_string(gpu);
        benchmark::RegisterBenchmark(name.c_str(), Comm_cudart_kernel, gpu,
                                     numaId)
            ->Arg(0)
            ->Arg(1)
            ->Arg(4)
            ->Arg(8)
            ->Arg(32)
            ->Arg(64)
            ->Arg(96)
            ->Arg(128)
            ->Arg(256)
            ->Arg(512)
            ->Arg(1024)
            ->Arg(2048)
            ->Arg(4096)
            ->Threads(numThreads)
            ->UseRealTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
