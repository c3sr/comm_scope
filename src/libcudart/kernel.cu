/* Measure the runtime cost of cudaMemcpy3DPeerAsync
 */

#include "sysbench/sysbench.hpp"

#define NAME "Comm_cudart_kernel"

// helper for passing some number of bytes by value
template<unsigned N> struct S {
  char bytes[N];
};

template<> struct S<0> {
};

template<unsigned N>
__global__ void Comm_cudart_kernel_kernel(S<N> s) {(void)s; }

auto Comm_cudart_kernel = [](benchmark::State &state, const int gpu) {
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu), "failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "");

  const size_t nArgs = state.range(0);

  #define LAUNCH(n) \
  case n: { \
    Comm_cudart_kernel_kernel<(1 << n)><<<1, 1>>>(S<(1 << n)>());\
      break;\
    } \

  for (auto _ : state) {
    // Start copy
    switch (nArgs) {
    LAUNCH(0)
    LAUNCH(1)
    LAUNCH(2)
    LAUNCH(3)
    LAUNCH(4)
    LAUNCH(5)
    LAUNCH(6)
    LAUNCH(7)
    LAUNCH(8)
    LAUNCH(9)
    LAUNCH(10)
    LAUNCH(11)
    LAUNCH(12)
    default: {
      state.SkipWithError("unexpected number of params");
      break;
    }
    }

    // measure one copy at a time
    state.PauseTiming();
    OR_SKIP_AND_BREAK(cudaGetLastError(),
                      "failed to start cudaMemcpy3DPeerAsync");

    OR_SKIP_AND_BREAK(cudaDeviceSynchronize(), "failed to synchronize");
    state.ResumeTiming();
  }

  state.counters["gpu"] = gpu;
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
      auto gpu = unique_cuda_device_ids()[i];
      name = std::string(NAME) + "/" + std::to_string(gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudart_kernel, gpu)
          ->DenseRange(0, 12, 1)
          ->UseRealTime();
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);
