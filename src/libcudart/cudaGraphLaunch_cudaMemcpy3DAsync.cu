/* compiled with nvcc so __CUDACC_VER_MAJOR__ is defined
 */

#include "scope/scope.hpp"

#include "args.hpp"

#include <cuda_runtime.h>

#define NAME "Comm_cudart_cudaGraphLaunch_cudaMemcpy3DAsync"

auto Comm_cudart_cudaGraphLaunch_cudaMemcpy3DAsync = [](benchmark::State &state,
                                                        const int gpu0,
                                                        const int gpu1) {
  const int iters = state.range(0);

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1),
                     NAME " failed to reset CUDA device");

  cudaGetLastError(); // clear any previous errors
  OR_SKIP_AND_RETURN(cudaGetLastError(), "last error");

  // small enough transfer that the runtime cost is larger
  cudaExtent copyExt;
  copyExt.width = 8;
  copyExt.height = 8;
  copyExt.depth = 8;

  // properties of the allocation
  cudaExtent allocExt;
  allocExt.width = copyExt.width;
  allocExt.height = copyExt.height;
  allocExt.depth = copyExt.depth;

  cudaMemcpy3DParms params = {};

  // allocate on gpu0 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&params.srcPtr, allocExt),
                     NAME " failed to perform cudaMalloc3D");
  allocExt.width = params.srcPtr.pitch;
  OR_SKIP_AND_RETURN(cudaMemset3D(params.srcPtr, 0, allocExt),
                     NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  cudaStream_t stream;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), NAME "failed to create stream");
  defer(cudaStreamDestroy(stream));

  // allocate on gpu1 and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), NAME "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc3D(&params.dstPtr, allocExt),
                     NAME " failed to perform cudaMalloc3D");
  OR_SKIP_AND_RETURN(cudaMemset3D(params.dstPtr, 0, allocExt),
                     NAME " failed to perform src cudaMemset");
  if (gpu0 != gpu1) {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError(NAME " failed to ensure peer access");
    }
  }

  // set up copy parameters
  params.dstArray = 0; // provided dstPtr
  params.srcArray = 0; // provided srcPtr
  params.dstPos = make_cudaPos(0, 0, 0);
  params.srcPos = make_cudaPos(0, 0, 0);
  params.extent = copyExt;
  params.kind = cudaMemcpyDeviceToDevice;

  cudaGraph_t graph = 0;
  cudaGraphExec_t instance = 0;

  // create the graph to launch
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "failed to set device");
  OR_SKIP_AND_RETURN(cudaStreamBeginCapture(stream
#if __CUDACC_VER_MAJOR__ >= 11 ||                                              \
    (__CUDACC_VER_MAJOR__ >= 10 && __CUDACC_VER_MINOR__ > 0)
                                            ,
                                            cudaStreamCaptureModeThreadLocal
#endif
                                            ),
                     "");

  cudaError_t err = cudaSuccess; // if iters is 0
  for (int i = 0; i < iters; ++i) {
    err = cudaMemcpy3DAsync(&params, stream);
  }
  OR_SKIP_AND_RETURN(err, "?"); // FIXME: what is the purpose of this little looper

  OR_SKIP_AND_RETURN(cudaStreamEndCapture(stream, &graph), "capture error");
  OR_SKIP_AND_RETURN(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0),
                     "error during instantiate");
  defer(cudaGraphDestroy(graph));
  defer(cudaGraphExecDestroy(instance));

  for (auto _ : state) {
    state.PauseTiming();
    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream), "error in sync");
    state.ResumeTiming();
    OR_SKIP_AND_BREAK(cudaGraphLaunch(instance, stream), "error in launch");
  }

  state.SetItemsProcessed(state.iterations());
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
};

static void registerer() {
  std::string name;

  const std::vector<Device> cudas = scope::system::cuda_devices();

  for (size_t i = 0; i < cudas.size(); ++i) {
    for (size_t j = i; j < cudas.size(); ++j) {
      int gpu0 = cudas[i];
      int gpu1 = cudas[j];
      int ok1, ok2;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok1, gpu0, gpu1)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&ok2, gpu1, gpu0))) {
        if ((ok1 && ok2) || i == j) {
          name = std::string(NAME) + "/" + std::to_string(gpu0) + "/" +
                 std::to_string(gpu1);
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_cudart_cudaGraphLaunch_cudaMemcpy3DAsync, gpu0,
              gpu1)
              ->UseRealTime()
              ->GRAPH_ARGS();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
