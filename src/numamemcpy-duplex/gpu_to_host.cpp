#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numamemcpy-duplex/args.hpp"

#define NAME "DUPLEX/Memcpy/GPUToHost" 

static void DUPLEX_Memcpy_GPUToHost(benchmark::State &state) {
  cudaProfilerStart();
  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const int numa   = state.range(1);
  const int gpu    = state.range(2);

  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }

  // There are two copies, one gpu -> host, one host -> gpu

  // Create One stream per copy
  cudaStream_t stream1, stream2;
  std::vector<cudaStream_t> streams = {stream1, stream2};
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  // Start and stop events for each copy
  cudaEvent_t start1, start2, stop1, stop2;
  std::vector<cudaEvent_t> starts = {start1, start2};
  std::vector<cudaEvent_t> stops  = {stop1, stop2};
  cudaEventCreate(&starts[0]);
  cudaEventCreate(&starts[1]);
  cudaEventCreate(&stops[0]);
  cudaEventCreate(&stops[1]);

  // Source and destination for each copy
  std::vector<char *> srcs;
  std::vector<char *> dsts;

  // create a source and destination allocation for first copy: gpu ->host
  char *ptr;
  if (PRINT_IF_ERROR(cudaSetDevice(gpu))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  dsts.push_back(ptr);
  defer(cudaFree(ptr));
  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform dst cudaMemset");
    return;
  }

  char *ptr2;
  if (PRINT_IF_ERROR(cudaSetDevice(numa))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  ptr2 = (char*) malloc(bytes);
  if(NULL == ptr2){
    state.SkipWithError(NAME " ptr is null");
    return;
  }
  srcs.push_back(ptr2);
  defer(free(ptr2));

  // create a source and destination for second copy: host -> gpu
  char *ptr3;
  if (PRINT_IF_ERROR(cudaSetDevice(numa))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  ptr3 = (char*) malloc(bytes);
  if(NULL == ptr3){
    state.SkipWithError(NAME " ptr is null");
    return;
  }
  dsts.push_back(ptr3);
  defer(free(ptr3));

  char *ptr4;
  if (PRINT_IF_ERROR(cudaSetDevice(gpu))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr4, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  srcs.push_back(ptr4);
  defer(cudaFree(ptr4)); 
  if (PRINT_IF_ERROR(cudaMemset(ptr4, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }


  assert(starts.size() == stops.size());
  assert(streams.size() == starts.size());
  assert(srcs.size() == dsts.size());
  assert(streams.size() == srcs.size());

  for (auto _ : state) {

    // Start all copies
    for (size_t i = 0; i < streams.size(); ++i) {
      auto start  = starts[i];
      auto stop   = stops[i];
      auto stream = streams[i];
      auto src    = srcs[i];
      auto dst    = dsts[i];
      if (PRINT_IF_ERROR(cudaEventRecord(start, stream))) {
        state.SkipWithError(NAME " failed to record start event");
        return;
      }
      if (PRINT_IF_ERROR(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream))) {
        state.SkipWithError(NAME " failed to start cudaMemcpyAsync");
        return;
      }
      if (PRINT_IF_ERROR(cudaEventRecord(stop, stream))) {
        state.SkipWithError(NAME " failed to record stop event");
        return;
      }
    }

    // Wait for all copies to finish
    for (auto s : stops) {
      if (PRINT_IF_ERROR(cudaEventSynchronize(s))) {
        state.SkipWithError(NAME " failed to synchronize");
        return;
      }
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for (const auto stop : stops) {
        float millis;

        if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }

        maxMillis = std::max(millis, maxMillis);
      }
    }

    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters.insert({{"bytes", bytes}});
  
  float stopSum = 0;
  float startSum = 0;
  for ( const auto stream : streams ){

        float startTime1, startTime2, stopTime1, stopTime2;
        if (PRINT_IF_ERROR(cudaEventElapsedTime(&startTime1, starts[0], starts[1]))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }
        if (PRINT_IF_ERROR(cudaEventElapsedTime(&startTime2, starts[1], starts[0]))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }
        if (PRINT_IF_ERROR(cudaEventElapsedTime(&stopTime1, stops[0], stops[1]))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }
        if (PRINT_IF_ERROR(cudaEventElapsedTime(&stopTime2, stops[1], stops[0]))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }

        startSum += std::max(startTime1, startTime2);
        stopSum += std::max(stopTime1, stopTime2);
  }

  state.counters["start_spread"] = startSum/state.iterations();
  state.counters["stop_spread"] = stopSum/state.iterations();

  cudaProfilerStop();
}

BENCHMARK(DUPLEX_Memcpy_GPUToHost)->Apply(ArgsCountNumaGpu)->UseManualTime();

