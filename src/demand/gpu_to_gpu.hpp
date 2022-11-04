#pragma once

#include "scope/scope.hpp"

struct Data {
  char *ptr;
  hipEvent_t start;
  hipEvent_t stop;
  size_t pageSize;
  bool error;
};

static Data setup(benchmark::State &state,
                  const std::string &name,
                  const size_t bytes,
                  const int src_gpu,
                  const int dst_gpu) {

  Data data;
  data.error = false;

  data.pageSize = page_size();

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_gpu))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(dst_gpu))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipSetDevice(dst_gpu))) {
    state.SkipWithError((name + " failed to set hip dst device").c_str());
    data.error = true;
    return data;
  }

  data.ptr = nullptr;
  if (PRINT_IF_ERROR(hipMallocManaged(&data.ptr, bytes))) {
    state.SkipWithError((name + " failed to perform hipMallocManaged").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipMemset(data.ptr, 0, bytes))) {
    state.SkipWithError((name + " failed to perform hipMemset").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipEventCreate(&data.start))) {
    state.SkipWithError((name + " failed to create start event").c_str());
    data.error = true;
    return data;
  }


  if (PRINT_IF_ERROR(hipEventCreate(&data.stop))) {
    state.SkipWithError((name + " failed to create stop event").c_str());
    data.error = true;
    return data;
  }

  return data;
}

static void prep_iteration(char *ptr, size_t bytes, int src_gpu, int dst_gpu) {
    hipMemPrefetchAsync(ptr, bytes, src_gpu);
    hipSetDevice(src_gpu);
    hipDeviceSynchronize();
    hipSetDevice(dst_gpu);
    hipDeviceSynchronize();
}