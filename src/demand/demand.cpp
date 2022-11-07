#include "demand.hpp"

/*extern*/ std::condition_variable cv;
/*extern*/ std::mutex m;
/*extern*/ volatile bool ready = false;

template <>
Data setup<Kind::GPUToGPU>(benchmark::State &state,
                  const std::string &name,
                  const size_t bytes,
                  const int src_id,
                  const int dst_id) {

  Data data;
  data.error = false;

  data.pageSize = page_size();

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_id))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }
  if (PRINT_IF_ERROR(scope::hip_reset_device(dst_id))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipSetDevice(dst_id))) {
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

template <>
Data setup<Kind::GPUToHost>(benchmark::State &state,
                  const std::string &name,
                  const size_t bytes,
                  const int src_id,
                  const int /*dst_id*/) {

  Data data;
  data.ptr = nullptr;
  data.error = false;
  data.pageSize = page_size();

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_id))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipSetDevice(src_id))) {
    state.SkipWithError((name + " failed to set hip src device").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipMallocManaged(&data.ptr, bytes))) {
    state.SkipWithError((name + " failed to perform hipMallocManaged").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipMemAdvise(
          data.ptr, bytes, hipMemAdviseSetPreferredLocation, src_id))) {
    state.SkipWithError((name + " failed to hipMemAdvise").c_str());
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

template <>
Data setup<Kind::HostToGPU>(benchmark::State &state,
                  const std::string &name,
                  const size_t bytes,
                  const int /*src_id*/,
                  const int dst_id) {

  Data data;
  data.error = false;

  data.pageSize = page_size();

  if (PRINT_IF_ERROR(scope::hip_reset_device(dst_id))) {
    state.SkipWithError((name + " failed to reset hip src device").c_str());
    data.error = true;
    return data;
  }

  if (PRINT_IF_ERROR(hipSetDevice(dst_id))) {
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

template <>
void prep_iteration<Kind::GPUToGPU>(char *ptr, size_t bytes, int src_id, int dst_id) {
    hipMemPrefetchAsync(ptr, bytes, src_id);
    hipSetDevice(src_id);
    hipDeviceSynchronize();
    hipSetDevice(dst_id);
    hipDeviceSynchronize();
}

template <>
void prep_iteration<Kind::GPUToHost>(char *ptr, size_t bytes, int src_id, int /*dst_id*/) {
    flush_all(ptr, bytes);
    hipSetDevice(src_id);
    hipMemPrefetchAsync(ptr, bytes, src_id);
    hipDeviceSynchronize();
}

template <>
void prep_iteration<Kind::HostToGPU>(char *ptr, size_t bytes, int /*src_id*/, int dst_id) {
    hipMemPrefetchAsync(ptr, bytes, hipCpuDeviceId);
    hipSetDevice(dst_id);
    hipDeviceSynchronize();
}