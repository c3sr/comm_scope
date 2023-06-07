#pragma once

#include <cassert>
#include <hip/hip_runtime.h>
#include <stdexcept>

#define OR_RETURN(x)                                                           \
  {                                                                            \
    hipError_t err = (x);                                                      \
    if (hipSuccess != err) {                                                   \
      return err;                                                              \
    }                                                                          \
  }

class HipMemcpyConfig {
public:
  hipMemcpyKind kind_;
  int src_id_; // device ID or numa ID
  int dst_id_;
  void *src_; // pointer to source allocation
  void *dst_;
  hipStream_t stream_; // stream for the transfer
  hipEvent_t start_;   // start time of the transfer
  hipEvent_t stop_;    // stop time of transfer

  HipMemcpyConfig() : kind_(hipMemcpyDefault) {}
  HipMemcpyConfig(const hipMemcpyKind kind, const int src, const int dst)
      : kind_(kind), src_id_(src), dst_id_(dst) {}

  virtual hipError_t cpu_allocate(const size_t bytes) = 0;
  virtual hipError_t cpu_free() = 0;

  hipError_t init(const size_t bytes) {

    OR_RETURN(cpu_allocate(bytes));

    if (hipMemcpyHostToDevice == kind_) {
      OR_RETURN(hipMalloc(&dst_, bytes));
      OR_RETURN(hipMemset(dst_, 1, bytes));
    } else if (hipMemcpyDeviceToHost == kind_) {
      OR_RETURN(hipMalloc(&src_, bytes));
      OR_RETURN(hipMemset(src_, 1, bytes));
    } else {
      throw std::logic_error("unexpected kind");
    }

    OR_RETURN(hipStreamCreate(&stream_));
    OR_RETURN(hipEventCreate(&start_));
    return hipEventCreate(&stop_);
  }

  hipError_t fini() {
    OR_RETURN(cpu_free());

    if (hipMemcpyHostToDevice == kind_) {
      OR_RETURN(hipFree(dst_));
    } else if (hipMemcpyDeviceToHost == kind_) {
      OR_RETURN(hipFree(src_));
    } else {
      throw std::logic_error("unexpected kind");
    }

    OR_RETURN(hipStreamDestroy(stream_));
    OR_RETURN(hipEventDestroy(start_));
    OR_RETURN(hipEventDestroy(stop_));
    return hipSuccess;
  }

  hipError_t reset_gpu() {
    if (hipMemcpyHostToDevice == kind_) {
      return scope::hip_reset_device(dst_id_);
    } else if (hipMemcpyDeviceToHost == kind_) {
      return scope::hip_reset_device(src_id_);
    } else {
      throw std::logic_error("unexpected kind");
    }
    return hipSuccess;
  }
};

class PageableCopyConfig : public HipMemcpyConfig {
public:
  PageableCopyConfig(const hipMemcpyKind kind, const int src, const int dst)
      : HipMemcpyConfig(kind, src, dst) {}

  hipError_t cpu_allocate(const size_t bytes) {

    if (hipMemcpyHostToDevice == kind_) {
      numa::bind_node(src_id_);
      src_ = aligned_alloc(page_size(), bytes);
      std::memset(src_, 0, bytes);
    } else if (hipMemcpyDeviceToHost == kind_) {
      numa::bind_node(dst_id_);
      dst_ = aligned_alloc(page_size(), bytes);
      std::memset(dst_, 0, bytes);
    } else {
      throw std::logic_error("unexpected kind");
    }
    numa::bind_node(-1);
    return hipSuccess;
  }

  hipError_t cpu_free() {
    if (hipMemcpyHostToDevice == kind_) {
      free(src_);
    } else if (hipMemcpyDeviceToHost == kind_) {
      free(dst_);
    } else {
      throw std::logic_error("unexpected kind");
    }
    return hipSuccess;
  }
};

class PinnedCopyConfig : public HipMemcpyConfig {
public:
  PinnedCopyConfig(const hipMemcpyKind kind, const int src, const int dst)
      : HipMemcpyConfig(kind, src, dst) {}

  hipError_t cpu_allocate(const size_t bytes) {
    if (hipMemcpyHostToDevice == kind_) {
      numa::bind_node(src_id_);
      OR_RETURN(hipHostMalloc(&src_, bytes, 0));
      std::memset(src_, 1, bytes);
    } else if (hipMemcpyDeviceToHost == kind_) {
      numa::bind_node(dst_id_);
      OR_RETURN(hipHostMalloc(&dst_, bytes, 0));
      std::memset(dst_, 1, bytes);
    } else {
      throw std::logic_error("unexpected kind");
    }
    numa::bind_node(-1);
    return hipSuccess;
  }

  hipError_t cpu_free() {
    if (hipMemcpyHostToDevice == kind_) {
      OR_RETURN(hipHostFree(src_));
    } else if (hipMemcpyDeviceToHost == kind_) {
      OR_RETURN(hipHostFree(dst_));
    } else {
      throw std::logic_error("unexpected kind");
    }
    return hipSuccess;
  }
};

#undef OR_RETURN
