#pragma once

#include <cassert>
#include <cuda_runtime.h>

#define OR_RETURN(x)                                                           \
  {                                                                            \
    cudaError_t err = (x);                                                     \
    if (cudaSuccess != err) {                                                  \
      return err;                                                              \
    }                                                                          \
  }

class CudaMemcpyConfig {
public:
  cudaMemcpyKind kind_;
  int src_id_; // cuda device ID or numa ID
  int dst_id_;
  void *src_; // pointer to source allocation
  void *dst_;
  cudaStream_t stream_; // stream for the transfer
  cudaEvent_t start_;   // start time of the transfer
  cudaEvent_t stop_;    // stop time of transfer

  CudaMemcpyConfig() : kind_(cudaMemcpyDefault) {}
  CudaMemcpyConfig(const cudaMemcpyKind kind, const int src, const int dst)
      : kind_(kind), src_id_(src), dst_id_(dst) {}

  virtual cudaError_t cpu_allocate(const size_t bytes) = 0;
  virtual cudaError_t cpu_free() = 0;

  cudaError_t init(const size_t bytes) {

    OR_RETURN(cpu_allocate(bytes));

    if (cudaMemcpyHostToDevice == kind_) {
      OR_RETURN(cudaMalloc(&dst_, bytes));
      OR_RETURN(cudaMemset(dst_, 0, bytes));
    } else if (cudaMemcpyDeviceToHost == kind_) {
      OR_RETURN(cudaMalloc(&src_, bytes));
      OR_RETURN(cudaMemset(src_, 0, bytes));
    } else {
      assert(0 && "Unexpected kind_");
    }

    OR_RETURN(cudaStreamCreate(&stream_));
    OR_RETURN(cudaEventCreate(&start_));
    return cudaEventCreate(&stop_);
  }

  cudaError_t fini() {
    OR_RETURN(cpu_free());

    if (cudaMemcpyHostToDevice == kind_) {
      OR_RETURN(cudaFree(dst_));
    } else if (cudaMemcpyDeviceToHost == kind_) {
      OR_RETURN(cudaFree(src_));
    } else {
      assert(0 && "Unexpected kind_");
    }

    OR_RETURN(cudaStreamDestroy(stream_));
    OR_RETURN(cudaEventDestroy(start_));
    OR_RETURN(cudaEventDestroy(stop_));
    return cudaSuccess;
  }

  cudaError_t reset_gpu() {
    if (cudaMemcpyHostToDevice == kind_) {
      return scope::cuda_reset_device(dst_id_);
    } else if (cudaMemcpyDeviceToHost == kind_) {
      return scope::cuda_reset_device(src_id_);
    } else {
      assert(0 && "Unexpected kind_");
    }
    return cudaSuccess;
  }
};

class PageableCopyConfig : public CudaMemcpyConfig {
public:
  PageableCopyConfig(const cudaMemcpyKind kind, const int src, const int dst)
      : CudaMemcpyConfig(kind, src, dst) {}

  cudaError_t cpu_allocate(const size_t bytes) {

    if (cudaMemcpyHostToDevice == kind_) {
      numa::bind_node(src_id_);
      src_ = aligned_alloc(page_size(), bytes);
      std::memset(src_, 0, bytes);
    } else if (cudaMemcpyDeviceToHost == kind_) {
      numa::bind_node(dst_id_);
      dst_ = aligned_alloc(page_size(), bytes);
      std::memset(dst_, 0, bytes);
    } else {
      assert(0 && "Unexpected kind_");
    }
    numa::bind_node(-1);
    return cudaSuccess;
  }

  cudaError_t cpu_free() {
    if (cudaMemcpyHostToDevice == kind_) {
      free(src_);
    } else if (cudaMemcpyDeviceToHost == kind_) {
      free(dst_);
    } else {
      assert(0 && "Unexpected kind_");
    }
    return cudaSuccess;
  }
};

class PinnedCopyConfig : public CudaMemcpyConfig {
public:
  PinnedCopyConfig(const cudaMemcpyKind kind, const int src, const int dst)
      : CudaMemcpyConfig(kind, src, dst) {}

  cudaError_t cpu_allocate(const size_t bytes) {
    if (cudaMemcpyHostToDevice == kind_) {
      numa::bind_node(src_id_);
      src_ = aligned_alloc(page_size(), bytes);
      OR_RETURN(cudaHostRegister(
          src_, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
      std::memset(src_, 0, bytes);
    } else if (cudaMemcpyDeviceToHost == kind_) {
      numa::bind_node(dst_id_);
      dst_ = aligned_alloc(page_size(), bytes);
      OR_RETURN(cudaHostRegister(
          dst_, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
      std::memset(dst_, 0, bytes);
    } else {
      assert(0 && "Unexpected kind_");
    }
    numa::bind_node(-1);
    return cudaSuccess;
  }

  cudaError_t cpu_free() {
    if (cudaMemcpyHostToDevice == kind_) {
      OR_RETURN(cudaHostUnregister(src_));
      free(src_);
    } else if (cudaMemcpyDeviceToHost == kind_) {
      OR_RETURN(cudaHostUnregister(dst_));
      free(dst_);
    } else {
      assert(0 && "Unexpected kind_");
    }
    return cudaSuccess;
  }
};

#undef OR_RETURN
