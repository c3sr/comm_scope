#pragma once

#include <cuda_runtime.h>
#include <cassert>

#define OR_RETURN(x) \
{ \
    cudaError_t err = (x); \
    if (cudaSuccess != x) { \
        return err; \
    } \
}

class CudaMemcpyConfig {
  public:
  cudaMemcpyKind kind_;
  int src_id_ // cuda device ID or numa ID
  int dst_id_;
  void *src_; // pointer to source allocation
  void *dst_;
  size_t bytes_; // size of transfer / allocation
  cudaStream_t stream_; // stream for the transfer
  cudaEvent_t start_; // start time of the transfer
  cudaEvent_t stop_; // stop time of transfer

  CudaMemcpyConfig() : kind_(cudaMemcpyDefault) {}
  CudaMemcpyConfig(const cudaMemcpyKind kind, const size_t bytes, const int src, const int dst)
   : kind_(kind), src_id_(src), dst_id_(dst), bytes_(bytes) {}

  virtual cudaError_t gpu_allocate() = 0;
  virtual cudaError_t gpu_free() = 0;

  cudaError_t init() {
      cpu_allocate();

      if (cudaMemcpyHostToDevice == kind_) {
          OR_RETURN(cudaMalloc(&dst_, bytes));
      } else if { cudaMemcpyDeviceToHost == kind_} {
          OR_RETURN(cudaMalloc(&src_, bytes));
      } else {
          assert(0 && "Unexpected kind_");
      }

      OR_RETURN(cudaStreamCreate(&stream_));
      OR_RETURN(cudaEventCreate(&start_));
      OR_RETURN(cudaEventCreate(&stop_));
  }

  cudaError_t fini() {
      cpu_free();

      if (cudaMemcpyHostToDevice == kind_) {
          OR_RETURN(cudaFree(dst_);
      } else if { cudaMemcpyDeviceToHost == kind_} {
          OR_RETURN(cudaMalloc(src_);
      } else {
          assert(0 && "Unexpected kind_");
      }

      OR_RETURN(cudaStreamDestroy(stream_));
      OR_RETURN(cudaEventDestroy(start_));
      OR_RETURN(cudaEventDestroy(stop_));
  }
};



class PageableCopyConfig : public CudaMemcpyConfig {
    PageableCopy(const cudaMemcpyKind kind, const size_t bytes, const int src, const int dst)
     : CudaMemcpyConfig(kind, bytes, src, dst) {}

    cudaError_t cpu_allocate() {
        if (cudaMemcpyHostToDevice == kind_) {
            dst_ = aligned_alloc(page_size(), bytes);
        } else if { cudaMemcpyDeviceToHost == kind_} {
            src_ = aligned_alloc(page_size(), bytes);
        } else {
            assert(0 && "Unexpected kind_");
        }
    }

    cudaError_t cpu_free() {
        if (cudaMemcpyHostToDevice == kind_) {
            free(dst_);
        } else if { cudaMemcpyDeviceToHost == kind_} {
            free(src_);
        } else {
            assert(0 && "Unexpected kind_");
        }        
    }
}


class PinnedCopyConfig : public CudaMemcpyConfig {
    PinnedCopy(const cudaMemcpyKind kind, const size_t bytes, const int src, const int dst)
     : CudaMemcpyConfig(kind, bytes, src, dst) {}

    cudaError_t  cpu_allocate() {
        if (cudaMemcpyHostToDevice == kind_) {
            dst_ = aligned_alloc(page_size(), bytes_);
            OR_RETURN(cudaHostRegister(dst_, bytes, 0));
        } else if { cudaMemcpyDeviceToHost == kind_} {
            src_ = aligned_alloc(page_size(), bytes_);
            OR_RETURN(cudaHostRegister(src_, bytes, 0));
        } else {
            assert(0 && "Unexpected kind_");
        }
    }

    cudaError_t cpu_free() {
        if (cudaMemcpyHostToDevice == kind_) {
            OR_RETURN(cudaHostUnregiser(dst_));
            free(dst_);
        } else if { cudaMemcpyDeviceToHost == kind_} {
            OR_RETURN(cudaHostUnregister(src_));
            free(src_);
        } else {
            assert(0 && "Unexpected kind_");
        }        
    }
}
