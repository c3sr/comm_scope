FROM nvidia/cuda:11.5.1-devel-ubuntu20.04

# Install NUMA
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Rebuild binary
RUN cd build \
  && cmake \
    .. \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_CUDA ON \
    -DSCOPE_USE_NVTX ON \
  && make

