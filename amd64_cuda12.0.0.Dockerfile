FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install NUMA
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Rebuild binary
RUN mkdir -p build \
  && cd build \
  && cmake .. \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_CUDA=ON \
    -DSCOPE_USE_NVTX=ON \
  && make

