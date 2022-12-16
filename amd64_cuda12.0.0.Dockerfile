FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install NUMA
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Add source
COPY . /opt/comm_scope
WORKDIR /opt/comm_scope

# Rebuild binary
RUN mkdir -p build \
  && cd build \
  && cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_CUDA=ON \
    -DSCOPE_USE_NVTX=ON \
  && make

