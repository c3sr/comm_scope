FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install NUMA
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# get a recent cmake
RUN mkdir -p /opt
WORKDIR /opt
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.tar.gz
RUN tar -xf cmake-3.25.1-linux-x86_64.tar.gz

# Add source
COPY . /opt/comm_scope
WORKDIR /opt/comm_scope

# Build
RUN mkdir -p build \
  && cd build \
  && /opt/cmake-3.25.1-linux-x86_64/bin/cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_CUDA=ON \
    -DSCOPE_USE_NVTX=ON \
  && make

