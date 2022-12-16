FROM rocm/dev-ubuntu-22.04:5.4-complete

# Install NUMA
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Build
RUN mkdir -p build \
  && cd build \
  && cmake .. \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_HIP=ON \
    -DSCOPE_USE_NVTX=OFF \
  && make

