FROM rocm/dev-ubuntu-22.04:5.4-complete

# Install NUMA
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Add source
COPY . /opt/comm_scope
WORKDIR /opt/comm_scope

# Build
RUN mkdir -p build \
  && cd build \
  && cmake .. \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DSCOPE_USE_NUMA=ON \
    -DSCOPE_USE_OPENMP=ON \
    -DSCOPE_ARCH_MI250X=ON \
    -DSCOPE_USE_NVTX=OFF \
  && make

