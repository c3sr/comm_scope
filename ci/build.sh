set -x -e

source ci/env.sh

which g++
which nvcc
which cmake

g++ --version
nvcc --version
cmake --version

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DUSE_NUMA=$USE_NUMA \
  -DUSE_CUDA=$USE_CUDA \
  -DUSE_OPENMP=$USE_OPENMP \
  -DUSE_NVTX=$USE_NVTX
make VERBOSE=1 