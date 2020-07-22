set -x -e

source ci/env.sh

# Install Cmake if it doesn't exist
mkdir -p $CMAKE_PREFIX
if [[ ! -f $CMAKE_PREFIX/bin/cmake ]]; then
    if [[ $TRAVIS_CPU_ARCH == "ppc64le" ]]; then
        wget -qSL https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz -O cmake.tar.gz
        tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
        rm cmake.tar.gz
        cd $CMAKE_PREFIX
        ./bootstrap --prefix=$CMAKE_PREFIX
        make -j `nproc` install
    elif [[ $TRAVIS_CPU_ARCH == "amd64" ]]; then
        wget -qSL https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz -O cmake.tar.gz
        tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
        rm cmake.tar.gz
    fi
fi
cd $HOME

sudo apt-get update 

## install NUMA
if [[ $USE_NUMA != "0" ]]; then
sudo apt-get install -y --no-install-recommends --no-install-suggests \
  libnuma-dev 
fi

## install CUDA
if [[ $USE_CUDA != "0" ]]; then
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

    if [[ $TRAVIS_CPU_ARCH == "ppc64le" ]]; then
        CUDA102="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-repo-ubuntu1804_10.2.89-1_ppc64el.deb"
        CUDA110="https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_ppc64el.deb"
    elif [[ $TRAVIS_CPU_ARCH == "amd64" ]]; then
        CUDA102="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb"
        CUDA110="https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb"
    fi

    if [[ $CUDA_VERSION == "102" ]]; then
        wget -qSL $CUDA102 -O cuda.deb
        sudo dpkg -i cuda.deb
        sudo apt-get update 
        sudo apt-get install -y --no-install-recommends \
        cuda-toolkit-10-2
    else 
        wget -qSL $CUDA110 -O cuda.deb
        sudo dpkg -i cuda.deb
        sudo apt-get update 
        sudo apt-get install -y --no-install-recommends \
        cuda-toolkit-11-0
    fi
fi