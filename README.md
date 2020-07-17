# Comm|Scope

[![Build Status](https://travis-ci.com/c3sr/comm_scope.svg?branch=master)](https://travis-ci.com/c3sr/comm_scope)

CUDA- and NUMA-Aware Multi-CPU / Multi-GPU communication benchmarks.

Docker images are [available](https://hub.docker.com/r/c3sr/comm_scope/) on Docker Hub.

## Prerequisites

* CMake 3.17+
* g++ >= 4.9
* CUDA toolkit >= 8.0

## Getting started

```
<dowload or clone Comm|Scope>
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make
./comm_scope --benchmark_list_tests=true
```

To choose specific benchmarks, filter by regex:

```
./comm_scope --benchmark_list_tests=true --benchmark_filter=<regex>
```

To limit the visible GPUs, use the `--cuda` option:

```
./comm_scope --cuda 0 --cuda 1
```

To limit the visible NUMA nodes, use the `--numa` option:

```
./comm_scope --numa 8
```

## OLCF Summit

Get a launch node: `bsub -W 2:00 -nnodes 1 -P csc362 -Is /bin/zsh`

You will need to load a newer gcc, as well as CUDA, to build **and run** Comm|Scope
```
module load cuda
module load gcc/5.4.0
```

Then to build, do

```
cmake ..
make -j
```

Example job submission scripts are in `scripts/summit`.

You may ignore messages like the following, where libscope fails to control CPU turbo and the governor.
On managed systems like Summit, this is not necessary.

```
[2020-07-15 17:58:00.763] [scope] [error] unable to disable CPU turbo: no permission. Run with higher privileges?
[2020-07-15 17:58:00.763] [scope] [error] unable to set OS CPU governor to maximum: no permission. Run with higher privileges?
```

## FAQ / Troubleshooting

** I get `CMake Error: Remove failed on file: <blah>: System Error: Device or resource busy`**

This somtimes happens on network file systems. You can retry, or do the build on a local disk.

** I get `-- The CXX compiler identification is GNU 4.8.5` after `module load gcc/5.4.0`.

A different version of GCC may be in the CMake cache.
Try running `cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc`, or deleting your build directory and restarting.

## Bumping the Version

Update the changelog and commit the changes.

Install bump2version

```pip install --user bump2version```

Check that everything seems good (minor version, for example)

```bump2version --dry-run minor --verbose```

Actually bump the version

```bump2version minor```

Push the changes

```git push && git push --tags```

## Contributing

Any work on the underlying `cwpearson/libscope` library will probably benefit from changing the submodule from http to SSH:

```
cd thirdparty/libscope
git set remote-url origin git@github.com:cwpearson/libscope.git
```

## Contributors

* [Carl Pearson](mailto:pearson@illinois.edu)
* [Sarah Hashash](mailto:hashash2@illinois.edu)

# Changelog

## v0.11.0 (July 17 2020)
* cwpearson/libscope v1.1.0
* `cudaGraphInstantiate` and `cudaGraphLaunch`
* Reduce maximum `cudaMemcpyPeerAsync` size, since it is not truly async above ~2^27 which breaks the measurement strategy.

## v0.10.0 (June 23 2020)
* Rely on `cwpearson/libscope` instead of `c3sr/scope`
* `cwpearson/libscope` v1.0.0
* Remove dependence on sugar
* Add 3D strided memory transfer benchmarks
* Add CUDA runtime microbenchmarks
* Remove some duplicate NUMA-/non-NUMA-aware implementations of cudaMemcpyAsync benchmarks

## v0.9.0 (June 5 2020)

* Add CPU-GPU and GPU-GPU sparse data transfer benchmarks
  * `cudaMemcpy3DAsync`
  * `cudaMemcpy3DPeerAsync`
  * `cudaMemcpy2DAsync`
  * custom 3D kernel
  * pack / `cudaMemcpyPeerAsync` / unpack

## v0.8.2 (March 6 2020)

* Fix a event-device mismatch in multi-GPU unidirectional `cudaMemcpyPeer` benchmarks

## v0.8.1 (March 5 2020)

* Disable peer access in non-peer `cudaMemcpyPeer` benchmarks

## v0.8.0 (March 5 2020)

* Add `cudaMemcpyPeer` uni/bidirectional benchmarks.

## v0.7.2 (April 8 2019)

* Add memory to the clobber list for for x86 and ppc64le cache flushing.

## v0.7.1 (April 5 2019)

* Add v0.7.0 and v0.7.1 changelog

## v0.7.0 (April 5 2019)

* Make POWER's cache flushing code match the linux kernel.
* rename "Coherence" benchmarks to "Demand"
* remove cudaStreamSynchronize from the timing path of zerocopy-duplex, demand-duplex, and prefetch-duplex
* Transition to better use of CMake's CUDA language support
* Use NVCC's compiler defines to check the CUDA version
* Disable Comm|Scope by default during Scope compilation

## v0.6.3 (Dec 20 2018)

* Add `USE_NUMA` CMake option
* Fix compile errors when USE_NUMA=0 or NUMA cannot be found 

## v0.6.2

* Fix checking non-existent cudaDeviceProp field in CUDA < 9

## v0.6.1

* Conform to updated SCOPE_REGSITER_AFTER_INIT

## v0.6.0

* Add unified memory allocation benchmarks
* Flush CPU caches in zero-copy benchmarks
* Add zerocopy duplex benchmarks
* Add unified memory prefetch duplex benchmark
* Add unified memory demand duplex benchmark
* Conform to updated SCOPE_REGSITER_AFTER_INIT

## v0.5.0

* Add zero-copy benchmarks
* Don't use nvToolsExt

## v0.4.0

* Add multithreaded Coherence GPU to Host benchmark
* Programatically register most benchmarks based on system configuration
* use cudaMemcpyAsync in numa-memcpy
* Add travis and Dockerfiles
* Use `aligned_alloc` in numa-memcpy/pinned-to-gpu
* Add x86 and POWER cache control functions

## v0.3.0

* Rework documentation
* Use `target_include_scope_directories` and `target_link_scope_libraries`.
* Use Clara for flags.
* Remove numa/rd and numa/wr.

## v0.2.0

* Add `--numa_ids` command line flag.
* Use `--cuda_device_ids` and --`numa_ids` to select CUDA and NUMA devices for benchmarks.



