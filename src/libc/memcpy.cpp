#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "libc_memcpy_NUMAToNUMA"

auto libc_memcpy_NUMAToNUMA = [](benchmark::State &state, const int src_id,
                                 const int dst_id, const bool flush) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  // allocate
  void *src = numa::alloc_onnode(bytes, src_id);
  defer(numa::free(src, bytes));
  void *dst = numa::alloc_onnode(bytes, dst_id);
  defer(numa::free(dst, bytes));

  // touch
  {
    numa::ScopedBind sb(src_id);
    std::memset(src, 1, bytes);
  }
  {
    numa::ScopedBind sb(dst_id);
    std::memset(dst, 1, bytes);
  }

  int i = 1;
  for (auto _ : state) {
    state.PauseTiming();
    {
      numa::ScopedBind sb(src_id);
      std::memset(src, i, bytes);
      if (flush) {
        flush_all(src, bytes);
      }
    }
    {
      numa::ScopedBind sb(dst_id);
      std::memset(dst, i + 1, bytes);
      if (flush) {
        flush_all(dst, bytes);
      }
    }
    {
      numa::ScopedBind sb(src_id);
      state.ResumeTiming();
      std::memcpy(dst, src, bytes);
      state.PauseTiming();
    }
    i += 2;
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_id"] = src_id;
  state.counters["dst_id"] = dst_id;
};

static void registerer() {
  LOG(trace, NAME " registerer...");

  if (numa::available()) {
    std::vector<MemorySpace> numaSpaces =
        scope::system::memory_spaces(MemorySpace::Kind::numa);

    std::string name;
    for (const auto &srcNuma : numaSpaces) {
      for (const auto &dstNuma : numaSpaces) {

        const int src_id = srcNuma.numa_id();
        const int dst_id = dstNuma.numa_id();

        if (numa::can_execute_in_node(src_id)) {
          name = std::string(NAME) + "/" + std::to_string(src_id) + "/" +
                 std::to_string(dst_id);
          benchmark::RegisterBenchmark(name.c_str(), libc_memcpy_NUMAToNUMA,
                                       src_id, dst_id, false)
              ->BYTE_ARGS();
          name = std::string(NAME) + "_flush/" + std::to_string(src_id) + "/" +
                 std::to_string(dst_id);
          benchmark::RegisterBenchmark(name.c_str(), libc_memcpy_NUMAToNUMA,
                                       src_id, dst_id, true)
              ->BYTE_ARGS();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
