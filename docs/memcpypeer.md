# `cudaMemcpyPeer` Bandwidth

These benchmarks examine memcpy bandwidth achieved through explicit `cudaMemcpyPeerAsync` calls.

## Implementations

| `--benchmark_filter=`|Description|Argument Format|
|-|-|-|
| `Comm_MemcpyPeer_Peer`        | GPU to GPU with peer access enabled | `log2 size` |
| `Comm_MemcpyPeer`             | GPU to GPU with peer access disabled | `log2 size` |
| `Comm_Duplex_MemcpyPeer_Peer` | GPU to GPU bidirectional with peer access enabled | `log2 size` |
| `Comm_Duplex_MemcpyPeer`      | GPU to GPU bidirectional with peer access disabled | `log2 size` |


## Technique

Whether the benchmark has one transfer or two, a source and destination allocation are established for each transfer.

For the unidirectional transfer, a call to `cudaMemcpyPeerAsync` is surrounded by two `cudaEventRecord`s, and then the time between those events is computed to determine the transfer time.

For the bidirectional transfer, a start event is recorded before the first issued `cudaMemcpyPeerAsync`.
Then, the second `cudaMemcpyPeerAsync` is issued and a first stop event is recorded.
Then, the first stream waits for the first stop event and records a second stop event.
The reported time is the difference between the start event and the second stop event.


