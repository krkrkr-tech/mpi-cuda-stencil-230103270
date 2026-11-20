# Cache Line and False Sharing Analysis

Run identity: `krkrkr-tech`; CPU: `AMD64::AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD::16`; cache line: `64 bytes`.

## Q1 — MESI transitions
With stride 1, both int64 values are in one 64-byte line. Core 0 first obtains the line in E (or M after its store). Core 1's read observes the line through the coherence fabric, making both copies S; its store issues an RFO, invalidates Core 0's copy (I), and obtains M. Core 0 then repeats the RFO/invalidation cycle. The store buffer cannot retire independently of ownership, so repeated RFOs stall it.

## Q2 — measured overhead
| regime | elapsed ns | ns/update | slowdown vs stride 8 |
|---|---:|---:|---:|
| stride 1 | 1062816000 | 4.428 | 576.052x |
| stride 4 | 2046700 | 0.009 | 1.109x |
| stride 8 | 1845000 | 0.008 | 1.000x |
| stride 16 | 1676400 | 0.007 | 0.909x |
| register-private | 249016000 | 1.038 | 134.968x |

Stride 1 is `576.052x` the stride-8 duration.

## Q3 — L2 spatial prefetching
Stride 8 separates workers by one 64-byte line. Stride 16 separates them by 128 bytes, avoiding adjacent-line pairing in implementations whose L2 prefetcher fetches 128-byte sectors. Stride 8 can still see secondary invalidations when adjacent lines are prefetched into a shared sector or when allocator placement causes another active object to share a neighboring line.

## Q4 — LLVM and stores
The loop has a loop-carried dependency because each iteration reads the value written by the preceding iteration. The audit therefore requires a native i64 store (`True`) and no Python-object fallback (`False`). Hoisting the value into a register would remove the required per-iteration memory interaction and is not valid for this shared-memory benchmark.

## Q5 — GIL contrast
Standard Python bytecode threads serialize on the GIL, so they do not execute simultaneous multi-core stores and cannot expose hardware false sharing. More threads instead add lock handoffs and scheduling overhead. Numba's `nogil=True` kernel releases the GIL, allowing native stores to contend through cache coherence.
