# XET VFS — v2 Architecture (composable, `fileReadPositional`-driven)

## Goal

Replace the current XET integration with one whose top-level contract is
exactly the existing ZML VFS contract: a `fileReadPositional(file, data,
offset)` that serves bytes from a remote Xet-backed file. No new interface
on the consumer side. PJRT-side `MemoryWriter` and `safetensors.TensorReader`
stay untouched; they pull bytes through `streamRemaining` like they already
do for plain HTTP/HF files.

XET protocol optimality is explicitly **not** the priority — composability
with VFS is. Optimizations are local to the XET backend and never leak into
the VFS or `zml.io.load` layer.

## Pinned decisions (scoping discussion, 2026-06-08)

| # | Decision |
|---|----------|
| 1 | Per-file XET state lives **inline on `hf.Handle`** as a tagged variant (`xet` vs `http`). |
| 2 | Reconstruction is done **once per `(repo, file)`**, lazily on first read. The `Plan` is owned by `xet.State` (process-wide cache); the Handle only caches a borrowed `*const Plan` to skip the lock on the hot path. |
| 3 | Decompression buffering: a **single decompressed chunk** (~64 KiB) ring per Handle, refilled chunk-by-chunk. |
| 4 | Xorb cache scope: **per-`xet.State`** (process-wide), shared across all Handles. |
| 5 | Detection stays in [hf.zig](../../zml/io/vfs/hf.zig) via the existing `X-Xet-Hash` probe at `dirOpenFile` time. |
| 6 | Reader model is **synchronous inside `fileReadPositional`** in v1. Prefetch is a follow-up after end-to-end validation. |

## Top-level contract

```
caller (TensorReader → streamRemaining → MemoryWriter)
        │
        ▼
VFS.fileReadPositional(handle, data, offset)
        │
        ▼  (hf.zig dispatch by Handle variant)
        │
        ├─ http: existing HTTP range GET path (unchanged)
        └─ xet:  new XET path (this doc)
```

The caller decides offset and length. The XET backend never produces
unsolicited bytes, never owns a Writer, never schedules its own threads in
v1. It is a pure "give me N bytes at offset X" service.

## Components

### `hf.Handle.Variant` (inline tagged union)

```
Variant = union(enum) {
    http: HttpFileState,   // existing fields
    xet:  XetFileState,    // new (this doc)
}
```

- Detection result at `dirOpenFile` chooses the variant.
- Lifetime tied to the Handle: closed via the existing `fileClose`.

### `XetFileState` (per-Handle, single-threaded use)

Owns:
- `file_id: [32]u8` — from `X-Xet-Hash` header at open.
- `repo: hf.Repo` reference (for token + reconstruction call).
- `plan: ?*const Plan` — `null` until first read. On first read, the
  Handle asks `xet.State.getOrBuildPlan(repo, file_id, file_size)` and
  caches the returned pointer. The `Plan` itself lives in `xet.State`;
  the Handle never owns it, never frees it. Subsequent reads dereference
  the cached pointer with **no lock**.
- `chunk_ring: ChunkRing` — one decompressed-chunk buffer + bookkeeping for
  the chunk it currently holds (xorb hash, chunk index, file-offset span).
- Small scratch buffers for HTTP body + LZ4 input (re-used across calls).

No mutex on `XetFileState` itself: one positional call at a time per
Handle (current `load()` usage opens one Handle per tensor; if that ever
changes, a Handle-level mutex gets added here, not in the XET protocol
code).

### `Plan` (per `(repo, file)`, owned by `xet.State`)

Built once by calling `xet.State.reconstruct(repo, 0, file_size)`:
- `terms: []Term` — file-order, each carries xorb hash, chunk range,
  `byte_skip`, `unpacked_length`.
- `term_prefix: []u64` — prefix sum of `unpacked_length` for O(log N) lookup
  of the term covering a given file offset.
- `fetch_info: HashMap(xorb_hash → []FetchUrl)` — URL choices per xorb.

Immutable once built, so safe to hand out `*const Plan` pointers to many
Handles concurrently without further synchronization.

### `xet.State` (process-wide, shared)

Existing struct in [zml/io/xet.zig](../../zml/io/xet.zig) gains/keeps:
- `(repo, path) → file_id` cache (already exists).
- `(repo) → CasAuth` cache with TTL refresh (already exists, TTL pending).
- **`plan_cache: HashMap((repo, file_id) → *Plan)`** — mutex-protected.
  `getOrBuildPlan` is the single entry point; first caller builds and
  inserts, subsequent callers receive the cached pointer. Plans live for
  the lifetime of the client.
- **`xorb_cache: XorbCache`** — process-wide, mutex-protected, LRU on
  decompressed-xorb-or-compressed-xorb bytes (decision to keep raw vs
  decompressed deferred; we already have `XorbCache` infra).

All three caches use independent mutexes — they're never held together.

### `ChunkRing` (per Handle)

Holds at most one decompressed chunk plus metadata identifying it
(`xorb_hash`, `chunk_index`, `file_offset_start`, `file_offset_end`). A
`fileReadPositional(offset, len)` call:

1. binary-search `term_prefix` → term index + intra-term offset.
2. while bytes left to deliver:
   a. compute `(xorb, chunk_index, intra_chunk_offset)` from the term.
   b. if the ring already holds that chunk: copy out, advance.
   c. else: fetch the chunk (via xorb cache if present, else HTTP range GET
      of the xorb covering that chunk + `ChunkIterator` to land on the
      right one), LZ4-decompress into the ring, then copy out.

Sequential reads (the `streamRemaining` case) hit the ring on every call
after the first chunk fault. Random reads pay one fault per chunk.

### `XorbCache` (on `xet.State`)

- Already exists in the cleanup branch. Keep its shape.
- Stores raw fetched xorb byte ranges keyed by xorb hash + byte range.
- LRU eviction with a fixed byte budget (e.g. 256 MiB, configurable).
- Mutex-protected.
- ChunkRing consults it before issuing a network GET.

## Read flow (concrete)

```
fileReadPositional(handle, [dst0, dst1, ...], offset):
    state = handle.variant.xet
    if state.plan is null:
        // first call: pay one mutex hit on xet.State; subsequent calls
        // dereference the cached pointer with no lock.
        state.plan = xet_client.getOrBuildPlan(state.repo, state.file_id, file_size)
    plan = state.plan   // *const Plan, immutable

    cursor = offset
    for dst in [dst0, dst1, ...]:
        while bytes_left_in(dst) > 0:
            (term, intra) = plan.locate(cursor)
            (xorb, chunk_idx, intra_chunk) = term.map(intra)
            chunk = state.ring.ensure(xorb, chunk_idx,
                                      via xet_client.xorb_cache,
                                      via HTTP range GET if miss)
            n = min(bytes_left_in(dst), chunk.len - intra_chunk)
            copy(chunk[intra_chunk..][..n], dst[written..])
            cursor += n
            written += n
    return total_written
```

All work is synchronous. All concurrency comes from the caller (`LimitedGroup`)
opening multiple Handles in parallel — and is absorbed by the `xet.State`
mutexes around its caches.

## What we are NOT doing in v1

- No prefetch thread, no async fetcher. Latency is consumed serially.
- No range reconstruction (`reconstruct(start, len)`); always whole-file.
- No cross-tensor Handle reuse; one Handle per `TensorReader` like today.
- No streaming LZ4 (we decompress chunk-at-a-time, the chunk header in the
  xorb tells us the compressed extent, so we don't need streaming).
- No write path. XET is read-only here.

## Memory ceiling estimate

With `parallelism = P` and the v1 layout:
- Per-Handle steady state: `O(64 KiB)` ring + `O(plan size)` (~10s of KiB
  per safetensors shard) + transient `O(xorb size, ≤64 MiB)` if we have to
  fetch fresh.
- Per `xet.State`: `xorb_cache` budget (configurable, default 256 MiB).
- Total: `P × (64 KiB + plan) + 256 MiB`.

The 64 MiB transient only exists during a fetch and is then either copied
into the cache (if budget allows) or freed. Worst case during a fetch:
`P × 64 MiB + 256 MiB`. With `P = 8` → ~768 MiB peak. Acceptable; reducible
later by streaming chunks out of the HTTP body directly.

## Failure / retry policy (v1)

- Reconstruction call: one retry on transient HTTP, then bubble up as
  `error.XetReconstructionFailed`.
- Xorb fetch: one retry on a different `FetchUrl` for the same xorb if
  available, else bubble up.
- LZ4 / chunk header parse failure: no retry, returns `error.CorruptXorb`.
- Token expiry mid-fetch: refresh token via `xet.State.casAuth`, retry the
  fetch once.

## Implementation plan (atomic increments)

Each step is independently testable; tests live next to the code, run via
`bazel test //zml/io:test` unless noted.

### Step 1 — Plan struct + locator

- Add `Plan`, `Plan.locate(offset)` and `Plan.fromReconstruct(resp)` in
  [zml/io/xet.zig](../../zml/io/xet.zig).
- Pure-function tests with synthetic terms (no network).
- Acceptance: locator returns correct `(term_index, intra_term_offset)` for
  edge cases (offset 0, last byte, term boundaries, beyond file end →
  error).

### Step 2 — ChunkRing (single-chunk decompressor)

- Add `ChunkRing` in [zml/io/xet.zig](../../zml/io/xet.zig).
- `ensure(xorb, chunk_idx, fetcher)` where `fetcher` is a closure that
  returns the raw xorb bytes covering at least that chunk.
- Unit tests with a fake fetcher returning a synthetic xorb (reuse
  `examples/xet_cas/util.zig` helpers if useful; otherwise inline test xorb).
- Acceptance: sequential reads issue one fetch per xorb; repeated reads of
  the same chunk issue zero fetches.

### Step 3 — `XetFileState` wiring on `hf.Handle`

- Turn `hf.Handle` fields into a tagged variant `http | xet`.
- `dirOpenFile`: on `X-Xet-Hash` present, build the `xet` variant
  (file_id + repo + lazily-`null` Plan + fresh ChunkRing). Otherwise keep
  current `http` path.
- `fileClose`: free per-variant state.
- Acceptance: existing HF VFS tests still pass; new variant compiles and is
  selected on a known XET file.

### Step 4 — `fileReadPositional` for the xet variant

- Dispatch on variant.
- On first call: `state.plan = xet_client.getOrBuildPlan(repo, file_id, file_size)`
  (single entry point, mutex-protected, builds via
  `xet.State.reconstruct` on miss, inserts into `plan_cache`, returns
  `*const Plan`).
- Loop: locate → ring.ensure → copy. No xorb cache yet; fetcher is a direct
  HTTP range GET.
- Acceptance: SHA-256 match against LFS oracle on the regression pair
  (Llama-3-70B `lm_head.weight` + AWQ-INT4 `q_proj.qweight`) with
  `parallelism = 1`.

### Step 5 — Wire `xet.State.xorb_cache` into ChunkRing fetcher

- Fetcher first consults `xorb_cache`, falls back to HTTP, populates cache.
- Acceptance: cross-tensor reuse measured on a real safetensors shard — net
  bytes for the second tensor in the same file drops to near zero for
  shared xorbs. Regression SHA still matches.

### Step 6 — Concurrency hardening

- Run regression with `parallelism = 8`. Add mutexes on `xet.State` caches
  if not already present. No prefetch, no per-Handle async.
- Acceptance: clean run with `tsan`-equivalent scrutiny (loom-style review,
  no actual TSan available), SHA matches.

**Status (done for the cache primitives):**
- `xet.State.WindowCache` now uses `std.Io.Mutex` plus a per-entry `pin`
  counter. `acquire`/`release` is the new API (HTTP fill happens unlocked
  while pinned; eviction skips pinned entries).
- `xet.State.bytes_fetched` is `std.atomic.Value(u64)` with monotonic
  fetch-add on miss.
- `hf.httpRangeGetIntoSlot` retries once on transient HTTP errors
  (`HttpConnectionClosing | ConnectionResetByPeer | UnexpectedReadFailure`).
- Acceptance unit test `WindowCache: parallel acquire/release stays
  consistent` (8 threads × 2000 iters, 16 KiB cap, 6 keys) verifies no
  torn writes, balanced pins, consistent `used_bytes`. Runs via
  `bazel test //zml/io/vfs:test`.
- Single-thread regression unchanged (Llama-3-70B BF16 shard 30: 1808 MB,
  ratio 0.861, ~19s).

**End-to-end acceptance:** `examples/xet_cas/test_vfs_parallel.zig` reads
4 Llama-3-70B shards concurrently through one shared `hf_vfs` (one
shared `xet.State`, one shared `WindowCache`). Latest run: 16 014 MB
read / 13 640 MB fetched, ratio 0.852, wall = 48.6 s (≈ longest-shard
time → real concurrency). All 4 SHA-256 match LFS oracle.

**Driver isolation:** the oracle SHA phase uses a separate
`std.http.Client` from the parallel phase. `std.http.Client` itself is
thread-safe for concurrent `request()` (verified via
`test_http_parallel.zig`); the issue we hit was a stale keep-alive
connection from the parallel phase being reused after a long idle gap,
which blocks forever on `readv` because there is no socket read
deadline. That is a defense-in-depth follow-up on `std.http.Client`,
not a Step 6 issue.

**Remaining concurrency caveats:** the lazy caches `file_id_cache` /
`cas_cache` / `plan_cache` on `xet.State` are still unprotected. The
pre-warm pattern in the test driver covers them; concurrent
`dirOpenFile` on different files in production would race. Tracked as a
separate hardening task.

### Step 7 — Replace old XET path

- Remove the legacy XET integration from [hf.zig](../../zml/io/vfs/hf.zig)
  and from `examples/xet_cas/test_file_to_device.zig` (or migrate the
  example to drive the VFS API instead of poking the protocol directly).
- Acceptance: `bazel test //zml/io:test //zml:test` PASS; end-to-end LLM
  load (e.g. SmolLM2-360M) succeeds via VFS only.

### Step 8 (post-v1) — Prefetch

- Add a 1-xorb look-ahead in `XetFileState`: when a chunk is served,
  schedule the next-needed-but-not-cached xorb fetch on a background
  worker. Strictly internal; the `fileReadPositional` contract is unchanged
  (it just hits the cache more often).
- Validate the speedup we historically measured (workers=4 ≈ 2.6×).

## Out of scope

- Write / upload path.
- Range reconstruction (`reconstruct(start, len)`).
- Cross-process xorb cache (disk persistence).
- Replacing `XorbCache` with a streaming chunk store.
- Token lifetime hardening beyond what `xet.State.casAuth` already does.

## Open questions deferred to implementation

- Should `xorb_cache` hold compressed or decompressed bytes? v1 keeps the
  current raw/compressed shape; revisit after Step 5 numbers.
- Plan size for very large monolithic files (>10 GiB) hitting the HF Xet
  protocol cap on a single reconstruction response — split the call. Not a
  v1 concern; safetensors shards are typically ≤5 GiB.
