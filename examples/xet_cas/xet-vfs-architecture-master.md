# XET VFS — proposed integration on master

Target branch: `master` (clean slate, ignore everything in `hugomano/xet` and
`oboulant/xet-pull-mode` except as a source of measured numbers).

This document captures the architecture that falls out of what we now
**measure** about XET on real model loads, rather than what the XET spec
*could* in principle deliver.

---

## What the measurements tell us

`scan_file_dedup` across four repos at scale:

| Repo | Size | Shards | dtype | chunks 1x per file | naive/minimal amort | on-wire/file | cross-shard amort |
|---|---|---|---|---|---|---|---|
| Llama-3.1-70B-AWQ-INT4 | 37.9 GB | 9 | INT4 | 99.99–100 % | 1.00 – 4.19× | 0.989 – 0.999 | 1.000× |
| Llama-3-70B | 134.6 GB | 30 | BF16 | 100 % | 1.17 – 1.42× | 0.855 – 0.873 | 1.042× |
| Qwen3.5-122B-A10B | 238.6 GB | 39 | BF16 | 100 % | 1.00 – 16.88× | 0.860 – 0.939 | 1.012× |
| LTX-2.3 22B | 44.0 GB | 1 | BF16 | 100 % | 9.65× | 0.871 | – |

Reading the table:

- **In-file chunk dedup is essentially zero.** Every shard scanned, every
  repo: 99.99 %–100 % of unique chunks are referenced exactly once *by the
  whole file*. The `XorbRangeCache` / per-chunk dedup machinery catches
  nothing on real model loads.
- **Cross-shard dedup is essentially zero.** 100 % of chunks unique
  across shards everywhere; cross-shard amortization 1.00–1.04×. The
  `XorbCache` cross-file cache earns nothing.
- **LZ4 wire compression: ~13–15 % on BF16, ~1 % on AWQ-INT4.** Real but
  modest. Confirmed across BF16 (Llama-3, Qwen3.5, LTX) and AWQ
  (Llama-3.1).
- **The reuse that *does* matter is structural, between adjacent or
  related tensor reads inside the same file.** Two patterns dominate:

  - **Small tensors share a chunk** (LTX-style). Per-tensor reuse hits
    counts of 3×, 5×, even 103× for individual chunks; xorbs are
    referenced by up to 604 tensors. Tensor sizes < chunk size →
    multiple tensors materialize from the same chunk.
  - **Same xorb appears in non-contiguous reconstruct terms** (Llama,
    Qwen). The `naive/minimal` column shows the cost of ignoring this:
    naively fetching each tensor independently transfers up to **16.88×**
    more bytes than fetching each xorb once (Qwen shard 39; 21× per
    Qwen shard 38; 4× typical AWQ shard). That entire factor disappears
    if a single in-flight GET serves all overlapping tensor reads.

So the win on master is **not dedup**. It is:
1. One `/reconstruct` call per file at open time (cheap, JSON, small).
2. **Parallel ranged CDN fetches per tensor** against that one plan.
3. **LZ4 decode** (~13–15 % wire savings on BF16; ~1 % on AWQ).
4. Two small mechanisms — *URL coalescer* and *refcounted chunk cache* —
   to absorb the structural reuse and recover the up-to-16× naive
   amortization gap without re-fetching or re-decoding.

Everything else (`XorbCache`, 2 GiB `WindowCache`, sequential cursor in
`Handle.XetState`, chunk-level dedup bookkeeping) is dead weight for this
workload and should not exist on master.

---

## What we keep from the spec / current branch

| Component | Why |
|---|---|
| `/reconstruct` call + JSON parsing | One per file, gives the full plan |
| Term → chunk → CDN-range math | Required for addressing, not dedup |
| LZ4 block decoder + ByteGrouping4 unshuffle | Required to read wire bytes |
| XET token manager (per-repo, refresh at exp-30s) | Required for auth |
| Per-file detection via `X-Xet-Hash` header on `/resolve/` | Cheap, correct |

## What we drop

| Component | Why drop |
|---|---|
| `XorbCache` (cross-file) | Cross-shard dedup ≈ 0 |
| `WindowCache` (2 GiB, per-xorb) | Was compensating for sequential cursor; not needed once plan is flat |
| Sequential `(term_index, chunk_idx, …)` cursor on Handle | O(N²) and mid-chunk-resume foot-gun (see `xet_protocol_caveats.md`) |
| Per-call chunk-level dedup bookkeeping | Catches nothing on model loads |
| In-file chunk dedup statistics in hot path | Useful for `scan_file_dedup`, not for production reads |

---

## Architecture

### Open path (once per file)

```
hf.zig openFile(repo, file):
  HEAD /resolve/{file} → check X-Xet-Hash
  if !xet:
    fall through to LFS path
  if xet:
    GET /api/.../reconstruction/{file_hash}  → ReconstructResponse
    plan := buildPlan(ReconstructResponse)   // flat []Fragment, no cursor
    handle.xet = .{ plan, url_coalescer, chunk_cache }
```

`buildPlan` walks the reconstruct terms once and produces a flat,
binary-searchable array indexed by logical file offset:

```zig
const Fragment = struct {
    logical_start: u64,      // offset in the decompressed file
    logical_len: u32,        // size in the decompressed file
    xorb_hash: [32]u8,
    chunk_index: u32,        // chunk index within the xorb
    chunk_offset: u32,       // offset within the decoded chunk
    fetch_url_idx: u32,      // index into plan.fetch_urls for this xorb
    url_byte_range: Range,   // HTTP byte range for the chunk (inclusive)
    lz4: bool,
};

const Plan = struct {
    fragments: []Fragment,                       // sorted by logical_start
    fetch_urls: []FetchUrl,                      // unique URLs
    refcount: []u16,                             // per fragment (or per (xorb,chunk))
};
```

Notes:
- `refcount[i]` is computed at plan-build time by counting how many tensor
  reads in the safetensors header overlap fragment `i`. We have that header
  before the first read.
- A xorb that is split across multiple `FetchUrl` entries (see
  `xet_multi_fetchurl_per_xorb.md`) shows up as multiple `Fragment`s with
  different `fetch_url_idx`. No special case in the read path.
- `url_byte_range` uses HTTP inclusive end (see `xet_protocol_caveats.md`).

### Execution model: strict pull, caller-driven concurrency

The entire data flow is driven by the standard VFS API:
`std.Io.Dir.openFile` → `file.reader` → `read` / `readPositional`.
**No push-mode API, no `writeAt`/`writeSlice` calls into the writer.**
The XET layer only implements `readPositional` and returns when `dst`
is fully populated.

Concurrency model:
- **Single `read()` call: sequential.** Walk overlapping fragments in
  logical-offset order; coalesce adjacent fragments under the same
  `FetchUrl` into one HTTP range GET; LZ4-decode; memcpy into `dst`;
  return. No fan-out, no `spawn` inside a read.
- **Network parallelism comes from the caller.** `zml/io.zig` `load()`
  drives many tensor reads concurrently (one `MemoryWriter` per tensor /
  per shard). Each of those is its own sequential pull. N concurrent
  callers → N concurrent CDN requests, naturally.
- **DMA overlap comes for free on direct platforms.** On CUDA/OneAPI the
  `DirectShardWriter` flip-flop pipelines CPU-fill against in-flight
  PJRT `transferData()` at depth 2. Buffered platforms (ROCm, TPU, CPU,
  Neuron) get no such pipelining — same as today's LFS path.

The two file-scoped mechanisms (URL coalescer, refcounted chunk cache)
exist exactly to make this caller-driven concurrency safe and efficient:
when two concurrent `readPositional` calls happen to touch the same
chunk or the same HTTP byte range, they coalesce / share without either
caller needing to know about the other.

### Read path (single `readPositional`, sequential inside)

```
hf.zig readPositional(handle, dst, offset):
  range := [offset, offset + dst.len)
  frags := plan.fragments[plan.firstOverlap(range)..plan.lastOverlap(range)+1]

  for each frag in frags (in order):
    chunk := chunk_cache.acquire(frag.xorb_hash, frag.chunk_index) orelse:
      wire := url_coalescer.fetch(frag.fetch_url_idx, frag.url_byte_range)
      chunk := lz4.decode(wire)  // ByteGrouping4 unshuffle if applicable
      chunk_cache.insert(frag.xorb_hash, frag.chunk_index, chunk)
    copy chunk[frag.chunk_offset .. frag.chunk_offset + frag.logical_len]
         into dst[frag.logical_start - offset ..]
    chunk_cache.release(frag.xorb_hash, frag.chunk_index)

  return dst.len
```

No cursor, no resume state, no plan re-walk between reads. Every tensor
read is independent and idempotent.

### URL coalescer

Purpose: if two tensor reads need fragments that resolve to the **same
HTTP GET** (same `FetchUrl` + same byte range), only one GET is issued;
the second waiter joins.

```zig
const Coalescer = struct {
    in_flight: HashMap(Key, *Pending),  // Key = (fetch_url_idx, range_start, range_end)
};
```

- Holds compressed wire bytes only while the request is in flight.
- Drops to zero memory when the file finishes loading.
- This is the mechanism that catches the Llama "same-xorb revisit"
  pattern (term 2 + term 4 of `5559d307...`).

### Refcounted chunk cache

Purpose: catch the LTX "many small tensors share one chunk" pattern **and**
the cross-tensor boundary case (large A's last chunk == large B's first
chunk) without depending on LRU timing.

```zig
const ChunkCache = struct {
    entries: HashMap(ChunkKey, Entry),   // ChunkKey = (xorb_hash, chunk_index)

    const Entry = struct {
        bytes: []u8,                     // decoded chunk
        refcount: u16,                   // initialised from plan
    };

    fn acquire(self, key) ?[]const u8 {
        // entry stays alive as long as refcount > 0
    }
    fn release(self, key) void {
        // when refcount == 0, free entry
    }
};
```

- Refcounts are seeded from the plan at open time.
- A chunk referenced once by one tensor read is freed as soon as that
  read consumes it — no LRU eviction policy needed.
- A chunk referenced N times (boundary case, small-tensor case, same-xorb
  revisit) stays resident until all N consumers are done.
- **Working set is bounded by genuine sharing**, not by a fixed budget.
- No cap needed in practice; on a 70B BF16 shard the resident chunks
  at any instant are O(parallel_readers × shared_chunks_per_read) ≤
  a few tens of MiB.

#### Optional safety cap (not for v1)

If a worst-case workload ever pushes residency too high, add a soft
`resident_bytes_cap` enforced by **admission control** on new `read()`
calls — never evict a chunk with refcount > 0. Plan-seeded refcounts
make a *hard* cap unsafe: a queued read may be the only future
consumer that will decrement the refcounts pinning the cache, so
blocking that read on the cap can deadlock (resident chunks waiting
for a read that is itself waiting for residency to drop).

One rule prevents this:

> **Always-make-progress override** — if no `read()` is currently
> in flight, admit the head of the admission queue regardless of
> the cap.

That overshoots the cap by at most one read's footprint (a few MB),
and guarantees liveness: as long as someone is running, refcounts
will drop and free chunks; if nobody is running, the override starts
the next one. Treat the cap as soft, not as a hard ceiling.

### Token + auth

Reuse the existing `XetTokenManager` design from `xet.zig`: per
`(repo, model, rev)` token, refresh at `exp - 30s`. Add: on HTTP 403
during a fragment fetch, invalidate the token and the `Plan.fetch_urls`
entries (presigned URLs expire ~1h), re-fetch `/reconstruction`, retry
once. **This is the one failure case not currently handled and easy
to forget.**

---

## What does *not* go into the hot path

- No xorb dedup statistics.
- No cross-file caches.
- No `Handle.XetState` cursor.
- No `WindowCache`.
- No "fall back to LFS if XET is slow" — XET-or-LFS is a per-file decision
  made at open time and stays.

---

## Expected wins vs. LFS

From `bench/`:
- 8B BF16: ~1.4× (cur ~1.39× with all the wrong machinery — close to ceiling).
- 70B AWQ-INT4: ~1.8× (cur ~1.83× — also close to ceiling; AWQ is nearly
  incompressible so the win is purely parallelism + lower per-byte
  HTTP overhead).

The architecture above should match or slightly beat these numbers with
**roughly 1/5 the code** and none of the foot-guns documented in
`xet_protocol_caveats.md` and `xet_multi_fetchurl_per_xorb.md`.

---

## File layout on master

```
zml/io/vfs/
  hf.zig              # unchanged structure; XET branch in dirOpenFile + readPositional
  xet/
    mod.zig           # public surface: detect, openPlan, read
    reconstruct.zig   # HTTP + JSON for /reconstruction
    plan.zig          # Plan, Fragment, buildPlan, binary search
    coalescer.zig     # URL coalescer
    cache.zig         # Refcounted ChunkCache
    lz4.zig           # block decoder + ByteGrouping4
    token.zig         # XetTokenManager
```

Total: ~6 small files. No `XorbCache`, no `WindowCache`, no cursor,
no stats. The complexity that lived in `examples/xet_cas/*` stays in
examples as diagnostic tooling and never enters `zml/io/vfs/`.

---

## What we still need to validate before writing master code

1. **Refcounts from the safetensors header.** We need to confirm we have
   the full per-tensor `(offset, len)` list at the moment we want to seed
   refcounts — i.e. before the first tensor read. In the HF VFS today,
   the safetensors header is read before tensor data, so this should hold;
   verify on a real open.
2. **403 / presigned URL expiry.** Force a stale URL and confirm the
   one-shot re-fetch path works end-to-end.
3. **Coalescer key choice.** Confirm that on Llama-style same-xorb revisit
   the two reads hit the *same* HTTP byte range (and so coalesce) rather
   than two adjacent ranges (which wouldn't).
