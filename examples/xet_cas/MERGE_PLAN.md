# XET merge plan — `hugomano/xet` → `oboulant/xet-reconstruction-cleanup`

## Goal

Take what's good from `hugomano/xet` (compression modules, modular
`Reader/Writer` pattern on the fetch + decompress side, `State`-style
caching) and bring it into `oboulant/xet-reconstruction-cleanup` while
**keeping oboulant's algorithm**: two-pass plan-build → xorb-centric
workers with non-monotonic writes.

This document is the single source of truth for the merge. It supersedes
the prior aborted attempt in branch `oboulant/xet-merge`
(see `Risks / things not to do` below).

## Branch state (snapshot at 2026-06-05)

| | merge-base | branch HEAD | commits ahead of base |
|---|---|---|---|
| `hugomano/xet` | `3db711aa` (zml/ops typo) | `d4ddcd18` ("ongoing") | 10 |
| `oboulant/xet-reconstruction-cleanup` | same | `173cd075` | 15 |

The hugomano diff touches **55 files / +7052 LoC** — most of which is
unrelated to XET (`zml/Sharding.zig` 790 lines, `examples/llm/models/*`
refactors, `bin/zml-smi`, etc.). Only ~10 files are actual XET work and
those are the only ones we want to port.

## Architecture map (what's where)

### `hugomano/xet` — split across `zml/io/vfs/`

| File | Purpose |
|---|---|
| `xet.zig` (~1574 LoC) | `State` (per-HF instance), `ReadRequest`, `ReconstructionLoader`, `XetRangeReader`, `XorbFetchWindow`, `XorbRangeCache`, `FrameDecoder`, `Auth` |
| `xet_reconstruction.zig` | JSON parser → `Reconstruction.Index` (terms + fetches, pre-indexed) |
| `xet_stats.zig` | `Stats` snapshot + `AtomicStats` shared counters |
| `lz4.zig` | `BlockReader` |
| `bg4.zig` | `DegroupWriter` + in-place degrouping |
| `hf.zig` | Handle carries optional xet hash; `dirOpenFile` detects via `X-Xet-Hash`; `fileReadPositional` calls `state.read(ReadRequest{...})` |

**Streaming model.** `std.Io.Reader` interface; writes go to
`OutputSlicesWriter (data: []const []u8)` which assumes **monotonic**,
contiguous output.

**Caches keyed by file hash.** `file_hashes`, `tokens`, `reconstructions`
(`*Index`), in-flight `Flight` dedup. Per-call `XorbRangeCache` lives on
`ReadScratch` — chunk reuse only within one read. **No** state-level xorb
byte cache.

### `oboulant/xet-reconstruction-cleanup` — split across `zml/io/` and `examples/xet_cas/`

| File | Purpose |
|---|---|
| [zml/io/xet.zig](../../zml/io/xet.zig) (~567 LoC) | Protocol primitives: `ReconstructionResponse`, `Term`, `FetchUrl`, `ChunkIterator`, `decompressChunk`, `xet.Client` (caches file_id and CAS auth per repo/path) |
| [zml/io/lz4.zig](../../zml/io/lz4.zig) | Standalone LZ4 + `ChunkIterator` dependency |
| [examples/xet_cas/test_file_to_device.zig](test_file_to_device.zig) (~524 LoC) | Main driver: all tensors of one file in a single batched pass |
| [examples/xet_cas/test_tensor_to_device.zig](test_tensor_to_device.zig) | Older per-tensor driver |
| [examples/xet_cas/scan_dedup.zig](scan_dedup.zig) | Dedup statistics scan |
| [examples/xet_cas/util.zig](util.zig) | HTTP range GET + sha256 |

**Algorithm (`test_file_to_device`).**

1. One `reconstruct(repo, 0, file_size)` per file (whole-file plan).
2. **Pass 1** (main thread): walk `resp.terms` → build `plans: []TermPlan`
   + dedup `xorbs: []XorbWork` by xorb_hash; compute per-term
   `head_skip` / `tail_clip` / `bytes_consumed`.
3. **Pass 2** (main thread): for each xorb, collect ALL covering
   `FetchUrl` entries (multi-FetchUrl per xorb is real — Llama-3-70B
   shard 1/30 has xorbs with up to 9 FetchUrl entries).
4. **Pass 3** (workers): atomic counter dispatches xorbs to N workers;
   each worker has its own `std.http.Client` + scratch buffers. Per
   xorb: HTTP range GET → `ChunkIterator` → for each chunk, scan plans,
   decompress once if any plan needs it, write to `Sink` at
   `p.dst_off + p.bytes_written`.
5. Writes are **non-monotonic**: a single xorb fetch fills multiple
   destination slots that are not necessarily adjacent or in order.

## Why a straight merge won't work

- Hugomano's `XetRangeReader` exposes a `std.Io.Reader` whose output
  **must** be monotonic and contiguous (`OutputSlicesWriter`).
- Oboulant's xorb-centric algorithm produces writes to arbitrary
  destination offsets out of order — this is the explicit win
  (one xorb download, many tensor slots filled).
- A previous attempt on branch `oboulant/xet-merge` (since deleted)
  bolted hugomano's reader onto whole-file plans without a state-level
  xorb byte cache, producing a **2× network regression** with 3× wall
  time on Repo A. The state-level cache is what saves you when the
  reader's per-call `XorbRangeCache` can't see across calls. Conclusion:
  do not ship a partial integration that disables oboulant's xorb
  pooling.

## Spliceable boundary (what we can reuse from hugomano)

| Layer | Source | Reason |
|---|---|---|
| LZ4 block decode | hugomano `lz4.zig` (`BlockReader`) | Cleaner, more factored |
| BG4 degrouping | hugomano `bg4.zig` | Dedicated module, more tested |
| `Reconstruction.Index` parser | hugomano `xet_reconstruction.zig` | Richer types (`FetchEntry`/`TermEntry`/`Range`) |
| `AtomicStats` / `Stats` | hugomano `xet_stats.zig` | Nice instrumentation |
| `Auth` + token cache w/ TTL | hugomano `Auth` + `State.tokens` | Oboulant cache is forever |
| File-hash cache | hugomano `State.file_hashes` | Needs HF VFS hook to populate |
| Reconstruction `Index` cache | hugomano `State.reconstructions` | Whole-file plan reuse |
| Plan building (2-pass) | **oboulant** | Xorb-centric, deduped |
| Xorb worker pool | **oboulant** | Non-monotonic writes |
| Multi-FetchUrl-per-xorb picker | **oboulant** | Already correct |
| Decompress-once-per-chunk | **oboulant** | Core dedup mechanism |
| `OutputSlicesWriter` / `XetRangeReader` | **drop on oboulant side** | Clashes with non-monotonic writes |
| `XorbFetchWindow` (prefetch) | hugomano | Can wrap oboulant worker fetch |

## Confirmed decisions (2026-06-05)

- **Branch name:** `oboulant/xet-merge` (previously deleted; reused).
- **Start with Phase 0** (baseline lock, no code).
- **Phase 4 IN** — needed for [TODO.md](TODO.md) item 2 (integrate into
  `zml/io.zig::load`).
- **Phase 2 DEMOTED to optional** — Pass-1/Pass-2 input types are not on
  the critical path. Revisit only if it materially simplifies Phase 4.

## Phased plan

### Phase 0 — Lock the baseline (no merge, no code)

- Record current SHA-256s and wall-clock for the regression workloads on
  **both branches** so we can detect regressions cleanly later.
- Workloads:
  - `meta-llama/Meta-Llama-3-70B` `model-00030-of-00030.safetensors`
    (`lm_head` shard, low dedup).
  - `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` file containing
    `q_proj.qweight` (high dedup).
- For `hugomano/xet`: rerun `bench/{A,B}.{lfs,xet}.log` measurements per
  the baseline note.
- Deliverable: a one-screen comparison table.

### Phase 1 — Compression-layer ports (oboulant ← hugomano)

Pure refactors. No behavior change. Validate with regression matrix
after each commit.

1.1 Replace [zml/io/lz4.zig](../../zml/io/lz4.zig) (361 LoC) with
    hugomano's `lz4.zig` (206 LoC, `BlockReader`). Adapt
    `ChunkIterator.decompressChunk` callers in
    [zml/io/xet.zig](../../zml/io/xet.zig) to use the new API.

1.2 Extract BG4 unshuffle from oboulant's `decompressChunk` into a new
    `zml/io/bg4.zig` (port from hugomano's `bg4.zig`). Keep
    `decompressChunk` as a thin wrapper for BG4+LZ4 / FBS+LZ4 cases.

1.3 Add `zml/io/xet_stats.zig` from hugomano. Wire `AtomicStats` into
    `test_file_to_device` worker (replace ad-hoc accumulators).

### Phase 2 — Promote the Reconstruction parser (OPTIONAL)

Pure type/parse refactor. **Not** about the number of `reconstruct()`
API calls — oboulant already does ONE call per file since `c08001af`.
Only the in-memory representation changes:

- **From** oboulant's flat
  `ReconstructionResponse { terms, fetch_info: map<hash, []FetchUrl> }`
- **To** hugomano's pre-indexed
  `Reconstruction.Index { terms: []TermEntry(.fetch_index), fetches: []FetchEntry }`

Multi-FetchUrl bookkeeping moves into the parser; worker Pass-2
simplifies.

2.1 Add `zml/io/xet_reconstruction.zig` (port from hugomano).
2.2 In `zml/io/xet.zig`, switch oboulant's Pass-1 input types over.
    Algorithm unchanged.
2.3 (optional) Add `parseIndexBody` test coverage to `//zml/io:test`.

**Revisit after Phase 3.** Only do it if it materially simplifies
Phase 4.

### Phase 3 — Promote `xet.Client` to `xet.State` (oboulant improvement)

This is "do oboulant's [TODO.md](TODO.md) items, using hugomano's
`State` struct as a template". The only genuinely new functionality
(beyond what TODO.md already calls out) is **in-flight dedup**
(`Flights`) — needed before any multi-shard driver can launch
concurrent shard loaders without each one firing its own auth-token
request.

3.1 Refactor `xet.Client` → `xet.State` (hugomano shape):
- `file_hashes` (already present)
- `tokens` with `exp: i64` TTL (today: cached forever)
- `reconstructions: StringHashMap(*Index)` keyed by xet hash
  (TODO.md item 2, second half)
- `auth_flights` + `reconstruction_flights` for concurrent-request
  dedup (**NEW** — needed for TODO.md item 1)

3.2 Keep `Client.fileId` / `.casAuth` / `.reconstruct` entrypoints as
    thin wrappers initially so `test_file_to_device` keeps compiling.

3.3 Add `getOrFetchIndex(repo, file_size) -> *const Reconstruction.Index`
    using the `Flight` pattern. **Mind the `.alloc_always`
    `parseFromSlice` gotcha** — default is `.alloc_if_needed` for
    Scanner-backed input and the source body is freed before `Parsed`
    is returned, dangling strings otherwise.

### Phase 4 — Wire HF VFS + `zml/io.zig::load` integration

Implements TODO.md item 2 ("Wire `zml.io.load(...)` to the batched
download").

4.1 Port hugomano's `hf.zig` change to populate `xet.State.file_hashes`
    on `dirOpenFile` via the `X-Xet-Hash` header. **Do not** plumb
    `XetRangeReader` into `fileReadPositional` — incompatible with
    oboulant's non-monotonic writes.

4.2 Add a VFS-level batch API to bridge the monotonic-pread interface
    with oboulant's xorb-centric executor. Options to evaluate:

- **(a)** `prefetchTensorRanges(file_uri, []TensorRange)` that
  materializes all ranges into a per-file scratch buffer; subsequent
  `fileReadPositional` calls serve from that buffer.
- **(b)** `loadFileToWriters(file_uri, [](range → MemoryWriter))` that
  skips the intermediate buffer entirely and writes straight into
  device sinks from the xorb worker pool.

Pick **(a)** for the first pass — keeps `zml/io.zig::load` changes
minimal. **(b)** is a Phase-5+ optimization.

4.3 Hook `zml/io.zig::load` to detect xet-eligible files (via
    `state.hasFileHash`) and call the batch API before the per-tensor
    streaming loop. Fall back to LFS path on miss.

### Phase 5 — Validate end-to-end

- Run regression matrix.
- Add multi-shard run (Llama-3-70B all 30 shards) — measures Phase-3
  `Flight` dedup + Phase-4 integration.
- Compare against Phase 0 baseline. Any regression > 5% blocks the
  phase.

## Risks / things to NOT do

- **Do not** bring hugomano's `XetRangeReader` / `OutputSlicesWriter` —
  they assume monotonic output. Oboulant's algorithm writes
  non-monotonically.
- **Do not** flip a single "use whole-file plan + per-call
  `XorbRangeCache`" switch — that is exactly the failure mode from the
  previous `oboulant/xet-merge` attempt (2× network, 3× wall time).
- **Do not** pull in unrelated hugomano commits: `zml/Sharding.zig`
  (790-line diff), `examples/llm/models/*` refactors,
  `bin/zml-smi` changes, etc. They are unrelated to XET and would
  balloon the merge surface.
- **Beware:** hugomano touches `zml/io.zig` (+197 lines, streaming
  pipeline). Oboulant touches `zml/io/io.zig` (different file, +1).
  Keep changes scoped to the `zml/io/` subdirectory and avoid
  hugomano's `zml/io.zig` edits unless Phase 4 actually needs them.

## Commit cadence and validation

- One commit per sub-step (1.1, 1.2, 1.3, 2.1, …) so any single step
  can be reverted without losing the rest.
- After each commit:
  - `bazel test //zml/io:test //zml:test`
  - Llama-3-70B `lm_head` shard regression run (workers=4)
  - AWQ-INT4 `q_proj.qweight` regression run (workers=4 + workers=1,
    confirm SHA match)

## Cross-reference: oboulant's [TODO.md](TODO.md)

`examples/xet_cas/TODO.md` lists 5 remaining items. Mapping to phases:

| TODO item | Phase | Notes |
|---|---|---|
| 1. Multi-shard / whole-repo driver | Phase 3 prereq, then trivial | Needs `Flight` dedup so 30 concurrent shard loaders share one auth fetch |
| 2. Integrate into `zml/io.zig::load` | Phase 4 | Second half ("cache `Parsed(ReconstructionResponse)` per file") is Phase 3.1 |
| 3. >10 GB reconstruct chunking | Independent | Do whenever; not gated by merge |
| 4. Plan-loop O(plans × chunks) → O(per-chunk) | Falls out of Phase 2 | Hugomano `TermEntry.fetch_index` already maps term→fetch |
| 5. Cleanup & polish | Post Phase 5 | Logging, gating `--verify`, deleting `test_tensor_to_device` |
