# Protocol for XET file reconstruction from HF

The objective is not to materialize the whole file either on the disk or in memory. 
We want ultimatetly to send data as theyr arrive and stream them into `std.io.Reader` via `std.io.Writer`. This document is not about how we send data to device memory, but rather about how to retrieve the data so we make this possible without materializing the whole file. 

## Protocol facts

1. **Xorbs have no header.** Flat stream: `[8-byte ChunkHeader | CompressedData]` repeated. Chunks and xorbs are variable size.
2. **A term = (xorb_hash, chunk_range \[start, end))** — ordered, concatenating decoded term outputs reconstructs the file byte range.
3. **Deduplication is general.** The same `(xorb_hash, chunk_index)` can appear in multiple terms at arbitrary positions in the file. One decompression should serve all destinations.
4. **fetch_info is not 1:1 with terms.** One fetch_info entry can encompass multiple terms' chunk ranges. Each fetch_info entry = one HTTP GET with `Range: bytes={url_range.start}-{url_range.end}`.
5. **`offset_into_first_range`** — bytes to skip from decoded output of the first term's chunks (can span multiple chunks).
6. **Groups are our artifact** — arbitrary 10GB batches of Range requests to the reconstruction API.

## What CAN be precomputed (from reconstruction responses + safetensor header)

- **Term → file byte range** `[file_start, file_end)` — cumulative `unpacked_length` + group range offset - `offset_into_first_range`
- **Term → candidate tensors** — interval intersection of term's file range with sorted tensor list
- **(xorb_hash, chunk_index) → list of terms that reference it** — so when a chunk is decoded, we know which terms to serve

## What CANNOT be precomputed

- **Chunk → exact file byte position** — depends on `uncompressed_size` of all preceding chunks within each term, only available from chunk headers after downloading
- **Chunk → tensor destinations** — follows from above

## The complete process

### Phase 1: Prepare (before any download)

1. Fetch the safetensor header → get tensor list with `[data_start, data_end)` per tensor.
2. Call the reconstruction API (per group) → get `terms[]`, `fetch_info{}`, `offset_into_first_range`.
3. Compute each term's file byte range from cumulative `unpacked_length`.
4. Build the **xorb map**: for each xorb, collect all terms referencing it and the chunk ranges they need.

### Phase 2: Download and route (per xorb, parallelizable across xorbs)

For each xorb in the xorb map:

1. **Initialize one byte counter per term** referencing this xorb, all set to 0.
2. **Download** the xorb data via its fetch_info URL(s) + Range header.
3. **Iterate chunks sequentially** (forced by the binary format — each 8-byte header gives compressed_size to find the next chunk):
    For chunk index N:
    - Look up which terms need chunk N (precomputed in step 4 above).
    - If none: skip the compressed data (advance read cursor by compressed_size), don't decompress.
    - If any: decompress once, then for each term that needs it:
        - Compute: `file_pos = term.file_start + term.byte_counter`
        - Find which tensor(s) overlap `[file_pos, file_pos + uncompressed_size)` (binary search into term's candidate tensors).
        - Memcpy the relevant byte slice into each tensor's host buffer at the correct offset.
        - Advance: `term.byte_counter += uncompressed_size`

### Handling `offset_into_first_range`

`offset_into_first_range` means the first N bytes of the first term's decoded output don't belong to this group's file range — they are a leftover tail from the previous group's last term (or padding). In practice:

- When computing `term.file_start` for the first term of a group, subtract `offset_into_first_range` from the group's range_start. This makes the first N decoded bytes map to file positions *before* the group's range, so they'll fall outside any tensor in this group and be naturally skipped during the "find which tensor" step.
- Alternatively, initialize the first term's byte_counter to `offset_into_first_range` instead of 0, effectively skipping those bytes. The result is the same: the first routable byte lands at the correct file position.

### Why it works

- Chunks within a single xorb are iterated in order (0, 1, 2, ...). This is the only ordering constraint.
- Each term's byte counter only advances for chunks within that term's range. By the time we reach a shared/overlapping chunk, all preceding chunks in each term's range have already been processed, so every counter is correct.
- Xorbs are independent — process them in any order or in parallel.
- Each chunk is decompressed exactly once, then routed to all its destinations.

### Order rule

Process fetch_info entries sorted by range.start (ascending). This guarantees that if a term's chunk range spans multiple fetch_info entries, its byte counter accumulates correctly across entries.

In practice (from both LTX and Llama data), each individual term's chunk range fits entirely within a single fetch_info entry — but sorting by range.start is the safe general rule and costs nothing.

```
fetch_info entries for one xorb (sorted): [0,5), [8,20), [26,43), ...

HTTP GET #1 → raw bytes for chunks 0,1,2,3,4
  → iterate sequentially, route each chunk

HTTP GET #2 → raw bytes for chunks 8,9,...,19
  → iterate sequentially, route each chunk

HTTP GET #3 → raw bytes for chunks 26,27,...,42
  → iterate sequentially, route each chunk
```

## Implementation steps

### Step 1: Compute term → file byte range

- Input: reconstruction JSON (terms[] + offset_into_first_range + group range_start)
- Output: for each term, `[file_start, file_end)`
- Test: verify `sum(all term lengths) - offset_into_first_range == group_range_end - group_range_start`. Verify term ranges are contiguous (term[i].file_end == term[i+1].file_start).

### Step 2: Build xorb map (chunk_index → list of terms)

- Input: terms[] from reconstruction JSON
- Output: `HashMap(xorb_hash, { terms: []Term, chunk_to_terms: [][]TermRef })`
- Test: verify every term's chunk range [start, end) is fully covered in the map. Verify total chunk references == sum of all term range lengths.

### Step 3: Parse a single xorb binary stream

- Input: raw bytes from one HTTP GET (one fetch_info entry)
- Output: iterator yielding `(chunk_index, uncompressed_size, compressed_size, compression_type, compressed_data_slice)`
- Test: download one small fetch_info entry, verify number of parsed chunks == `range.end - range.start`. Verify sum of `(8 + compressed_size)` per chunk == total downloaded bytes.

### Step 4: Decompress a single chunk

- Input: `(compression_type, compressed_data_slice)`
- Output: decompressed byte slice
- API: `decompressChunk(chunk, dst, tmp) ![]u8` — zero-alloc, caller provides buffers.
- Max chunk size is 128 KiB (from [HF xet chunking spec](https://huggingface.co/docs/xet/en/chunking): target 64K, min 8K, max 128K).
- Hot path pattern: allocate `dst` and `tmp` once at 128 KiB per xorb iteration, reuse for every chunk.
- Types: 0=None (memcpy), 1=LZ4, 2=BG4+LZ4 (needs tmp), 3=FBS+LZ4 (needs tmp).
- Test: verify `decompressed.len == uncompressed_size` from header. For compression_type 0 (None), output == input.

### Step 5: Full xorb decode — reconstruct file byte ranges

- Input: xorb binary data + xorb map for this xorb
- Output: for each term, its complete decoded byte range (concatenation of all its chunks' decompressed outputs)
- Test: verify each term's total decoded length == `term.unpacked_length` from the reconstruction JSON. This validates the counter logic end-to-end without needing tensor routing.

### Step 6: Route decoded bytes to tensor buffers

- Input: decoded chunk + `file_pos` + tensor list sorted by data_start
- Output: memcpy into correct tensor buffer(s) at correct offset
- Test: reconstruct a small safetensors file entirely, compare SHA256 against `huggingface-cli download` of the same file. Or: reconstruct just one known tensor, compare against `safetensors.load()` in Python.

### Implementation order

| Order | Step | Why this order |
|-------|------|---------------|
| 1 | Step 1 | Pure arithmetic, no I/O, validates understanding of the JSON format |
| 2 | Step 2 | Pure data structure, no I/O, needed by everything downstream |
| 3 | Step 3 | First real I/O — download one fetch_info entry, parse chunk headers |
| 4 | Step 4 | LZ4 + ByteGrouping4 decompression (need to bring in a dependency or write it) |
| 5 | Step 5 | Combines 3+4 with the counter logic. Verifiable without tensor routing. |
| 6 | Step 6 | Final integration: route to tensor buffers. End-to-end correctness. |

Steps 1-2 can be tested today with the debug JSON. Step 3 requires one HTTP request. Step 4 needs an LZ4 decoder. Step 5 is where the "does it actually reconstruct correctly?" validation happens. Step 6 adds the tensor dimension.

## Dependency: zig-xet

We use [jedisct1/zig-xet](https://github.com/jedisct1/zig-xet) (MIT license) for xorb parsing and chunk decompression. It's a pure Zig implementation cross-verified against the Rust reference.

### What we take from zig-xet

| Our step | zig-xet file | What it provides |
|----------|-------------|-----------------|
| Step 3 | `xorb.zig` | `ChunkHeader` (extern struct, 8 bytes), `XorbReader.nextChunk()` |
| Step 4 | `compression.zig` | `decompress()` — handles None, LZ4, ByteGrouping4LZ4, **and** FullBitsliceLZ4 (a 4th type we hadn't accounted for) |
| Step 4 | `lz4` (transitive dep) | LZ4 frame decompression |
| — | `constants.zig` | `CompressionType` enum, `XorbChunkHeaderSize`, `XorbVersion` |

### What we write ourselves

- Routing layer (byte counters, tensor mapping) — ZML-specific, doesn't exist in zig-xet
- Integration with ZML's `MemoryWriter` / PJRT buffer pipeline
- HTTP fetch orchestration (already in `examples/xet_cas/main.zig`)

### Integration: vendor into third_party

Vendor the source files into `third_party/zig-xet/` with a `BUILD.bazel` target, following the existing pattern of other third_party Zig dependencies in this repo. Files to vendor:

```
third_party/zig-xet/
  src/xorb.zig
  src/compression.zig
  src/constants.zig
  src/hashing.zig          (dependency of xorb.zig)
  BUILD.bazel              (zig_library target exposing the above)
  lz4/                     (transitive dependency — zig-xet uses a zig lz4 package)
```

This avoids build system conflicts (zig-xet uses `build.zig`, we use Bazel) and gives us pinned, auditable source.

### Impact on implementation steps

- **Step 3** becomes: use vendored `XorbReader` directly (or adapt its `nextChunk()` into a streaming iterator that doesn't require the full xorb in memory).
- **Step 4** becomes: call vendored `compression.decompress()`. No custom LZ4/BG4 code needed.
- **Steps 1, 2, 5, 6** are unchanged — they're our routing logic.

## File organization

```
third_party/zig-xet/           — Vendored xorb parsing + decompression from zig-xet
zml/io/xet.zig                 — Core reconstruction logic (Steps 1, 2, 5, 6)
examples/xet_cas/main.zig     — Keep as-is (fetches reconstruction JSON for debugging)
```

| File | Contents | Depends on |
|------|----------|-----------|
| `third_party/zig-xet/` | Vendored `ChunkHeader`, `XorbReader`, `compression.decompress()`, LZ4 | nothing (self-contained) |
| `zml/io/xet.zig` | `Term`, `XorbMap`, `computeTermRanges()`, `buildXorbMap()`, `processXorb()` (the byte-counter loop), tensor routing | third_party/zig-xet |
| `examples/xet_cas/main.zig` | Existing code (HTTP fetch, auth, safetensor header parsing, reconstruction API calls, JSON output) | unchanged |

### Rationale

- Vendoring zig-xet gives us battle-tested xorb parsing and all 4 compression types without writing any decompression code.
- `xet.zig` is the orchestrator — it ties everything together. The `processXorb()` function implements the byte-counter algorithm using vendored decompression primitives.

### Integration point (later)

`zml/io.zig` already has module loading. When ready, add an XET backend there that:
1. Calls reconstruction API (move HTTP logic from example)
2. Calls `xet.computeTermRanges()` + `xet.buildXorbMap()`
3. Spawns N download threads, each calling `xet.processXorb()` writing directly into the `MemoryWriter` tensor buffers

## Coding guidelines

### Principles

- **Write as little code as possible.** Minimal surface area. No speculative abstractions.
- **Use `std.Io.Reader` and `std.Io.Writer` for all IO.** This is how ZML does streaming — vtable-based interfaces that compose (see `TensorReader`, `ProgressWriter`, `MemoryWriter` in `zml/io.zig`).
- **Do as little allocation as possible.** Prefer fixed buffers, stack allocation, and streaming over collecting into ArrayLists.

### Repo conventions (observed from zml/)

- **Vtable pattern for IO interfaces:** Implement `std.Io.Reader` or `std.Io.Writer` with a struct that holds state + a vtable. Use `@fieldParentPtr` to recover `self` from the interface pointer (see `TensorReader.stream`, `DirectShardWriter.drain`).
- **`std.Io` threading:** Use `std.Io.Group` and `stdx.Io.LimitedGroup` for bounded concurrency — not raw threads.
- **No intermediate files.** Data flows reader → writer. `MemoryWriter` streams directly into device memory (CUDA: DMA, others: buffered then transferred).
- **`FixedBufferAllocator`** for small temporary buffers (path joins, formatting) instead of heap.
- **Scoped log:** `const log = std.log.scoped(.@"zml/io");`
- **Error handling:** Return errors up, don't panic. Use `errdefer` for cleanup.
- **Naming:** `PascalCase` types, `lowerCamelCase` functions/fields. Public API at top of struct, private helpers below.
- **Tests:** Inline `test "..." {}` blocks at bottom of file.
