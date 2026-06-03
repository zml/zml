# XET Push-Based Reconstruction Prototype

## Objective

Validate, in a standalone main (no ZML orchestration), that XET reconstruction
data can be **pushed** chunk-by-chunk directly into per-tensor device buffers,
fetching each xorb at most once even when its bytes feed non-adjacent terms.

The prototype must demonstrate:

- Resolving a model and parsing its safetensors header to obtain tensor
  metadata (name, dtype, shape, file byte range).
- Selecting **two non-contiguous tensors** from the file as the targets.
- Allocating **one PJRT device buffer per target tensor** (full tensor size,
  on-device — no per-tensor host staging buffer).
- Driving the CAS reconstruction API and routing decompressed chunk bytes to
  the correct `(tensor, offset)` destinations.
- Bounded host residency: at most one xorb (≤ 64 MiB) plus a small
  per-chunk scratch slab (≤ 128 KiB) in host memory at any time.
- End-to-end correctness check: read the device buffers back and verify the
  bytes match what a direct safetensors range read would produce.

Out of scope (deliberately deferred):

- `TensorReader`, `DirectMemoryWriter`, `BufferedMemoryWriter`,
  `StreamPlanner`, sharding, mirrored shards, pinned DMA pools, double-
  buffered async transfers, ZML `load()` integration.
- Concurrency / parallel xorb fetches.
- Eviction policies beyond "one xorb at a time".

## Approach (Rephrased Plan)

1. **Pre-pass over terms.** Walk the CAS response `terms[]` once to build a
   `xorb_hash → []TermSlice` map. Each `TermSlice` records every place that
   xorb's bytes are consumed by the target tensors. Non-adjacent reuses of
   the same xorb collapse into a single entry.
2. **Iterate xorbs**, not terms. For each unique xorb:
   1. Download it once into a single 64 MiB host scratch buffer (using
      `fetch_info` from the CAS response).
   2. Walk its chunks linearly with `ChunkIterator`, decompressing each chunk
      into a small reusable slab.
   3. For every chunk, consult the `TermSlice` list to find every target
      term that overlaps it, and push the relevant byte range to that
      term's `(device_buffer, dest_offset)` via PJRT
      `transfer_manager.transferData(offset)`.
3. **Finalize.** Mark `is_last=true` on the final transfer for each tensor
   and wait for completion. Verify device buffer contents against a
   reference read.

Net effect: every xorb is fetched and decompressed exactly once regardless
of how scattered its consumers are in `terms[]`; data lands directly in
device memory; host residency is bounded.

## Implementation Duties

### Reuse as-is from `zml/io/xet.zig`

- `Term`, `ReconstructionResponse` — JSON shapes.
- `buildXorbMap` / `XorbMap` — already produces the
  `xorb → chunk_index → []term_index` lookup the push loop needs.
- `ChunkIterator` — linear xorb walk.
- `decompressChunk` — per-chunk decompression into caller-owned slabs.
- `computeTermRanges` — per-term absolute file offsets.

### Additions to `zml/io/xet.zig`

- Parse `fetch_info` from the CAS reconstruction response (xorb hash →
  download URL + byte range + auth). Currently omitted.
  - `fetch_info` URLs are short-lived presigned CloudFront links
    (`Expires=<epoch>` in the query string, typically minutes to hours).
    They **must** be obtained from a live call to the CAS reconstruction
    endpoint at run time. Saved snapshots like `xet-llama-debug/group_000.json`
    are stale and unusable for downloads — treat them as debug-only.
  - If a download fails with 403/expired, re-issue the reconstruction
    request to refresh `fetch_info` and retry. (For the prototype: assume
    a single in-flight request and don't bother with proactive refresh.)
- Small helper: given a `XorbMap.Entry`, a `chunk_index`, a consuming
  `term_index`, the per-chunk uncompressed sizes seen so far in that xorb,
  and `term_ranges`, return the destination file offset and byte slice to
  push. (Chunks have variable uncompressed size, so maintain a running
  cumulative offset per term while walking the xorb's chunks.)

### Standalone main (`examples/xet_cas/`)

- CLI: model URL + two tensor names (or indices) to target.
- Resolve repo, fetch safetensors header, locate the two tensors' byte
  ranges, request CAS reconstruction for the union of those ranges.
- Create a PJRT client on the available platform; for each target tensor
  call `createBuffersForAsyncHostToDevice(shape)` and keep the returned
  transfer manager + device buffer handle.
- Allocate one 64 MiB host scratch buffer (xorb) and one 128 KiB host slab
  (per-chunk decompression). Reuse both for every xorb / chunk.
- Run the xorb-ordered loop described above; push each decompressed chunk
  range via `transfer_manager.transferData(slice, dest_offset, is_last)`.
- After the last chunk per tensor, wait on the transfer event, then
  `toHostBuffer` the device buffer and compare against a reference
  safetensors range read.

### Verification

- Byte-exact comparison against a reference read of the same ranges from
  the original safetensors file (LFS or local).
- Log: number of xorbs fetched, total bytes downloaded, peak host
  residency, time to device-ready per tensor.
