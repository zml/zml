# Xet CAS download path — remaining work

Status after `xet: download all tensors of a file in one batched pass`
(branch `oboulant/xet-reconstruction-cleanup`).

Current end-to-end on Llama-3-70B shard 1/30 (15 tensors, 4.4 GiB,
129 xorbs, 282 fetch ranges, 4 workers):

| stage       | time    |
|-------------|---------|
| reconstruct | 1.58 s  |
| plan build  | 0.001 s |
| workers     | 6.01 s  |
| **xet total** | **7.60 s (575 MiB/s)** |
| LFS reference | 23.0 s (190 MiB/s) |

## 1. Multi-shard / whole-repo driver

The biggest remaining fixed cost is the `reconstruct` RTT (≈1.5 s per
shard). For Llama-3-70B that's 30 × 1.5 s = ~45 s if done sequentially.

- Add a top-level binary (or extend `test_file_to_device`) that
  enumerates every `*.safetensors` shard in the repo and runs N
  file-loaders concurrently.
- Share one `xet.Client` and `std.http.Client` across shards.
- Decide on the right concurrency knob: workers-per-shard vs
  shards-in-flight (likely a single shared worker pool that pulls
  xorbs from any shard).

## 2. Integrate into `zml/io.zig::load`

The batched path currently only exists in the test binary. Move it
behind the real entry point so production model loading benefits.

- Wire `zml.io.load(...)` to the batched download.
- Cache `Parsed(ReconstructionResponse)` per file on `xet.Client` so
  repeated lookups (metadata pass + tensor pass, or multiple `load`
  calls for the same file) don't re-hit CAS.

## 3. >10 GB reconstruct chunking

`xet_client.reconstruct(repo, 0, file_size)` is called once. The CAS
API caps a single reconstruct response at 10 GB.

- For any file >10 GB, issue multiple windowed `reconstruct(repo, lo,
  hi)` calls and stitch the term lists + fetch_info maps into one
  combined plan.
- Validate against a monolithic >10 GB shard.

## 4. Plan-loop O(plans × chunks) → O(plans-per-chunk)

The worker's inner loop scans all plans per chunk. Today it's <0.1 ms
because plan count is small, but at ~10k plans × millions of chunks
this becomes the bottleneck.

- Bucket plans by `(xorb_idx, chunk_start)` so each chunk only visits
  the few plans that intersect it.
- Likely a sorted `[]TermPlan` per xorb plus a binary search by
  `chunk_idx`.

## 5. Cleanup & polish

- Drop the LFS verification download from the default path; gate it
  behind `--verify`.
- Reconcile `test_tensor_to_device` (single tensor) with
  `test_file_to_device` (whole file). Once the file path is integrated
  into `zml/io.zig::load`, the per-tensor binary is mostly historical
  — consider deleting it or keeping a single SHA-validation flavour.
- Reset `std_options.log_level` back to `.info` once we stop debugging.

## Priority

1, 2 give the next real user-visible speedup; 3 is correctness for the
few monster shards; 4 and 5 are scaling/polish.
