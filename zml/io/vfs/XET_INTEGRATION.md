# Native XET download for ZML (via xet-core C-API)

Downloads HuggingFace XET-backed files by driving the mature **xet-core** (Rust)
client through its C-API, wired into the existing `hf://` VFS provider.

## Why the C-API

XET files reconstruct client-side (fetch compressed chunks from the CAS, LZ4/BG4
decompress, reassemble). Doing that fast needs adaptive concurrency, bounded
streaming, decompress-once scheduling, connection pooling and an on-disk chunk
cache; xet-core already has all of it. On an m6i.2xlarge, cold download of
`model-00001-of-00004.safetensors` (4.98 GB):

| path | MB/s |
|------|------|
| Zig -> C-API -> xet-core (raw), `HF_XET_HIGH_PERFORMANCE=1` | ~1070 |
| hf-xet (Python, same Rust core) | ~1030 |
| ZML `parallel_read.zig` (plain parallel HTTPS) | ~1000 |
| from-scratch Zig reimpl (jedisct1/zig-xet) | ~386 |

## How it is wired

- `xet_hub.zig`: HF Hub HTTPS/JSON (CAS read-token exchange, file listing).
- `xet_capi.zig`: `extern` declarations of the `hf_xet.h` C ABI (Session +
  downloadToPath). Symbols are declared inline, so only linking `libxet_capi.a`
  is needed, no header include path.
- `xet.zig`: high-level `downloadFile()`.
- `hf.zig`: `performRead` detects XET-backed files on first access and
  reconstructs the whole file once into a local cache via the C-API, then serves
  reads from that local file; non-XET files fall back to the existing `resolve`
  range-GET path. So `hf://` XET files download transparently through the C-API.

Whole-file `download_to_path` means the read path is **eager**: the first read of
a shard pulls the whole shard (download-to-cache), unlike the previous lazy
positional-read behaviour. Set `HF_XET_HIGH_PERFORMANCE=1` for peak concurrency.

## Verification (built + run in ZML's Bazel build)

Built `//examples/io:playground` with Zig 0.16.0 + `libxet_capi` from xet-core's
`assaf/c-api` branch (`cargo build --release -p xet_capi`), on an m6i.2xlarge.

- `xet_capi.zig` compiles and `libxet_capi.a` links into the real ZML binary,
  and `hf.zig` reaches it.
- `playground cp hf://.../model-00001-of-00004.safetensors file://…`
  reconstructed the full 4.98 GB file (exact size) through the integrated path at
  **~228 MB/s**, vs **~23 MB/s** for the current lazy `resolve` path on the same
  `cp` access pattern (~10x). The 228 (below the ~1070 raw) is the cost of the
  download-to-cache disk round-trip (write to `/tmp`, copy out); pointing the
  cache at tmpfs closes most of that gap.

## Known issues / follow-ups

- **Pre-existing crash on exit (not from this change):** `examples/io cp` aborts
  in ZML's own `file://` provider (`file.zig` `getFileHandle` -> `fileClose`,
  from `examples/io/main.zig:96`) after the file is fully written. The stack is
  entirely in the local-file provider close path, unrelated to XET.
- **Cache location / lifecycle:** files are materialized under `/tmp/zml-xet-*`
  and never evicted. Should live under xet-core's own cache dir and be reused.
- **Eager whole-file download** vs the lazy positional-read model; a design
  decision for large shards.
- **Bazel `@xet_capi`** external repo must be declared (prebuilt per platform or
  via rules_rust). The Rust static lib is large and cross-compiled per target,
  which is the one real cost of this approach.
